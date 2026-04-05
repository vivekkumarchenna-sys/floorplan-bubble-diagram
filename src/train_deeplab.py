"""
train_deeplab.py — DeepLabV3+ training for 16-class semantic segmentation
=========================================================================
Backbone   : ResNet-101 pretrained on COCO  (torchvision)
Decoder    : DeepLabV3+ (ASPP + low-level features from layer1)
Input size : 512 × 512
Classes    : 16
Loss       : weighted CrossEntropy + 0.5 × Dice
Optimiser  : AdamW  lr=6e-5  weight_decay=0.01
Schedule   : CosineAnnealingLR  T_max=100  eta_min=1e-7
Early stop : patience=15  (monitor val mIoU)
Logs       : JSON  (one record per epoch)
Checkpoint : best val-mIoU  →  Drive / local output dir

Colab quick-start
-----------------
    from google.colab import drive
    drive.mount('/content/drive')
    !pip install -q torch torchvision albumentations opencv-python-headless
    !python train_deeplab.py

Directory layout expected (configure via CFG)
---------------------------------------------
    data/
      train/images/  *.png
      train/masks/   *.png  (values 0-15; 255 = ignore)
      val/images/    *.png
      val/masks/     *.png
"""

# ── stdlib ───────────────────────────────────────────────────────────────────
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ── third-party ──────────────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.models as tvm
import torchvision.models.segmentation as tvseg

# ── local ─────────────────────────────────────────────────────────────────────
from dataset import ResPlanSegDataset


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                        GLOBAL CONFIGURATION                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class CFG:
    # ── data ──────────────────────────────────────────────────────────────────
    # Project root = the folder containing this script (works on Drive & locally).
    ROOT          = Path(__file__).resolve().parent

    DATA_ROOT     = ROOT / "data"
    IMG_DIR       = DATA_ROOT / "resplan_raster"
    MSK_DIR       = DATA_ROOT / "resplan_masks"
    TRAIN_SPLIT   = DATA_ROOT / "splits" / "train.txt"
    VAL_SPLIT     = DATA_ROOT / "splits" / "val.txt"

    # ── model ─────────────────────────────────────────────────────────────────
    # backbone: "resnet50" | "resnet101"  (both available in torchvision)
    BACKBONE      = "resnet101"
    NUM_CLASSES   = 16
    IMG_SIZE      = 512
    # output_stride: 8 or 16 (16 is faster; 8 gives finer predictions)
    OUTPUT_STRIDE = 16

    # ── training ──────────────────────────────────────────────────────────────
    EPOCHS        = 50
    BATCH_SIZE    = 16         # 16 for A100-40GB, 24-32 for A100-80GB
    NUM_WORKERS   = 8          # 0 on Windows or if multiprocessing hangs
    PIN_MEMORY    = True

    LR            = 6e-5
    WEIGHT_DECAY  = 0.01
    LR_MIN        = 1e-7

    DICE_WEIGHT   = 0.5        # total loss = CE + DICE_WEIGHT * Dice

    # Per-class CE weights (16 floats) or None for uniform.
    CLASS_WEIGHTS = None

    IGNORE_INDEX  = 255

    # ── early stopping ────────────────────────────────────────────────────────
    ES_PATIENCE   = 15

    # ── misc ──────────────────────────────────────────────────────────────────
    SEED          = 42
    AMP           = True       # fp16 mixed precision
    GRAD_CLIP     = 1.0        # max gradient norm (None to disable)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                         DEEPLAB V3+ MODEL                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class _ASPPConv(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, dilation: int):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=dilation,
                      dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class _ASPPPool(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(self.gap(x), size=x.shape[-2:],
                             mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling (DeepLabV3+)."""

    def __init__(self, in_ch: int, out_ch: int = 256, output_stride: int = 16):
        super().__init__()
        rates = (6, 12, 18) if output_stride == 16 else (12, 24, 36)

        self.branches = nn.ModuleList([
            nn.Sequential(                                    # 1×1 conv
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ),
            _ASPPConv(in_ch, out_ch, rates[0]),
            _ASPPConv(in_ch, out_ch, rates[1]),
            _ASPPConv(in_ch, out_ch, rates[2]),
            _ASPPPool(in_ch, out_ch),
        ])
        self.proj = nn.Sequential(
            nn.Conv2d(out_ch * 5, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(torch.cat([b(x) for b in self.branches], dim=1))


class DeepLabV3Plus(nn.Module):
    """
    DeepLabV3+ with a ResNet-50 or ResNet-101 backbone.

    The encoder uses dilated convolutions at layer3/layer4 so the backbone
    never needs to be retrained from scratch — torchvision COCO weights fit.

    low-level features  : layer1 output  →  48-ch projection  (1/4 scale)
    high-level features : ASPP on layer4 output               (1/OS scale)
    decoder             : concat → 3×3 conv → bilinear ×4 → classifier
    """

    _BACKBONE_OUT_CH = {"resnet50": 2048, "resnet101": 2048}
    _LOW_LEVEL_CH    = {"resnet50": 256,  "resnet101": 256}

    def __init__(
        self,
        num_classes: int,
        backbone: str = "resnet101",
        output_stride: int = 16,
        pretrained: bool = True,
    ):
        super().__init__()

        if backbone not in self._BACKBONE_OUT_CH:
            raise ValueError(f"backbone must be resnet50 or resnet101, got {backbone!r}")

        # ── load torchvision backbone ─────────────────────────────────────────
        weights = (tvm.ResNet101_Weights.IMAGENET1K_V2 if backbone == "resnet101"
                   else tvm.ResNet50_Weights.IMAGENET1K_V2) if pretrained else None
        _rn = (tvm.resnet101 if backbone == "resnet101" else tvm.resnet50)(weights=weights)

        # Adapt layer3/layer4 for dilated convolutions (DeepLab trick).
        # layer3: stride 2 → stride 1, dilation 2
        # layer4: stride 2 → stride 1, dilation 2 or 4  (OS-16 / OS-8)
        s3_dil, s4_dil = (1, 2) if output_stride == 16 else (2, 4)
        _patch_stride(_rn.layer3, stride=1, dilation=s3_dil)
        _patch_stride(_rn.layer4, stride=1, dilation=s4_dil)

        self.stem     = nn.Sequential(_rn.conv1, _rn.bn1, _rn.relu, _rn.maxpool)
        self.layer1   = _rn.layer1   # 1/4  →  low-level features
        self.layer2   = _rn.layer2   # 1/8
        self.layer3   = _rn.layer3   # 1/8  (OS-16 mode)
        self.layer4   = _rn.layer4   # 1/8  (OS-16 mode)

        # ── encoder head (ASPP) ───────────────────────────────────────────────
        self.aspp = ASPP(
            in_ch=self._BACKBONE_OUT_CH[backbone],
            out_ch=256,
            output_stride=output_stride,
        )

        # ── low-level feature projection ─────────────────────────────────────
        self.low_proj = nn.Sequential(
            nn.Conv2d(self._LOW_LEVEL_CH[backbone], 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        # ── decoder ───────────────────────────────────────────────────────────
        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(256, num_classes, 1)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]

        x        = self.stem(x)
        low_feat = self.layer1(x)           # 1/4
        x        = self.layer2(low_feat)
        x        = self.layer3(x)
        x        = self.layer4(x)

        x = self.aspp(x)                    # 256 ch, 1/OS
        # upsample to 1/4 scale to match low-level features
        x = F.interpolate(x, size=low_feat.shape[-2:],
                          mode="bilinear", align_corners=False)

        low_feat = self.low_proj(low_feat)  # 48 ch, 1/4
        x = torch.cat([x, low_feat], dim=1)    # 304 ch
        x = self.decoder(x)                     # 256 ch

        # upsample to original resolution
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
        return self.classifier(x)               # (B, num_classes, H, W)


def _patch_stride(layer: nn.Module, stride: int, dilation: int) -> None:
    """
    Replace stride-2 convolutions in a ResNet bottleneck layer with
    stride-1 + dilation, keeping all other layer weights intact.
    Applied in-place; preserves pretrained parameters.
    """
    for m in layer.modules():
        if isinstance(m, nn.Conv2d):
            if m.stride == (2, 2):
                m.stride  = (stride, stride)
            if m.kernel_size == (3, 3):
                m.dilation = (dilation, dilation)
                m.padding  = (dilation, dilation)
        elif isinstance(m, nn.MaxPool2d):
            m.stride = stride


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                              LOSS                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class DiceLoss(nn.Module):
    """Soft multiclass Dice loss."""

    def __init__(self, num_classes: int, ignore_index: int = 255, eps: float = 1e-6):
        super().__init__()
        self.C            = num_classes
        self.ignore_index = ignore_index
        self.eps          = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        valid     = (targets != self.ignore_index)
        targets_c = targets.clone()
        targets_c[~valid] = 0

        probs   = F.softmax(logits, dim=1)
        one_hot = F.one_hot(targets_c, self.C).permute(0, 3, 1, 2).float()

        vm      = valid.unsqueeze(1).float()
        probs   = probs   * vm
        one_hot = one_hot * vm

        dims  = (0, 2, 3)
        inter = (probs * one_hot).sum(dims)
        denom = (probs + one_hot).sum(dims)
        return 1.0 - ((2.0 * inter + self.eps) / (denom + self.eps)).mean()


class SegLoss(nn.Module):
    """CE + α·Dice."""

    def __init__(
        self,
        num_classes: int,
        class_weights: torch.Tensor | None = None,
        dice_weight: float = 0.5,
        ignore_index: int = 255,
    ):
        super().__init__()
        self.ce   = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)
        self.dice = DiceLoss(num_classes, ignore_index=ignore_index)
        self.dw   = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.ce(logits, targets) + self.dw * self.dice(logits, targets)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                            METRICS                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class StreamingMIoU:
    """Confusion-matrix–based mIoU, accumulated across batches."""

    def __init__(self, num_classes: int, ignore_index: int = 255):
        self.C      = num_classes
        self.ignore = ignore_index
        self.reset()

    def reset(self):
        self.mat = torch.zeros(self.C, self.C, dtype=torch.long)

    @torch.no_grad()
    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        preds   = preds.view(-1).cpu()
        targets = targets.view(-1).cpu()
        valid   = targets != self.ignore
        preds   = preds[valid].clamp(0, self.C - 1)
        targets = targets[valid]
        idx     = targets * self.C + preds
        self.mat += torch.bincount(idx, minlength=self.C * self.C).reshape(self.C, self.C)

    def compute(self) -> dict:
        mat   = self.mat.float()
        inter = mat.diag()
        union = mat.sum(1) + mat.sum(0) - inter
        iou   = inter / union.clamp(min=1)
        miou  = iou[union > 0].mean().item()
        return {"mIoU": round(miou, 6), "iou_per_class": iou.tolist()}


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                         TRAINING ENGINE                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: SegLoss,
    metric: StreamingMIoU,
    optimiser=None,
    scaler=None,
    device: torch.device = torch.device("cpu"),
    amp: bool = True,
    epoch: int = 0,
    tag: str = "",
) -> dict:
    training = optimiser is not None
    model.train() if training else model.eval()
    metric.reset()

    total_loss = 0.0
    n_batches  = 0

    desc = f"Epoch {epoch:03d} {tag}" if epoch else tag
    pbar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for batch in pbar:
            images = batch["image"].to(device, non_blocking=True)  # (B,3,H,W)
            masks  = batch["mask"].to(device, non_blocking=True)   # (B,H,W)

            with torch.autocast(device.type, enabled=amp):
                logits = model(images)                             # (B,C,H,W)
                loss   = criterion(logits, masks)

            if training:
                optimiser.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    if CFG.GRAD_CLIP:
                        scaler.unscale_(optimiser)
                        nn.utils.clip_grad_norm_(model.parameters(), CFG.GRAD_CLIP)
                    scaler.step(optimiser)
                    scaler.update()
                else:
                    loss.backward()
                    if CFG.GRAD_CLIP:
                        nn.utils.clip_grad_norm_(model.parameters(), CFG.GRAD_CLIP)
                    optimiser.step()

            metric.update(logits.argmax(dim=1), masks)
            total_loss += loss.item()
            n_batches  += 1

            avg_loss = total_loss / n_batches
            pbar.set_postfix(loss=f"{avg_loss:.4f}")

    stats         = metric.compute()
    stats["loss"] = round(total_loss / max(n_batches, 1), 6)
    return stats


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                              MAIN                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def main():
    # ── reproducibility ───────────────────────────────────────────────────────
    torch.manual_seed(CFG.SEED)
    np.random.seed(CFG.SEED)

    # ── performance optimisations ─────────────────────────────────────────────
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    # ── output directories ────────────────────────────────────────────────────
    ckpt_dir = CFG.ROOT / "checkpoints" / "deeplab"
    log_dir  = CFG.ROOT / "results"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[init] Checkpoints : {ckpt_dir}")
    print(f"[init] Logs        : {log_dir}")

    ckpt_path = ckpt_dir / "best_deeplab.pth"
    log_path  = log_dir  / "history_deeplab.json"

    # ── device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[init] Device : {device}")

    # ── datasets ──────────────────────────────────────────────────────────────
    train_ds = ResPlanSegDataset(
        CFG.IMG_DIR, CFG.MSK_DIR,
        splits_file=CFG.TRAIN_SPLIT,
        split="train", img_size=CFG.IMG_SIZE,
    )
    val_ds = ResPlanSegDataset(
        CFG.IMG_DIR, CFG.MSK_DIR,
        splits_file=CFG.VAL_SPLIT,
        split="val", img_size=CFG.IMG_SIZE,
    )
    print(f"[data]  train={len(train_ds)}  val={len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=CFG.BATCH_SIZE, shuffle=True,
        num_workers=CFG.NUM_WORKERS, pin_memory=CFG.PIN_MEMORY, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=CFG.BATCH_SIZE, shuffle=False,
        num_workers=CFG.NUM_WORKERS, pin_memory=CFG.PIN_MEMORY,
    )

    # ── model ─────────────────────────────────────────────────────────────────
    print(f"[model] Building DeepLabV3+  backbone={CFG.BACKBONE}  "
          f"output_stride={CFG.OUTPUT_STRIDE}  classes={CFG.NUM_CLASSES}")
    model = DeepLabV3Plus(
        num_classes=CFG.NUM_CLASSES,
        backbone=CFG.BACKBONE,
        output_stride=CFG.OUTPUT_STRIDE,
        pretrained=True,
    ).to(device)

    # freeze early backbone layers (pretrained, rarely need fine-tuning)
    for param in model.stem.parameters():
        param.requires_grad = False
    for param in model.layer1.parameters():
        param.requires_grad = False

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"[model] Parameters: {n_params:.1f} M  (trainable: {n_trainable:.1f} M)")

    # torch.compile for faster training on A100
    if hasattr(torch, "compile"):
        model = torch.compile(model)
        print("[model] torch.compile enabled")

    # ── loss ──────────────────────────────────────────────────────────────────
    class_weights = None
    if CFG.CLASS_WEIGHTS is not None:
        class_weights = torch.tensor(CFG.CLASS_WEIGHTS, dtype=torch.float32, device=device)

    criterion = SegLoss(
        num_classes=CFG.NUM_CLASSES,
        class_weights=class_weights,
        dice_weight=CFG.DICE_WEIGHT,
        ignore_index=CFG.IGNORE_INDEX,
    )

    # ── optimiser — separate LRs for backbone vs decoder ─────────────────────
    # Backbone is pretrained → lower effective LR via weight decay only.
    # Decoder is randomly initialised → full LR.
    # stem and layer1 are frozen — only include trainable backbone layers
    backbone_params = (
        list(model.layer2.parameters()) +
        list(model.layer3.parameters()) +
        list(model.layer4.parameters())
    )
    decoder_params = (
        list(model.aspp.parameters())       +
        list(model.low_proj.parameters())   +
        list(model.decoder.parameters())    +
        list(model.classifier.parameters())
    )
    optimiser = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": CFG.LR * 0.1},   # 6e-6
            {"params": decoder_params,  "lr": CFG.LR},          # 6e-5
        ],
        weight_decay=CFG.WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=CFG.EPOCHS, eta_min=CFG.LR_MIN,
    )

    # ── AMP ───────────────────────────────────────────────────────────────────
    scaler = torch.cuda.amp.GradScaler(enabled=CFG.AMP and device.type == "cuda")

    # ── metrics ───────────────────────────────────────────────────────────────
    train_metric = StreamingMIoU(CFG.NUM_CLASSES, CFG.IGNORE_INDEX)
    val_metric   = StreamingMIoU(CFG.NUM_CLASSES, CFG.IGNORE_INDEX)

    # ── training loop ─────────────────────────────────────────────────────────
    history    = []
    best_miou  = -1.0
    es_counter = 0

    print(f"\n{'─'*68}")
    print(f"  DeepLabV3+ ({CFG.BACKBONE})  |  {CFG.NUM_CLASSES} classes  |  {CFG.EPOCHS} epochs")
    print(f"{'─'*68}\n")

    for epoch in range(1, CFG.EPOCHS + 1):
        t0 = time.time()

        train_stats = run_epoch(
            model, train_loader, criterion, train_metric,
            optimiser=optimiser, scaler=scaler, device=device, amp=CFG.AMP,
            epoch=epoch, tag="train",
        )
        val_stats = run_epoch(
            model, val_loader, criterion, val_metric,
            device=device, amp=CFG.AMP,
            epoch=epoch, tag="val",
        )

        scheduler.step()
        elapsed = time.time() - t0
        # report the decoder (higher) LR
        lr_now = optimiser.param_groups[1]["lr"]

        record = {
            "epoch"            : epoch,
            "lr_backbone"      : round(optimiser.param_groups[0]["lr"], 9),
            "lr_decoder"       : round(lr_now, 8),
            "train_loss"       : train_stats["loss"],
            "train_mIoU"       : train_stats["mIoU"],
            "val_loss"         : val_stats["loss"],
            "val_mIoU"         : val_stats["mIoU"],
            "val_iou_per_class": val_stats["iou_per_class"],
            "elapsed_s"        : round(elapsed, 1),
        }
        history.append(record)

        print(
            f"[{epoch:03d}/{CFG.EPOCHS}]  "
            f"lr={lr_now:.2e}  "
            f"train loss={train_stats['loss']:.4f}  mIoU={train_stats['mIoU']:.4f}  │  "
            f"val loss={val_stats['loss']:.4f}  mIoU={val_stats['mIoU']:.4f}  "
            f"({elapsed:.0f}s)"
        )

        # ── checkpoint ────────────────────────────────────────────────────────
        if val_stats["mIoU"] > best_miou:
            best_miou  = val_stats["mIoU"]
            es_counter = 0
            torch.save(
                {
                    "epoch"     : epoch,
                    "model"     : model.state_dict(),
                    "optimiser" : optimiser.state_dict(),
                    "scheduler" : scheduler.state_dict(),
                    "val_mIoU"  : best_miou,
                    "cfg": {
                        "backbone"      : CFG.BACKBONE,
                        "num_classes"   : CFG.NUM_CLASSES,
                        "output_stride" : CFG.OUTPUT_STRIDE,
                        "img_size"      : CFG.IMG_SIZE,
                    },
                },
                ckpt_path,
            )
            print(f"           ✓ new best mIoU={best_miou:.4f}  → saved {ckpt_path}")
        else:
            es_counter += 1
            if es_counter >= CFG.ES_PATIENCE:
                print(f"\n[early stop] No improvement for {CFG.ES_PATIENCE} epochs. Stopping.")
                break

        # ── flush log ─────────────────────────────────────────────────────────
        with open(log_path, "w") as f:
            json.dump(history, f, indent=2)

    print(f"\nTraining complete.  Best val mIoU = {best_miou:.4f}")
    print(f"  checkpoint → {ckpt_path}")
    print(f"  history    → {log_path}")


# ── inference helper (load best checkpoint) ───────────────────────────────────

def load_best_model(ckpt_path: str | Path, device: torch.device | None = None) -> DeepLabV3Plus:
    """Restore the best checkpoint for inference."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt  = torch.load(ckpt_path, map_location=device)
    cfg   = ckpt["cfg"]
    model = DeepLabV3Plus(
        num_classes=cfg["num_classes"],
        backbone=cfg["backbone"],
        output_stride=cfg["output_stride"],
        pretrained=False,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint  epoch={ckpt['epoch']}  val_mIoU={ckpt['val_mIoU']:.4f}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
