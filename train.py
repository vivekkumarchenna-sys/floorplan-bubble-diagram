"""
train.py — SegFormer-B3 training for 16-class semantic segmentation
====================================================================
Pretrained : nvidia/segformer-b3-finetuned-ade-512-512
Input size : 512 × 512
Classes    : 16
Loss       : weighted CrossEntropy + 0.5 × Dice
Optimiser  : AdamW  lr=6e-5  weight_decay=0.01
Schedule   : CosineAnnealingLR  T_max=100
Early stop : patience=15  (monitor val mIoU)
Logs       : JSON  (one record per epoch)
Checkpoint : best val-mIoU model saved to Drive / local output dir

Colab quick-start
-----------------
    from google.colab import drive
    drive.mount('/content/drive')
    # then run:  !python train.py
    # or set DRIVE_ROOT below and execute cells.

Directory layout expected (configure via CFG)
---------------------------------------------
    data/
      train/images/   *.png
      train/masks/    *.png
      val/images/     *.png
      val/masks/      *.png
"""

# ── stdlib ──────────────────────────────────────────────────────────────────
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ── third-party ─────────────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
)

# ── local ────────────────────────────────────────────────────────────────────
from dataset import ResPlanSegDataset


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                        GLOBAL CONFIGURATION                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class CFG:
    # ── data ─────────────────────────────────────────────────────────────────
    # Project root = the folder containing this script (works on Drive & locally).
    ROOT           = Path(__file__).resolve().parent

    DATA_ROOT      = ROOT / "data"
    IMG_DIR        = DATA_ROOT / "resplan_raster"
    MSK_DIR        = DATA_ROOT / "resplan_masks"
    TRAIN_SPLIT    = DATA_ROOT / "splits" / "train.txt"
    VAL_SPLIT      = DATA_ROOT / "splits" / "val.txt"

    # ── model ─────────────────────────────────────────────────────────────────
    PRETRAINED     = "nvidia/segformer-b3-finetuned-ade-512-512"
    NUM_CLASSES    = 16
    IMG_SIZE       = 512

    # ── training ──────────────────────────────────────────────────────────────
    EPOCHS         = 100
    BATCH_SIZE     = 8          # reduce to 4 on 12 GB VRAM
    NUM_WORKERS    = 4          # set 0 on Windows / some Colab configs
    PIN_MEMORY     = True

    LR             = 6e-5
    WEIGHT_DECAY   = 0.01
    LR_MIN         = 1e-7       # CosineAnnealingLR eta_min

    DICE_WEIGHT    = 0.5        # loss = CE + DICE_WEIGHT * Dice

    # Per-class CE weights (length must equal NUM_CLASSES).
    # Set to None to use uniform weights.
    CLASS_WEIGHTS  = None

    # ── early stopping ────────────────────────────────────────────────────────
    ES_PATIENCE    = 15         # epochs without val mIoU improvement

    # ── misc ──────────────────────────────────────────────────────────────────
    SEED           = 42
    AMP            = True       # mixed-precision training (fp16)
    GRAD_CLIP      = 1.0        # max gradient norm; None to disable


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                              LOSS                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class DiceLoss(nn.Module):
    """Soft multiclass Dice loss, ignores index 255."""

    def __init__(self, num_classes: int, ignore_index: int = 255, eps: float = 1e-6):
        super().__init__()
        self.C            = num_classes
        self.ignore_index = ignore_index
        self.eps          = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits  : (B, C, H, W)  – raw (unscaled) logits
        targets : (B, H, W)     – integer class indices
        """
        valid_mask = (targets != self.ignore_index)          # (B, H, W)
        targets_c  = targets.clone()
        targets_c[~valid_mask] = 0                           # avoid OOB index

        probs   = F.softmax(logits, dim=1)                   # (B, C, H, W)
        one_hot = F.one_hot(targets_c, self.C)               # (B, H, W, C)
        one_hot = one_hot.permute(0, 3, 1, 2).float()        # (B, C, H, W)

        # mask out ignored pixels
        vm = valid_mask.unsqueeze(1).float()                 # (B, 1, H, W)
        probs   = probs   * vm
        one_hot = one_hot * vm

        dims    = (0, 2, 3)
        inter   = (probs * one_hot).sum(dims)
        denom   = (probs + one_hot).sum(dims)

        dice_per_class = (2.0 * inter + self.eps) / (denom + self.eps)
        return 1.0 - dice_per_class.mean()


class SegLoss(nn.Module):
    """Weighted CE + α·Dice."""

    def __init__(
        self,
        num_classes: int,
        class_weights: torch.Tensor | None = None,
        dice_weight: float = 0.5,
        ignore_index: int = 255,
    ):
        super().__init__()
        self.ce   = nn.CrossEntropyLoss(
            weight=class_weights, ignore_index=ignore_index
        )
        self.dice = DiceLoss(num_classes, ignore_index=ignore_index)
        self.dw   = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.ce(logits, targets) + self.dw * self.dice(logits, targets)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                            METRICS                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class StreamingMIoU:
    """
    Accumulates a confusion matrix over batches; computes mIoU on demand.
    Ignores pixels labelled `ignore_index`.
    """

    def __init__(self, num_classes: int, ignore_index: int = 255):
        self.C      = num_classes
        self.ignore = ignore_index
        self.reset()

    def reset(self):
        self.mat = torch.zeros(self.C, self.C, dtype=torch.long)

    @torch.no_grad()
    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """preds, targets : (B, H, W) integer tensors on any device."""
        preds   = preds.view(-1).cpu()
        targets = targets.view(-1).cpu()

        valid   = targets != self.ignore
        preds   = preds[valid]
        targets = targets[valid]

        # clip predictions to valid range (model can output C-1 at most)
        preds = preds.clamp(0, self.C - 1)

        idx  = targets * self.C + preds
        self.mat += torch.bincount(idx, minlength=self.C * self.C).reshape(self.C, self.C)

    def compute(self) -> dict:
        mat   = self.mat.float()
        inter = mat.diag()
        union = mat.sum(1) + mat.sum(0) - inter

        iou_per_class = inter / union.clamp(min=1)
        present       = union > 0
        miou          = iou_per_class[present].mean().item()

        return {
            "mIoU"         : round(miou, 6),
            "iou_per_class": iou_per_class.tolist(),
        }


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                          MODEL BUILDER                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def build_model(num_classes: int, pretrained: str) -> SegformerForSemanticSegmentation:
    """
    Load SegFormer-B3 pretrained on ADE-150 and replace the decode head
    classifier with a fresh `num_classes`-output layer.
    """
    id2label = {i: str(i) for i in range(num_classes)}
    label2id = {v: k for k, v in id2label.items()}

    model = SegformerForSemanticSegmentation.from_pretrained(
        pretrained,
        num_labels=num_classes,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,   # replaces mismatched classifier head
    )
    return model


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                         TRAINING ENGINE                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _upscale_logits(logits: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
    """Bilinearly upsample SegFormer logits (B, C, H/4, W/4) → (B, C, H, W)."""
    return F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False)


def run_epoch(
    model,
    loader: DataLoader,
    criterion: SegLoss,
    metric: StreamingMIoU,
    optimiser=None,
    scaler=None,
    device: torch.device = torch.device("cpu"),
    amp: bool = True,
) -> dict:
    training = optimiser is not None
    model.train() if training else model.eval()
    metric.reset()

    total_loss = 0.0
    n_batches  = 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for batch in loader:
            images  = batch["image"].to(device, non_blocking=True)   # (B,3,H,W)
            masks   = batch["mask"].to(device, non_blocking=True)    # (B,H,W)

            with torch.autocast(device.type, enabled=amp):
                outputs = model(pixel_values=images)
                logits  = _upscale_logits(outputs.logits, masks.shape[-2:])
                loss    = criterion(logits, masks)

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

            preds = logits.argmax(dim=1)
            metric.update(preds, masks)
            total_loss += loss.item()
            n_batches  += 1

    stats          = metric.compute()
    stats["loss"]  = round(total_loss / max(n_batches, 1), 6)
    return stats


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                              MAIN                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def main():
    # ── reproducibility ───────────────────────────────────────────────────────
    torch.manual_seed(CFG.SEED)
    np.random.seed(CFG.SEED)

    # ── output directories ────────────────────────────────────────────────────
    ckpt_dir = CFG.ROOT / "checkpoints" / "segformer"
    log_dir  = CFG.ROOT / "results"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[init] Checkpoints : {ckpt_dir}")
    print(f"[init] Logs        : {log_dir}")

    ckpt_path = ckpt_dir / "best_model.pth"
    log_path  = log_dir  / "history_segformer.json"

    # ── device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[init] Device: {device}")

    # ── datasets & loaders ────────────────────────────────────────────────────
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
    print(f"[data] train={len(train_ds)}  val={len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=CFG.BATCH_SIZE,
        shuffle=True,
        num_workers=CFG.NUM_WORKERS,
        pin_memory=CFG.PIN_MEMORY,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=CFG.BATCH_SIZE,
        shuffle=False,
        num_workers=CFG.NUM_WORKERS,
        pin_memory=CFG.PIN_MEMORY,
    )

    # ── model ─────────────────────────────────────────────────────────────────
    print(f"[model] Loading {CFG.PRETRAINED} …")
    model = build_model(CFG.NUM_CLASSES, CFG.PRETRAINED).to(device)

    # ── loss ──────────────────────────────────────────────────────────────────
    class_weights = None
    if CFG.CLASS_WEIGHTS is not None:
        class_weights = torch.tensor(CFG.CLASS_WEIGHTS, dtype=torch.float32, device=device)

    criterion = SegLoss(
        num_classes=CFG.NUM_CLASSES,
        class_weights=class_weights,
        dice_weight=CFG.DICE_WEIGHT,
    )

    # ── optimiser & scheduler ─────────────────────────────────────────────────
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=CFG.LR,
        weight_decay=CFG.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser,
        T_max=CFG.EPOCHS,
        eta_min=CFG.LR_MIN,
    )

    # ── AMP scaler ────────────────────────────────────────────────────────────
    scaler = torch.cuda.amp.GradScaler(enabled=CFG.AMP and device.type == "cuda")

    # ── metrics ───────────────────────────────────────────────────────────────
    train_metric = StreamingMIoU(CFG.NUM_CLASSES)
    val_metric   = StreamingMIoU(CFG.NUM_CLASSES)

    # ── resume from checkpoint if available ───────────────────────────────────
    history     = []
    best_miou   = -1.0
    es_counter  = 0
    start_epoch = 1

    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimiser.load_state_dict(ckpt["optimiser"])
        scheduler.load_state_dict(ckpt["scheduler"])
        best_miou   = ckpt["val_mIoU"]
        start_epoch = ckpt["epoch"] + 1
        print(f"[resume] Checkpoint loaded — epoch {ckpt['epoch']}  val_mIoU={best_miou:.4f}")
        if log_path.exists():
            history = json.load(open(log_path))
            print(f"[resume] History loaded  ({len(history)} epochs)")
    else:
        print("[resume] No checkpoint found — starting from scratch.")

    print(f"\n{'─'*65}")
    print(f"  Training SegFormer-B3  |  {CFG.NUM_CLASSES} classes  |  {CFG.EPOCHS} epochs")
    print(f"{'─'*65}\n")

    for epoch in range(start_epoch, CFG.EPOCHS + 1):
        t0 = time.time()

        # ---- train ----
        train_stats = run_epoch(
            model, train_loader, criterion, train_metric,
            optimiser=optimiser, scaler=scaler,
            device=device, amp=CFG.AMP,
        )

        # ---- validate ----
        val_stats = run_epoch(
            model, val_loader, criterion, val_metric,
            device=device, amp=CFG.AMP,
        )

        scheduler.step()
        elapsed = time.time() - t0
        lr_now  = scheduler.get_last_lr()[0]

        # ---- log ----
        record = {
            "epoch"      : epoch,
            "lr"         : round(lr_now, 8),
            "train_loss" : train_stats["loss"],
            "train_mIoU" : train_stats["mIoU"],
            "val_loss"   : val_stats["loss"],
            "val_mIoU"   : val_stats["mIoU"],
            "val_iou_per_class": val_stats["iou_per_class"],
            "elapsed_s"  : round(elapsed, 1),
        }
        history.append(record)

        print(
            f"[{epoch:03d}/{CFG.EPOCHS}]  "
            f"lr={lr_now:.2e}  "
            f"train loss={train_stats['loss']:.4f}  mIoU={train_stats['mIoU']:.4f}  │  "
            f"val loss={val_stats['loss']:.4f}  mIoU={val_stats['mIoU']:.4f}  "
            f"({elapsed:.0f}s)"
        )

        # ---- checkpoint ----
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
                    "cfg"       : {k: str(v) for k, v in vars(CFG).items()
                                   if not k.startswith("_")},
                },
                ckpt_path,
            )
            print(f"           ✓ new best mIoU={best_miou:.4f}  → saved {ckpt_path}")
        else:
            es_counter += 1
            if es_counter >= CFG.ES_PATIENCE:
                print(f"\n[early stop] No val mIoU improvement for {CFG.ES_PATIENCE} epochs. Stopping.")
                break

        # ---- flush JSON log ----
        with open(log_path, "w") as f:
            json.dump(history, f, indent=2)

    print(f"\nTraining complete.  Best val mIoU = {best_miou:.4f}")
    print(f"  checkpoint → {ckpt_path}")
    print(f"  history    → {log_path}")


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
