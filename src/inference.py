"""
inference.py - Run trained SegFormer on test set and evaluate against GT
========================================================================
Pipeline per image:
    1. Load image → model → predicted segmentation mask
    2. Load GT mask
    3. Pixel metrics   : per-class IoU / precision / recall  (compute_miou)
    4. Build graphs    : G_pred from predicted mask, G_gt from GT mask
    5. Edge metrics    : edge precision / recall / F1 + type accuracy
    6. Graph edit dist : structural similarity (GED)
    7. Proximity diff  : Frobenius norm between adjacency matrices

Outputs saved to  results/eval_<timestamp>/  :
    per_image.csv - per-image metrics
    summary.csv - aggregated means across test set
    class_iou.csv - per-class IoU averaged over all images
    sample_*.png - overlay visualisations (first N images)

Colab usage:
    !python inference.py                              # defaults
    !python inference.py --ckpt path/to/best.pth      # custom checkpoint
    !python inference.py --save-vis 20                 # save 20 visual samples
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    _SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    _SCRIPT_DIR = Path("/content/drive/MyDrive/bubble_diagram_project")

sys.path.insert(0, str(_SCRIPT_DIR))

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import SegformerForSemanticSegmentation

# local imports
from build_graph import build_graph_from_segmentation
from build_gt_graph import build_gt_graph_from_resplan, load_resplan_records
from proximity import compute_proximity_matrix
from evaluate import compute_miou, edge_metrics, graph_edit_distance, frobenius_norm
# NB: `from visualize import draw_bubble_diagram` is imported lazily inside
# save_visualisation() instead of here, because visualize.py derives its
# ROOM_COLORS from this module's _PALETTE at import time; a top-level import
# either way would create a circular import.


# ══════════════════════════════════════════════════════════════════════════════
# Config defaults (mirrors train.py)
# ══════════════════════════════════════════════════════════════════════════════

PRETRAINED   = "nvidia/segformer-b3-finetuned-ade-512-512"
NUM_CLASSES  = 16
IMG_SIZE     = 512
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Model loading
# ══════════════════════════════════════════════════════════════════════════════

def load_model(ckpt_path: str | Path, device: torch.device) -> SegformerForSemanticSegmentation:
    """Load SegFormer-B3 and restore weights from train.py checkpoint."""
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    id2label = {i: str(i) for i in range(NUM_CLASSES)}
    label2id = {v: k for k, v in id2label.items()}

    model = SegformerForSemanticSegmentation.from_pretrained(
        PRETRAINED,
        num_labels=NUM_CLASSES,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    epoch = ckpt.get("epoch", "?")
    miou  = ckpt.get("val_mIoU", "?")
    print(f"[model] Loaded checkpoint: epoch={epoch}  val_mIoU={miou}")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Single-image inference
# ══════════════════════════════════════════════════════════════════════════════

def preprocess(image_bgr: np.ndarray, img_size: int = IMG_SIZE) -> torch.Tensor:
    """BGR uint8 image → (1, 3, H, W) normalised float tensor."""
    img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = np.transpose(img, (2, 0, 1))          # (3, H, W)
    return torch.from_numpy(img).unsqueeze(0)    # (1, 3, H, W)


@torch.no_grad()
def predict_mask(
    model: SegformerForSemanticSegmentation,
    image_bgr: np.ndarray,
    device: torch.device,
    img_size: int = IMG_SIZE,
) -> np.ndarray:
    """Run model on a single image, return (H, W) int predicted mask at original size."""
    h_orig, w_orig = image_bgr.shape[:2]
    tensor = preprocess(image_bgr, img_size).to(device)

    outputs = model(pixel_values=tensor)
    logits = outputs.logits                                      # (1, C, H/4, W/4)
    logits = F.interpolate(logits, size=(img_size, img_size),
                           mode="bilinear", align_corners=False)  # (1, C, H, W)
    pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()          # (H, W)

    # resize back to original resolution for graph extraction
    if (h_orig, w_orig) != (img_size, img_size):
        pred = cv2.resize(pred.astype(np.uint8), (w_orig, h_orig),
                          interpolation=cv2.INTER_NEAREST)
    return pred.astype(np.int64)


# ══════════════════════════════════════════════════════════════════════════════
# Visualisation helper
# ══════════════════════════════════════════════════════════════════════════════

# colour palette for mask overlay (class_id -> BGR, since this blends with an
# OpenCV BGR image via cv2.addWeighted below).
#
# The per-class labels below are the authoritative class -> colour mapping and
# match build_graph.CLASS_NAMES. The numeric BGR values here are, by design,
# the BGR encoding of the same colours that visualize.ROOM_COLORS and
# generate_bubble.py use (stored there as RGB) for the same class id, so a
# class renders as the same colour whether this overlay or a bubble diagram
# draws it - in fact visualize.ROOM_COLORS is derived from this array. (An
# earlier revision had inline labels shifted by one - e.g. index 1 mislabelled
# "Living" when class 1 is Bedroom, index 7 mislabelled a non-existent "Dining"
# when class 7 is Stair; only those labels were wrong, never the values, and
# they are now corrected.)
_PALETTE = np.zeros((NUM_CLASSES, 3), dtype=np.uint8)
_PALETTE[1]  = (74, 175, 77)   # Bedroom
_PALETTE[2]  = (163, 78, 152)   # Bathroom
_PALETTE[3]  = (184, 126, 55)   # Kitchen
_PALETTE[4]  = (0, 127, 255)   # Living
_PALETTE[5]  = (47, 217, 255)   # Balcony
_PALETTE[6]  = (153, 153, 153)   # Storage
_PALETTE[7]  = (40, 86, 166)   # Stair
_PALETTE[8]  = (191, 129, 247)   # Parking
_PALETTE[9]  = (136, 150, 0)   # Pool
_PALETTE[10] = (80, 80, 80)   # Wall
_PALETTE[11] = (0, 0, 200)   # Door
_PALETTE[12] = (200, 200, 0)   # Window
_PALETTE[13] = (99, 30, 233)   # FrontDoor


def _overlay(image_bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Blend a colour-coded mask onto the image."""
    colour = _PALETTE[mask.clip(0, NUM_CLASSES - 1)]
    overlay = cv2.addWeighted(image_bgr, 1 - alpha, colour, alpha, 0)
    # keep original where mask == 0 (background)
    overlay[mask == 0] = image_bgr[mask == 0]
    return overlay


def save_visualisation(
    image_bgr: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    G_pred, G_gt,
    stem: str,
    out_path: Path,
):
    """Save a 2x2 figure: image, GT overlay, pred overlay, bubble diagrams."""
    # lazy import to avoid a circular import (see the note by the local-imports
    # block above): visualize derives ROOM_COLORS from this module's _PALETTE.
    from visualize import draw_bubble_diagram

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # top-left: original image
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title("Input Image")
    axes[0, 0].axis("off")

    # top-right: GT mask overlay
    gt_overlay = cv2.cvtColor(_overlay(image_bgr, gt_mask), cv2.COLOR_BGR2RGB)
    axes[0, 1].imshow(gt_overlay)
    axes[0, 1].set_title("Ground Truth")
    axes[0, 1].axis("off")

    # bottom-left: predicted mask overlay
    pred_overlay = cv2.cvtColor(_overlay(image_bgr, pred_mask), cv2.COLOR_BGR2RGB)
    axes[1, 0].imshow(pred_overlay)
    axes[1, 0].set_title("Prediction")
    axes[1, 0].axis("off")

    # bottom-right: bubble diagrams (pred vs GT)
    ax_bubble = axes[1, 1]
    if G_pred.number_of_nodes() > 0:
        draw_bubble_diagram(G_pred, ax=ax_bubble, title="Pred Graph")
    else:
        ax_bubble.text(0.5, 0.5, "No rooms detected", ha="center", va="center",
                       transform=ax_bubble.transAxes, fontsize=14)
        ax_bubble.set_title("Pred Graph")
        ax_bubble.axis("off")

    fig.suptitle(f"Sample: {stem}", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Main evaluation loop
# ══════════════════════════════════════════════════════════════════════════════

def _save_checkpoint(rows, class_iou_accum, out_dir, total, t_start):
    """Save intermediate CSVs after each batch."""
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "per_image.csv", index=False)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_summary = df[numeric_cols].mean().to_frame("mean").T
    df_summary.to_csv(out_dir / "summary.csv", index=False)

    if class_iou_accum:
        df_all_class = pd.concat(class_iou_accum, ignore_index=True)
        df_class_avg = (
            df_all_class
            .groupby(["class_id", "class_name"])
            .agg({"iou": "mean", "precision": "mean", "recall": "mean", "support": "sum"})
            .reset_index()
            .sort_values("class_id")
        )
        df_class_avg.to_csv(out_dir / "class_iou.csv", index=False)

    elapsed = time.time() - t_start
    print(f"  [saved] {len(rows)}/{total} images -> {out_dir / 'per_image.csv'}  ({elapsed:.0f}s)")


def run_evaluation(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[init] Device: {device}")

    # paths
    root      = Path(args.root)
    img_dir   = root / "data" / "resplan_raster"
    mask_dir  = root / "data" / "resplan_masks"
    split_file = root / "data" / "splits" / "test.txt"

    if not split_file.exists():
        raise FileNotFoundError(f"Test split not found: {split_file}")

    stems = [l.strip() for l in split_file.read_text().splitlines() if l.strip()]
    if args.limit > 0:
        stems = stems[:args.limit]
    print(f"[data] Test samples: {len(stems)}")

    resplan_pkl = root / "data" / "resplan_raw" / "ResPlan.pkl"
    print(f"[gt] loading ResPlan's own graphs from {resplan_pkl}")
    resplan_records = load_resplan_records(resplan_pkl)

    # output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = root / "results" / f"eval_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = out_dir / "visualisations"
    if args.save_vis > 0:
        vis_dir.mkdir(exist_ok=True)
    print(f"[output] {out_dir}")

    # load model
    model = load_model(args.ckpt, device)

    # accumulators
    rows = []
    class_iou_accum = []
    batch_size = args.batch_save

    # resume support: load existing per_image.csv if present
    per_image_path = out_dir / "per_image.csv"
    done_stems = set()
    if per_image_path.exists():
        df_existing = pd.read_csv(per_image_path)
        rows = df_existing.to_dict("records")
        done_stems = set(df_existing["stem"].astype(str))
        print(f"[resume] Found {len(done_stems)} already-processed images, resuming...")

    t_start = time.time()
    batch_count = 0

    for i, stem in enumerate(stems):
        # skip already-processed images (resume support)
        if stem in done_stems:
            continue

        img_path  = img_dir  / f"{stem}.png"
        mask_path = mask_dir / f"{stem}_mask.png"

        if not img_path.exists() or not mask_path.exists():
            print(f"  [skip] {stem} - file missing")
            continue

        # --- load ---
        image_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        gt_mask   = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE).astype(np.int64)

        # --- predict ---
        pred_mask = predict_mask(model, image_bgr, device, IMG_SIZE)

        # resize GT to match pred if needed
        if gt_mask.shape != pred_mask.shape:
            gt_mask = cv2.resize(gt_mask.astype(np.uint8), pred_mask.shape[::-1],
                                 interpolation=cv2.INTER_NEAREST).astype(np.int64)

        # --- 1. pixel metrics ---
        df_iou = compute_miou(pred_mask, gt_mask, num_classes=NUM_CLASSES)
        mean_row = df_iou[df_iou.class_name == "mean"].iloc[0]
        miou = mean_row["iou"]
        class_iou_accum.append(df_iou[df_iou.class_name != "mean"])

        # --- 2. build graphs ---
        try:
            G_pred = build_graph_from_segmentation(pred_mask)
        except Exception as e:
            print(f"  [warn] {stem} - pred graph failed: {e}")
            G_pred = __import__("networkx").Graph()

        try:
            plan_id = int(stem)
            G_gt = build_gt_graph_from_resplan(resplan_records[plan_id])
        except Exception as e:
            print(f"  [warn] {stem} - GT graph failed: {e}")
            G_gt = __import__("networkx").Graph()

        # --- 3. edge metrics ---
        df_edge = edge_metrics(G_pred, G_gt)
        e = df_edge.iloc[0]

        # --- 4. graph edit distance ---
        if args.skip_ged:
            ged = float("nan")
        else:
            df_ged = graph_edit_distance(G_pred, G_gt, timeout=args.ged_timeout)
            ged = df_ged["ged"].iloc[0]

        # --- 5. frobenius ---
        A_pred, _ = compute_proximity_matrix(G_pred) if G_pred.number_of_nodes() > 0 else (np.zeros((0, 0)), [])
        A_gt, _   = compute_proximity_matrix(G_gt)   if G_gt.number_of_nodes() > 0   else (np.zeros((0, 0)), [])
        if A_pred.size == 0 and A_gt.size == 0:
            frob, frob_norm = 0.0, 0.0
        else:
            df_frob = frobenius_norm(A_pred, A_gt)
            frob      = df_frob["frobenius"].iloc[0]
            frob_norm = df_frob["normalized"].iloc[0]

        # --- collect ---
        rows.append({
            "stem":           stem,
            "mIoU":           round(miou, 4),
            "edge_precision": e["edge_precision"],
            "edge_recall":    e["edge_recall"],
            "edge_f1":        e["edge_f1"],
            "type_accuracy":  e["type_accuracy"],
            "ged":            ged,
            "frobenius":      round(frob, 4),
            "frob_normalized": round(frob_norm, 4),
            "n_rooms_pred":   G_pred.number_of_nodes(),
            "n_rooms_gt":     G_gt.number_of_nodes(),
            "n_edges_pred":   G_pred.number_of_edges(),
            "n_edges_gt":     G_gt.number_of_edges(),
        })
        batch_count += 1

        # --- visualisation ---
        if i < args.save_vis:
            save_visualisation(
                image_bgr, gt_mask, pred_mask,
                G_pred, G_gt, stem,
                vis_dir / f"sample_{stem}.png",
            )

        # progress
        if batch_count % 50 == 0 or (i + 1) == len(stems):
            elapsed = time.time() - t_start
            print(f"  [{len(rows):>5}/{len(stems)}]  mIoU={miou:.4f}  "
                  f"edge_f1={e['edge_f1']}  ged={ged:.0f}  "
                  f"({elapsed:.0f}s)")

        # --- save after every batch_size images ---
        if batch_count % batch_size == 0:
            _save_checkpoint(rows, class_iou_accum, out_dir, len(stems), t_start)

    # ── final save ────────────────────────────────────────────────────────────
    df_per_image = pd.DataFrame(rows)
    df_per_image.to_csv(out_dir / "per_image.csv", index=False)

    # summary
    numeric_cols = df_per_image.select_dtypes(include=[np.number]).columns
    df_summary = df_per_image[numeric_cols].mean().to_frame("mean").T
    df_summary.to_csv(out_dir / "summary.csv", index=False)

    # per-class IoU (averaged across all images)
    if class_iou_accum:
        df_all_class = pd.concat(class_iou_accum, ignore_index=True)
        df_class_avg = (
            df_all_class
            .groupby(["class_id", "class_name"])
            .agg({"iou": "mean", "precision": "mean", "recall": "mean", "support": "sum"})
            .reset_index()
            .sort_values("class_id")
        )
        df_class_avg.to_csv(out_dir / "class_iou.csv", index=False)
    else:
        df_class_avg = pd.DataFrame()

    # ── print summary ─────────────────────────────────────────────────────────
    total_time = time.time() - t_start
    print(f"\n{'='*65}")
    print(f"  EVALUATION COMPLETE  ({len(rows)} images, {total_time:.0f}s)")
    print(f"{'='*65}")
    print(f"  mIoU            : {df_summary['mIoU'].iloc[0]:.4f}")
    print(f"  Edge F1         : {df_summary['edge_f1'].iloc[0]:.4f}")
    print(f"  Edge Type Acc   : {df_summary['type_accuracy'].iloc[0]:.4f}")
    print(f"  GED (mean)      : {df_summary['ged'].iloc[0]:.2f}")
    print(f"  Frobenius (norm): {df_summary['frob_normalized'].iloc[0]:.4f}")
    print(f"{'='*65}")

    if not df_class_avg.empty:
        print("\nPer-class IoU:")
        for _, r in df_class_avg.iterrows():
            print(f"  {r['class_name']:>12s}  IoU={r['iou']:.4f}  P={r['precision']:.4f}  R={r['recall']:.4f}")

    print(f"\nResults saved to: {out_dir}")
    return df_per_image, df_summary, df_class_avg


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="SegFormer inference + evaluation")
    parser.add_argument(
        "--root", type=str,
        default=str(_SCRIPT_DIR),
        help="Project root (contains data/ and checkpoints/)",
    )
    parser.add_argument(
        "--ckpt", type=str,
        default=str(_SCRIPT_DIR / "checkpoints" / "segformer" / "best_model.pth"),
        help="Path to .pth checkpoint from train.py",
    )
    parser.add_argument(
        "--save-vis", type=int, default=10,
        help="Number of sample visualisations to save (0 to disable)",
    )
    parser.add_argument(
        "--ged-timeout", type=float, default=3.0,
        help="Seconds to spend on GED per image (lower = faster, but fewer "
             "graphs converge within the cutoff; matches the canonical "
             "recompute_ged_parallel.py default of 3.0)",
    )
    parser.add_argument(
        "--skip-ged", action="store_true",
        help="Skip GED entirely (some room graphs make optimize_graph_edit_distance "
             "take far longer than ged_timeout to yield its first candidate)",
    )
    parser.add_argument(
        "--batch-save", type=int, default=300,
        help="Save intermediate CSVs every N images (default: 300)",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Max test images to process (0 = all)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(args)
