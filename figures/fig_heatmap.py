"""
fig_heatmap.py — Fig 5: Proximity matrix heatmaps (GT vs Pred side-by-side)
============================================================================
Picks a sample image, builds GT and pred graphs, computes proximity matrices,
and plots them as side-by-side heatmaps with a difference map.

Usage:
    python fig_heatmap.py
    python fig_heatmap.py --stem 16649 --ckpt path/to/model.pth
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    _SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    _SCRIPT_DIR = Path("/content/drive/MyDrive/bubble_diagram_project")

sys.path.insert(0, str(_SCRIPT_DIR))

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from build_graph import build_graph_from_segmentation
from build_gt_graph import mask_to_plan_dict, build_gt_graph_from_polygons
from proximity import compute_proximity_matrix
from inference import load_model, predict_mask, IMG_SIZE, NUM_CLASSES

import torch

# ── style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 13,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def plot_heatmaps(
    A_gt: np.ndarray, labels_gt: list[str],
    A_pred: np.ndarray, labels_pred: list[str],
    stem: str,
    save_path: str,
):
    """Plot GT, Pred, and Difference proximity matrix heatmaps."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    vmax = max(A_gt.max(), A_pred.max()) if A_gt.size > 0 and A_pred.size > 0 else 1.0

    # GT heatmap
    im0 = axes[0].imshow(A_gt, cmap="YlOrRd", vmin=0, vmax=vmax, aspect="equal")
    axes[0].set_title("Ground Truth", fontweight="bold")
    axes[0].set_xticks(range(len(labels_gt)))
    axes[0].set_xticklabels(labels_gt, rotation=45, ha="right", fontsize=8)
    axes[0].set_yticks(range(len(labels_gt)))
    axes[0].set_yticklabels(labels_gt, fontsize=8)
    # value annotations
    for i in range(A_gt.shape[0]):
        for j in range(A_gt.shape[1]):
            if A_gt[i, j] > 0:
                axes[0].text(j, i, f"{A_gt[i,j]:.1f}", ha="center", va="center",
                             fontsize=7, color="white" if A_gt[i, j] > vmax * 0.5 else "black")

    # Pred heatmap
    im1 = axes[1].imshow(A_pred, cmap="YlOrRd", vmin=0, vmax=vmax, aspect="equal")
    axes[1].set_title("Predicted", fontweight="bold")
    axes[1].set_xticks(range(len(labels_pred)))
    axes[1].set_xticklabels(labels_pred, rotation=45, ha="right", fontsize=8)
    axes[1].set_yticks(range(len(labels_pred)))
    axes[1].set_yticklabels(labels_pred, fontsize=8)
    for i in range(A_pred.shape[0]):
        for j in range(A_pred.shape[1]):
            if A_pred[i, j] > 0:
                axes[1].text(j, i, f"{A_pred[i,j]:.1f}", ha="center", va="center",
                             fontsize=7, color="white" if A_pred[i, j] > vmax * 0.5 else "black")

    # Difference heatmap
    n = max(A_gt.shape[0], A_pred.shape[0])
    G = np.zeros((n, n))
    P = np.zeros((n, n))
    G[:A_gt.shape[0], :A_gt.shape[1]] = A_gt
    P[:A_pred.shape[0], :A_pred.shape[1]] = A_pred
    diff = P - G

    dmax = max(abs(diff.min()), abs(diff.max()), 0.1)
    im2 = axes[2].imshow(diff, cmap="RdBu_r", vmin=-dmax, vmax=dmax, aspect="equal")
    axes[2].set_title("Difference (Pred − GT)", fontweight="bold")
    all_labels = labels_gt if len(labels_gt) >= len(labels_pred) else labels_pred
    axes[2].set_xticks(range(n))
    axes[2].set_xticklabels(all_labels[:n], rotation=45, ha="right", fontsize=8)
    axes[2].set_yticks(range(n))
    axes[2].set_yticklabels(all_labels[:n], fontsize=8)
    for i in range(n):
        for j in range(n):
            if abs(diff[i, j]) > 0.01:
                axes[2].text(j, i, f"{diff[i,j]:+.1f}", ha="center", va="center",
                             fontsize=7, color="white" if abs(diff[i, j]) > dmax * 0.5 else "black")

    # colorbars
    fig.colorbar(im0, ax=axes[0], shrink=0.8, label="Weight")
    fig.colorbar(im1, ax=axes[1], shrink=0.8, label="Weight")
    fig.colorbar(im2, ax=axes[2], shrink=0.8, label="Δ Weight")

    fig.suptitle(f"Proximity Matrix Comparison — Sample {stem}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=str(_SCRIPT_DIR))
    parser.add_argument("--ckpt", type=str,
                        default=str(_SCRIPT_DIR / "checkpoints" / "segformer" / "best_model.pth"))
    parser.add_argument("--stem", type=str, default="",
                        help="Image stem to use. If empty, picks the first from test.txt")
    parser.add_argument("--out", type=str, default=str(_SCRIPT_DIR / "fig5_heatmap.pdf"))
    args = parser.parse_args()

    root = Path(args.root)

    # pick stem
    if args.stem:
        stem = args.stem
    else:
        split_file = root / "data" / "splits" / "test.txt"
        stem = split_file.read_text().splitlines()[0].strip()

    img_path  = root / "data" / "resplan_raster" / f"{stem}.png"
    mask_path = root / "data" / "resplan_masks" / f"{stem}_mask.png"

    # load and predict
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.ckpt, device)

    image_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE).astype(np.int64)
    pred_mask = predict_mask(model, image_bgr, device, IMG_SIZE)

    if gt_mask.shape != pred_mask.shape:
        gt_mask = cv2.resize(gt_mask.astype(np.uint8), pred_mask.shape[::-1],
                             interpolation=cv2.INTER_NEAREST).astype(np.int64)

    # build graphs and matrices
    G_pred = build_graph_from_segmentation(pred_mask)
    G_gt = build_gt_graph_from_polygons(mask_to_plan_dict(gt_mask))

    A_pred, labels_pred = compute_proximity_matrix(G_pred) if G_pred.number_of_nodes() > 0 else (np.zeros((0, 0)), [])
    A_gt, labels_gt = compute_proximity_matrix(G_gt) if G_gt.number_of_nodes() > 0 else (np.zeros((0, 0)), [])

    plot_heatmaps(A_gt, labels_gt, A_pred, labels_pred, stem, args.out)


if __name__ == "__main__":
    main()
