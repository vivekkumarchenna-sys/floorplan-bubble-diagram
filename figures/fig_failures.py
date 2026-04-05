"""
fig_failures.py — Fig 7: Failure cases (3–5 worst plans annotated)
==================================================================
Reads per_image.csv, picks the worst edge_f1 samples, and generates
an annotated panel showing what went wrong.

Each row: Input | GT Mask | Pred Mask | GT Graph | Pred Graph | Metrics

Usage:
    python fig_failures.py
    python fig_failures.py --n 5 --out fig7_failures.pdf
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

import json

import cv2
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from build_graph import build_graph_from_segmentation
from build_gt_graph import mask_to_plan_dict, build_gt_graph_from_polygons
from visualize import draw_bubble_diagram
from inference import load_model, predict_mask, IMG_SIZE, NUM_CLASSES

# ── style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8,
    "axes.titlesize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

_PALETTE = np.zeros((NUM_CLASSES, 3), dtype=np.uint8)
_PALETTE[1]  = (132, 199, 129)
_PALETTE[2]  = (100, 150, 200)
_PALETTE[3]  = (255, 138, 101)
_PALETTE[4]  = (100, 200, 100)
_PALETTE[5]  = (140, 220, 180)
_PALETTE[6]  = (200, 200, 200)
_PALETTE[7]  = (220, 180, 140)
_PALETTE[8]  = (180, 180, 180)
_PALETTE[9]  = (140, 180, 220)
_PALETTE[10] = (80,  80,  80)
_PALETTE[11] = (200, 0,   0)
_PALETTE[12] = (0,   200, 200)
_PALETTE[13] = (255, 140, 0)


def _mask_to_rgb(mask):
    return _PALETTE[mask.clip(0, NUM_CLASSES - 1)]


def _load_pixel_scale(root):
    path = root / "pixel_scale.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def generate_failures(df_worst, model, device, root, save_path):
    img_dir  = root / "data" / "resplan_raster"
    mask_dir = root / "data" / "resplan_masks"
    pixel_scale_map = _load_pixel_scale(root)

    n_rows = len(df_worst)
    n_cols = 6
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, n_rows * 3.5))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Input", "GT Mask", "Pred Mask", "GT Graph", "Pred Graph", "Failure Analysis"]

    for row, (_, info) in enumerate(df_worst.iterrows()):
        stem = str(int(info["stem"]) if isinstance(info["stem"], float) else info["stem"])

        image_bgr = cv2.imread(str(img_dir / f"{stem}.png"), cv2.IMREAD_COLOR)
        gt_mask = cv2.imread(str(mask_dir / f"{stem}_mask.png"), cv2.IMREAD_GRAYSCALE).astype(np.int64)
        pred_mask = predict_mask(model, image_bgr, device, IMG_SIZE)

        if gt_mask.shape != pred_mask.shape:
            gt_mask = cv2.resize(gt_mask.astype(np.uint8), pred_mask.shape[::-1],
                                 interpolation=cv2.INTER_NEAREST).astype(np.int64)

        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        scale = pixel_scale_map.get(stem)
        try:
            G_pred = build_graph_from_segmentation(pred_mask, pixel_scale=scale)
        except Exception:
            import networkx as nx
            G_pred = nx.Graph()
        try:
            G_gt = build_gt_graph_from_polygons(mask_to_plan_dict(gt_mask))
            if scale is not None:
                for nid in G_gt.nodes():
                    area_px = G_gt.nodes[nid].get("area", 0)
                    G_gt.nodes[nid]["area_sqm"] = round(area_px * scale, 2)
        except Exception:
            import networkx as nx
            G_gt = nx.Graph()

        # col 0: input
        axes[row, 0].imshow(img_rgb)
        axes[row, 0].set_ylabel(f"#{stem}", fontsize=10, fontweight="bold")

        # col 1: GT mask
        axes[row, 1].imshow(_mask_to_rgb(gt_mask))

        # col 2: pred mask — highlight errors
        pred_rgb = _mask_to_rgb(pred_mask)
        error_mask = (pred_mask != gt_mask)
        pred_rgb[error_mask] = [255, 0, 0]  # red for errors
        axes[row, 2].imshow(pred_rgb)

        # col 3: GT graph
        if G_gt.number_of_nodes() > 0:
            draw_bubble_diagram(G_gt, ax=axes[row, 3], title="")
        else:
            axes[row, 3].text(0.5, 0.5, "N/A", ha="center", va="center",
                              transform=axes[row, 3].transAxes)

        # col 4: pred graph
        if G_pred.number_of_nodes() > 0:
            draw_bubble_diagram(G_pred, ax=axes[row, 4], title="")
        else:
            axes[row, 4].text(0.5, 0.5, "N/A", ha="center", va="center",
                              transform=axes[row, 4].transAxes)

        # col 5: metrics annotation
        ax = axes[row, 5]
        ax.axis("off")
        error_pct = error_mask.sum() / error_mask.size * 100

        metrics_text = (
            f"mIoU: {info['mIoU']:.4f}\n"
            f"Edge F1: {info['edge_f1']:.4f}\n"
            f"Edge Prec: {info['edge_precision']:.4f}\n"
            f"Edge Rec: {info['edge_recall']:.4f}\n"
            f"Type Acc: {info['type_accuracy']:.4f}\n"
            f"GED: {info['ged']:.0f}\n"
            f"Pixel Error: {error_pct:.1f}%\n"
            f"\n"
            f"Rooms: {int(info['n_rooms_pred'])} pred / {int(info['n_rooms_gt'])} GT\n"
            f"Edges: {int(info['n_edges_pred'])} pred / {int(info['n_edges_gt'])} GT"
        )
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
                fontsize=9, verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff3e0", edgecolor="#ff9800"))

        for c in range(5):
            axes[row, c].set_xticks([])
            axes[row, c].set_yticks([])

    for c, title in enumerate(col_titles):
        axes[0, c].set_title(title, fontweight="bold", fontsize=11)

    fig.suptitle("Failure Case Analysis — Worst Edge F1 Samples",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=str(_SCRIPT_DIR))
    parser.add_argument("--ckpt", type=str,
                        default=str(_SCRIPT_DIR / "checkpoints" / "segformer" / "best_model.pth"))
    parser.add_argument("--csv", type=str, default=str(_SCRIPT_DIR / "results" / "eval_20260403_155913" / "per_image.csv"))
    parser.add_argument("--n", type=int, default=4, help="Number of failure cases")
    parser.add_argument("--out", type=str, default=str(_SCRIPT_DIR / "fig7_failures.pdf"))
    args = parser.parse_args()

    root = Path(args.root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.ckpt, device)

    df = pd.read_csv(args.csv)
    df_worst = df.nsmallest(args.n, "edge_f1")
    print(f"[failures] Worst {args.n} samples by edge_f1:")
    print(df_worst[["stem", "mIoU", "edge_f1", "ged"]].to_string(index=False))

    generate_failures(df_worst, model, device, root, args.out)


if __name__ == "__main__":
    main()
