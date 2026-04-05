"""
fig_qualitative.py — Fig 6: Qualitative pipeline panel (3 rows × 7 cols)
=========================================================================
Each row is one sample. Columns:
    Input | GT Mask | Pred Mask | GT Graph | Pred Graph | GT Heatmap | Pred Heatmap

Usage:
    python fig_qualitative.py
    python fig_qualitative.py --stems 16649,5674,9470
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

from build_graph import CLASS_NAMES, build_graph_from_segmentation
from build_gt_graph import mask_to_plan_dict, build_gt_graph_from_polygons
from proximity import compute_proximity_matrix
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

# mask colour palette (class_id → RGB)
_PALETTE = np.zeros((NUM_CLASSES, 3), dtype=np.uint8)
_PALETTE[1]  = (132, 199, 129)   # Bedroom
_PALETTE[2]  = (100, 150, 200)   # Bathroom
_PALETTE[3]  = (255, 138, 101)   # Kitchen
_PALETTE[4]  = (100, 200, 100)   # Living
_PALETTE[5]  = (140, 220, 180)   # Balcony
_PALETTE[6]  = (200, 200, 200)   # Storage
_PALETTE[7]  = (220, 180, 140)   # Stair
_PALETTE[8]  = (180, 180, 180)   # Parking
_PALETTE[9]  = (140, 180, 220)   # Pool
_PALETTE[10] = (80,  80,  80)    # Wall
_PALETTE[11] = (200, 0,   0)     # Door
_PALETTE[12] = (0,   200, 200)   # Window
_PALETTE[13] = (255, 140, 0)     # FrontDoor


def _mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    return _PALETTE[mask.clip(0, NUM_CLASSES - 1)]


def _mini_heatmap(ax, matrix, labels):
    if matrix.size == 0:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return
    vmax = matrix.max() if matrix.max() > 0 else 1.0
    ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=vmax, aspect="equal")
    n = len(labels)
    ax.set_xticks(range(n))
    ax.set_xticklabels([l.split("[")[0][:6] for l in labels], rotation=45, ha="right", fontsize=5)
    ax.set_yticks(range(n))
    ax.set_yticklabels([l.split("[")[0][:6] for l in labels], fontsize=5)


def _load_pixel_scale(root):
    path = root / "pixel_scale.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def generate_panel(stems, model, device, root, save_path):
    img_dir  = root / "data" / "resplan_raster"
    mask_dir = root / "data" / "resplan_masks"
    pixel_scale_map = _load_pixel_scale(root)

    n_rows = len(stems)
    n_cols = 7
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, n_rows * 3.5))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Input", "GT Mask", "Pred Mask", "GT Graph", "Pred Graph",
                  "GT Proximity", "Pred Proximity"]

    for row, stem in enumerate(stems):
        image_bgr = cv2.imread(str(img_dir / f"{stem}.png"), cv2.IMREAD_COLOR)
        gt_mask = cv2.imread(str(mask_dir / f"{stem}_mask.png"), cv2.IMREAD_GRAYSCALE).astype(np.int64)
        pred_mask = predict_mask(model, image_bgr, device, IMG_SIZE)

        if gt_mask.shape != pred_mask.shape:
            gt_mask = cv2.resize(gt_mask.astype(np.uint8), pred_mask.shape[::-1],
                                 interpolation=cv2.INTER_NEAREST).astype(np.int64)

        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # graphs
        try:
            scale = pixel_scale_map.get(str(stem))
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

        # proximity matrices
        A_pred, l_pred = compute_proximity_matrix(G_pred) if G_pred.number_of_nodes() > 0 else (np.zeros((0, 0)), [])
        A_gt, l_gt = compute_proximity_matrix(G_gt) if G_gt.number_of_nodes() > 0 else (np.zeros((0, 0)), [])

        # col 0: input
        axes[row, 0].imshow(img_rgb)
        axes[row, 0].set_ylabel(f"Sample {stem}", fontsize=10, fontweight="bold")

        # col 1: GT mask
        axes[row, 1].imshow(_mask_to_rgb(gt_mask))

        # col 2: pred mask
        axes[row, 2].imshow(_mask_to_rgb(pred_mask))

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

        # col 5: GT heatmap
        _mini_heatmap(axes[row, 5], A_gt, l_gt)

        # col 6: pred heatmap
        _mini_heatmap(axes[row, 6], A_pred, l_pred)

        # turn off ticks for image columns
        for c in range(n_cols):
            if c not in (5, 6):
                axes[row, c].set_xticks([])
                axes[row, c].set_yticks([])

    # column titles
    for c, title in enumerate(col_titles):
        axes[0, c].set_title(title, fontweight="bold", fontsize=11)

    fig.suptitle("Qualitative Pipeline Results", fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=str(_SCRIPT_DIR))
    parser.add_argument("--ckpt", type=str,
                        default=str(_SCRIPT_DIR / "checkpoints" / "segformer" / "best_model.pth"))
    parser.add_argument("--stems", type=str, default="",
                        help="Comma-separated image stems. If empty, picks 3 good samples from per_image.csv")
    parser.add_argument("--out", type=str, default=str(_SCRIPT_DIR / "fig6_qualitative.pdf"))
    args = parser.parse_args()

    root = Path(args.root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.ckpt, device)

    if args.stems:
        stems = [s.strip() for s in args.stems.split(",")]
    else:
        # pick 3 samples: best, median, slightly-below-median edge_f1
        csv_path = root / "results" / "eval_20260403_155913" / "per_image.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path).sort_values("edge_f1", ascending=False)
            n = len(df)
            stems = [
                str(int(df.iloc[0]["stem"])),              # best
                str(int(df.iloc[n // 2]["stem"])),          # median
                str(int(df.iloc[int(n * 0.75)]["stem"])),   # 75th percentile (below median)
            ]
        else:
            split_file = root / "data" / "splits" / "test.txt"
            stems = [l.strip() for l in split_file.read_text().splitlines()[:3]]

    print(f"[samples] {stems}")
    generate_panel(stems, model, device, root, args.out)


if __name__ == "__main__":
    main()
