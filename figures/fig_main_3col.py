"""
fig_main_3col.py - Clean 3-column main-text qualitative / failure figure.

Each row is one plan. Columns:
    (a) Rasterised floor plan (input) | (b) Predicted label map |
    (c) Extracted typed bubble diagram
with a per-plan metrics line beneath each row. Uses the current (corrected)
palette and bubble-diagram layout from src/, so it matches the mask overlays
and the appendix multi-panel figures.

    python fig_main_3col.py --stems 15389,16649,13388 --out fig6_main.pdf
    python fig_main_3col.py --stems 476,7463,8835,11211 --out fig7_main.pdf --title "Failure analysis"
"""
from __future__ import annotations
import argparse, sys, json
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
_SRC = _SCRIPT_DIR.parent / "src"
if _SRC.is_dir():
    sys.path.insert(0, str(_SRC))

import cv2, numpy as np, pandas as pd, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.gridspec import GridSpec

from build_graph import build_graph_from_segmentation
from visualize import (draw_bubble_diagram, ROOM_COLORS, EDGE_STYLES,
                       _FALLBACK_COLOR, _style_line)
from inference import load_model, predict_mask, IMG_SIZE, NUM_CLASSES, _PALETTE as _BGR

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 11, "figure.dpi": 300, "savefig.dpi": 300,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
_PALETTE = _BGR[:, ::-1].copy()


def _mask_to_rgb(m):
    return _PALETTE[m.clip(0, NUM_CLASSES - 1)]


def _load_scale(root):
    p = root / "pixel_scale.json"
    return json.load(open(p)) if p.exists() else {}


def _metrics_line(row):
    if row is None:
        return ""
    return (f"Plan {int(row['stem'])}: {int(row['n_rooms_gt'])} rooms, "
            f"{int(row['n_edges_gt'])} ground-truth edges.  "
            f"Edge F1 {row['edge_f1']:.3f}  ·  edge-type accuracy {row['type_accuracy']:.2f}  "
            f"·  GED {row['ged']:.0f}  ·  {int(row['n_edges_pred'])} predicted edges")


def generate(stems, model, device, root, save_path, df, title):
    img_dir = root / "data" / "resplan_raster"
    mask_dir = root / "data" / "resplan_masks"
    scale_map = _load_scale(root)
    n = len(stems)

    fig = plt.figure(figsize=(10.5, n * 3.75 + 0.9))
    # per plan: one tall image row + one thin metrics row
    gs = GridSpec(2 * n, 3, figure=fig,
                  height_ratios=sum([[1.0, 0.14] for _ in range(n)], []),
                  hspace=0.05, wspace=0.06)

    seen_classes, seen_edges = [], []
    col_titles = ["(a) Rasterised floor plan (input)",
                  "(b) Predicted label map",
                  "(c) Extracted typed bubble diagram"]

    for i, stem in enumerate(stems):
        bgr = cv2.imread(str(img_dir / f"{stem}.png"), cv2.IMREAD_COLOR)
        pred = predict_mask(model, bgr, device, IMG_SIZE)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        try:
            G = build_graph_from_segmentation(pred, pixel_scale=scale_map.get(str(stem)))
        except Exception:
            import networkx as nx
            G = nx.Graph()

        ax0 = fig.add_subplot(gs[2 * i, 0]); ax0.imshow(rgb)
        ax1 = fig.add_subplot(gs[2 * i, 1]); ax1.imshow(_mask_to_rgb(pred))
        ax2 = fig.add_subplot(gs[2 * i, 2])
        if G.number_of_nodes() > 0:
            draw_bubble_diagram(G, ax=ax2, title="", show_legend=False)
            # off-node labels are offset in points and can spill past the axis;
            # add top/side headroom so they never collide with the column title
            y0, y1 = ax2.get_ylim(); ax2.set_ylim(y0, y1 + 0.16 * (y1 - y0))
            x0, x1 = ax2.get_xlim(); ax2.set_xlim(x0 - 0.06 * (x1 - x0), x1 + 0.06 * (x1 - x0))
        else:
            ax2.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax2.transAxes)
        for ax in (ax0, ax1, ax2):
            ax.set_xticks([]); ax.set_yticks([])
        ax0.set_ylabel(f"Plan {stem}", fontsize=12, fontweight="bold")
        if i == 0:
            for ax, t in zip((ax0, ax1, ax2), col_titles):
                ax.set_title(t, fontsize=12, fontweight="bold", pad=14)

        for nid in G.nodes():
            c = G.nodes[nid]["class_name"]
            if c not in seen_classes:
                seen_classes.append(c)
        for _, _, d in G.edges(data=True):
            et = d.get("edge_type")
            if et in EDGE_STYLES and et not in seen_edges:
                seen_edges.append(et)

        row = df[df["stem"] == int(stem)]
        mrow = row.iloc[0] if len(row) else None
        tax = fig.add_subplot(gs[2 * i + 1, :]); tax.axis("off")
        tax.text(0.5, 0.6, _metrics_line(mrow), ha="center", va="center",
                 fontsize=10.5, transform=tax.transAxes)

    handles = [mpatches.Patch(facecolor=ROOM_COLORS[c], edgecolor="#333333", linewidth=1.0, label=c)
               for c in ROOM_COLORS if c in seen_classes]
    handles += [mpatches.Patch(facecolor=_FALLBACK_COLOR, edgecolor="#333333", linewidth=1.0, label=c)
                for c in seen_classes if c not in ROOM_COLORS]
    handles += [mlines.Line2D([], [], label=et, **_style_line(et)) for et in EDGE_STYLES if et in seen_edges]

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=0.998)
    fig.tight_layout(rect=[0, 0.03, 1, 0.99])
    if handles:
        fig.legend(handles=handles, title="Room types & edges", loc="lower center",
                   ncol=len(handles), bbox_to_anchor=(0.5, 0.0), frameon=True, fontsize=10)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {save_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(_SCRIPT_DIR.parent))
    ap.add_argument("--ckpt", default=str(_SCRIPT_DIR.parent / "checkpoints" / "segformer" / "best_model.pth"))
    ap.add_argument("--csv", default=str(_SCRIPT_DIR.parent / "per_image.csv"))
    ap.add_argument("--stems", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    a = ap.parse_args()
    root = Path(a.root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(a.ckpt, device)
    df = pd.read_csv(a.csv)
    stems = [s.strip() for s in a.stems.split(",")]
    generate(stems, model, device, root, a.out, df, a.title)


if __name__ == "__main__":
    main()
