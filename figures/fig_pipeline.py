"""
fig_pipeline.py — Fig 1: Pipeline overview diagram (minimal)
=============================================================
Simple left-to-right flow: Input → M1 → M2 → M3 → M4

Usage:
    python fig_pipeline.py
    python fig_pipeline.py --out fig1_pipeline.pdf
"""

from __future__ import annotations

import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# muted palette
_INPUT_CLR = ("#E8EEF5", "#7A8FA6")
_MODULE_CLR = ("#F5F0E8", "#A69A7A")
_OUTPUT_CLR = ("#E8F5EA", "#7AA67E")


def _box(ax, cx, cy, w, h, label, sub, fill, edge):
    box = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.015",
        facecolor=fill, edgecolor=edge, linewidth=1.2,
        zorder=3,
    )
    ax.add_patch(box)
    ax.text(cx, cy + 0.012, label, ha="center", va="center",
            fontsize=9.5, fontweight="bold", zorder=4)
    ax.text(cx, cy - 0.016, sub, ha="center", va="center",
            fontsize=7, color="#555555", zorder=4)


def _arrow(ax, x1, x2, y):
    arrow = FancyArrowPatch(
        (x1, y), (x2, y),
        arrowstyle="->,head_width=4,head_length=4",
        color="#888888", linewidth=1.2, zorder=2,
    )
    ax.add_patch(arrow)


def generate_pipeline(save_path: str = "fig1_pipeline.pdf"):
    fig, ax = plt.subplots(figsize=(14, 2.2))
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.04, 0.1)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")

    bw = 0.14
    bh = 0.065
    gap = 0.045
    y = 0.03

    # 5 boxes: input + 4 modules
    steps = [
        ("Floor Plan\nImage",         "",                    _INPUT_CLR),
        ("Segmentation",              "M1  SegFormer-B3",    _MODULE_CLR),
        ("Graph\nConstruction",       "M2  Typed adjacency", _MODULE_CLR),
        ("Proximity\nMatrix",         "M3  Weighted N\u00d7N",  _MODULE_CLR),
        ("Bubble\nDiagram",           "M4  F-R layout",      _OUTPUT_CLR),
    ]

    xs = []
    x = 0.04
    for _ in steps:
        xs.append(x + bw / 2)
        x += bw + gap

    for i, (label, sub, (fill, edge)) in enumerate(steps):
        _box(ax, xs[i], y, bw, bh, label, sub, fill, edge)

    for i in range(len(steps) - 1):
        _arrow(ax, xs[i] + bw / 2 + 0.005, xs[i + 1] - bw / 2 - 0.005, y)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved -> {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="fig1_pipeline.pdf")
    args = parser.parse_args()
    generate_pipeline(args.out)
