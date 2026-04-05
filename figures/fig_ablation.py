"""
fig_ablation.py — Publication-quality ablation bar chart (Fig 4)
================================================================
Grouped bars: Edge F1 (blue) + mIoU (orange) with dual y-axes.
Outputs 300 DPI PDF.

Usage:
    # with placeholder data (edit values below once runs complete)
    python fig_ablation.py

    # or import and pass your own data
    from fig_ablation import plot_ablation
    plot_ablation(variants, edge_f1, miou, save_path="fig4_ablation.pdf")
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
# Ablation data — UPDATE these with actual values from your 6 runs
# ══════════════════════════════════════════════════════════════════════════════

VARIANTS = [
    "Full\npipeline",
    "w/o door\nedges",
    "w/o room\ntypes",
    "w/o corridor\nadj.",
    "w/o post-\nprocessing",
    "w/o edge\ntyping",
]

# Placeholder values — replace with real numbers from inference runs
EDGE_F1 = [0.73, 0.51, 0.68, 0.65, 0.42, 0.70]
MIOU    = [0.965, 0.965, 0.965, 0.965, 0.948, 0.965]


# ══════════════════════════════════════════════════════════════════════════════
# Plot function
# ══════════════════════════════════════════════════════════════════════════════

def plot_ablation(
    variants: list[str] = VARIANTS,
    edge_f1: list[float] = EDGE_F1,
    miou: list[float] = MIOU,
    save_path: str = "fig4_ablation.pdf",
):
    # font setup
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 13,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,       # TrueType in PDF (editable text)
        "ps.fonttype": 42,
    })

    n = len(variants)
    x = np.arange(n)
    bar_w = 0.35

    fig, ax1 = plt.subplots(figsize=(8, 4.5))

    # ── Edge F1 bars (left y-axis) ────────────────────────────────────────────
    color_f1 = "#4472C4"
    bars1 = ax1.bar(
        x - bar_w / 2, edge_f1, bar_w,
        label="Edge F1",
        color=color_f1, edgecolor="white", linewidth=0.5,
        zorder=3,
    )

    ax1.set_ylabel("Edge F1", color=color_f1, fontweight="bold")
    ax1.set_ylim(0, 1.05)
    ax1.tick_params(axis="y", colors=color_f1)
    ax1.spines["left"].set_color(color_f1)

    # ── mIoU bars (right y-axis) ─────────────────────────────────────────────
    ax2 = ax1.twinx()
    color_miou = "#ED7D31"
    bars2 = ax2.bar(
        x + bar_w / 2, miou, bar_w,
        label="mIoU",
        color=color_miou, edgecolor="white", linewidth=0.5,
        zorder=3,
    )

    ax2.set_ylabel("mIoU", color=color_miou, fontweight="bold")
    ax2.set_ylim(0.90, 1.002)
    ax2.tick_params(axis="y", colors=color_miou)
    ax2.spines["right"].set_color(color_miou)

    # ── value labels on bars ─────────────────────────────────────────────────
    for bar in bars1:
        h = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2, h + 0.012,
            f"{h:.2f}", ha="center", va="bottom",
            fontsize=8, fontweight="bold", color=color_f1,
        )

    for bar in bars2:
        h = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2, h + 0.001,
            f"{h:.3f}", ha="center", va="bottom",
            fontsize=8, fontweight="bold", color=color_miou,
        )

    # ── x-axis ───────────────────────────────────────────────────────────────
    ax1.set_xticks(x)
    ax1.set_xticklabels(variants, ha="center")
    ax1.set_xlabel("Ablation Variant")

    # ── highlight the full pipeline bar ──────────────────────────────────────
    bars1[0].set_edgecolor("#2F5496")
    bars1[0].set_linewidth(1.5)
    bars2[0].set_edgecolor("#C55A11")
    bars2[0].set_linewidth(1.5)

    # ── grid + styling ───────────────────────────────────────────────────────
    ax1.set_axisbelow(True)
    ax1.yaxis.grid(True, linestyle="--", alpha=0.3, zorder=0)
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    # ── legend ───────────────────────────────────────────────────────────────
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2, labels1 + labels2,
        loc="upper right", framealpha=0.9, edgecolor="#cccccc",
    )

    # ── title ────────────────────────────────────────────────────────────────
    ax1.set_title("Ablation Study: Component Contributions", fontweight="bold", pad=12)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved → {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    plot_ablation()
