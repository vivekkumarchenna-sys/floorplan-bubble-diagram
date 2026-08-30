"""
fig_training.py - Fig 2 & Fig 3: Training vs validation loss and mIoU curves
=============================================================================
Reads history_segformer.json and generates two publication-quality PDF plots.

Usage:
    python fig_training.py
    python fig_training.py --history path/to/history_segformer.json --out results/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    _SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    _SCRIPT_DIR = Path("/content/drive/MyDrive/bubble_diagram_project")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def plot_loss(history: list[dict], save_path: str):
    """Fig 2: Training vs validation loss."""
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(epochs, train_loss, "-o", color="#4472C4", markersize=4,
            linewidth=1.8, label="Train Loss")
    ax.plot(epochs, val_loss, "-s", color="#ED7D31", markersize=4,
            linewidth=1.8, label="Val Loss")

    # mark best val loss
    best_idx = int(np.argmin(val_loss))
    ax.annotate(
        f"Best: {val_loss[best_idx]:.4f}\n(epoch {epochs[best_idx]})",
        xy=(epochs[best_idx], val_loss[best_idx]),
        xytext=(epochs[best_idx] + 2, val_loss[best_idx] + 0.05),
        arrowprops=dict(arrowstyle="->", color="#ED7D31", lw=1.2),
        fontsize=9, color="#ED7D31", fontweight="bold",
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training vs Validation Loss", fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9, edgecolor="#cccccc")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(epochs[0] - 0.5, epochs[-1] + 0.5)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {save_path}")


def plot_miou(history: list[dict], save_path: str):
    """Fig 3: Training vs validation mIoU."""
    epochs = [h["epoch"] for h in history]
    train_miou = [h["train_mIoU"] for h in history]
    val_miou = [h["val_mIoU"] for h in history]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(epochs, train_miou, "-o", color="#4472C4", markersize=4,
            linewidth=1.8, label="Train mIoU")
    ax.plot(epochs, val_miou, "-s", color="#ED7D31", markersize=4,
            linewidth=1.8, label="Val mIoU")

    # mark best val mIoU
    best_idx = int(np.argmax(val_miou))
    ax.annotate(
        f"Best: {val_miou[best_idx]:.4f}\n(epoch {epochs[best_idx]})",
        xy=(epochs[best_idx], val_miou[best_idx]),
        xytext=(epochs[best_idx] - 7, val_miou[best_idx] - 0.22),
        arrowprops=dict(arrowstyle="->", color="#ED7D31", lw=1.2),
        fontsize=9, color="#ED7D31", fontweight="bold",
        ha="center",
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("mIoU")
    ax.set_title("Training vs Validation mIoU", fontweight="bold")
    ax.legend(loc="lower right", framealpha=0.9, edgecolor="#cccccc")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(epochs[0] - 0.5, epochs[-1] + 0.5)
    ax.set_ylim(0, 1.05)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=str,
                        default=str(_SCRIPT_DIR.parent / "history_segformer.json"))
    parser.add_argument("--out", type=str, default=str(_SCRIPT_DIR))
    args = parser.parse_args()

    with open(args.history) as f:
        history = json.load(f)

    out = Path(args.out)
    plot_loss(history, str(out / "fig2_loss.pdf"))
    plot_miou(history, str(out / "fig3_miou.pdf"))


if __name__ == "__main__":
    main()
