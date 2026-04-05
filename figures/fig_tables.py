"""
fig_tables.py — Generate all 5 paper tables as publication-quality PDFs
=======================================================================
Each table is rendered as a matplotlib figure and saved as a PDF.

Required CSV files:
    Table 1: No CSV needed — values hardcoded from compute_split_stats.py output
    Table 2: class_iou.csv (from inference.py)
    Table 3: No CSV needed — values hardcoded (model specs + results)
    Table 4: summary.csv (from inference.py) + per_image.csv (for GED std dev)
    Table 5: results/ablation/ablation_results.csv (from run_ablation.py)

Usage:
    python fig_tables.py
    python fig_tables.py --out results/tables/
"""

from __future__ import annotations

import argparse
import csv
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
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def _render_table(col_labels, row_data, title, save_path, col_widths=None):
    """Render a table as a matplotlib figure and save as PDF."""
    n_rows = len(row_data)
    n_cols = len(col_labels)

    fig_height = 1.0 + n_rows * 0.4
    fig_width = max(10, n_cols * 1.8)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=row_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    # style header
    for j in range(n_cols):
        cell = table[0, j]
        cell.set_facecolor("#4472C4")
        cell.set_text_props(color="white", fontweight="bold", fontsize=9)

    # alternate row colors
    for i in range(1, n_rows + 1):
        for j in range(n_cols):
            cell = table[i, j]
            if i % 2 == 0:
                cell.set_facecolor("#D9E2F3")
            else:
                cell.set_facecolor("#FFFFFF")

    if col_widths:
        for j, w in enumerate(col_widths):
            for i in range(n_rows + 1):
                table[i, j].set_width(w)

    ax.set_title(title, fontsize=12, fontweight="bold", pad=20)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved → {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Table 1: Dataset Partition Statistics
# Source: compute_split_stats.py output (hardcoded)
# ══════════════════════════════════════════════════════════════════════════════

def table1_dataset(save_path):
    col_labels = ["Split", "Floor Plans", "Room Instances", "Door Instances",
                  "Mean Rooms/Plan", "Mean Doors/Plan"]
    row_data = [
        ["Train",  "11,974", "98,332",  "83,247",  "8.21", "6.95"],
        ["Val",    "2,566",  "21,141",  "17,913",  "8.24", "6.98"],
        ["Test",   "2,567",  "21,176",  "17,939",  "8.25", "6.99"],
        ["Total",  "17,107", "140,649", "119,099", "—",    "—"],
    ]
    _render_table(col_labels, row_data,
                  "Table 1: Dataset Partition Statistics", save_path)


# ══════════════════════════════════════════════════════════════════════════════
# Table 2: Per-Class Segmentation Performance
# Source: class_iou.csv
# ══════════════════════════════════════════════════════════════════════════════

def table2_class_iou(save_path, csv_path):
    col_labels = ["Class", "Precision", "Recall", "F1", "IoU"]
    row_data = []

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            name = r["class_name"]
            if name.startswith("class_"):
                continue  # skip dead classes
            p = float(r["precision"])
            rc = float(r["recall"])
            iou = float(r["iou"])
            f1 = 2 * p * rc / (p + rc) if (p + rc) > 0 else 0.0

            row_data.append([
                name,
                f"{p:.4f}",
                f"{rc:.4f}",
                f"{f1:.4f}",
                f"{iou:.4f}",
            ])

    # add mean row
    ious = [float(row[4]) for row in row_data if float(row[4]) > 0]
    mean_iou = sum(ious) / len(ious) if ious else 0.0
    precs = [float(row[1]) for row in row_data if float(row[1]) > 0]
    recs = [float(row[2]) for row in row_data if float(row[2]) > 0]
    f1s = [float(row[3]) for row in row_data if float(row[3]) > 0]
    row_data.append([
        "Mean (active)",
        f"{sum(precs)/len(precs):.4f}" if precs else "—",
        f"{sum(recs)/len(recs):.4f}" if recs else "—",
        f"{sum(f1s)/len(f1s):.4f}" if f1s else "—",
        f"{mean_iou:.4f}",
    ])

    _render_table(col_labels, row_data,
                  "Table 2: Per-Class Segmentation Performance (Test Set)", save_path)


# ══════════════════════════════════════════════════════════════════════════════
# Table 3: Backbone Comparison
# Source: hardcoded (model specs from published papers + our results)
# ══════════════════════════════════════════════════════════════════════════════

def table3_backbone(save_path, deeplab_json=None):
    col_labels = ["Method", "Params (M)", "FLOPs (G)", "mIoU",
                  "GPU-hours", "Inference (ms/img)"]

    # DeepLabV3+ results from history_deeplab.json if available
    dl_miou = "—"
    dl_hours = "—"
    if deeplab_json is not None and Path(deeplab_json).exists():
        import json
        with open(deeplab_json) as f:
            h = json.load(f)
        best = max(h, key=lambda x: x["val_mIoU"])
        dl_miou = f"{best['val_mIoU']:.4f}"
        total_s = sum(e["elapsed_s"] for e in h)
        dl_hours = f"~{total_s/3600:.1f} (A100)"

    row_data = [
        ["SegFormer-B3", "47.3", "79.0", "0.9974", "~3 (A100)", "~20"],
        ["DeepLabV3+ (R101)", "59.3", "83.6", dl_miou, dl_hours, "~35"],
    ]
    _render_table(col_labels, row_data,
                  "Table 3: Backbone Comparison", save_path)


# ══════════════════════════════════════════════════════════════════════════════
# Table 4: Adjacency Graph Extraction Performance
# Source: summary.csv + per_image.csv (for GED std dev)
# ══════════════════════════════════════════════════════════════════════════════

def table4_graph(save_path, summary_csv, per_image_csv):
    # read summary
    with open(summary_csv) as f:
        reader = csv.DictReader(f)
        s = next(reader)

    # compute GED std dev from per_image
    geds = []
    with open(per_image_csv) as f:
        for r in csv.DictReader(f):
            geds.append(float(r["ged"]))

    ged_mean = sum(geds) / len(geds)
    ged_std = (sum((g - ged_mean) ** 2 for g in geds) / len(geds)) ** 0.5

    col_labels = ["Method", "Edge Prec.", "Edge Rec.", "Edge F1",
                  "GED (mean ± SD)", "Edge-type Acc."]
    row_data = [
        [
            "Dilation-based (ours)",
            f"{float(s['edge_precision']):.4f}",
            f"{float(s['edge_recall']):.4f}",
            f"{float(s['edge_f1']):.4f}",
            f"{ged_mean:.2f} ± {ged_std:.2f}",
            f"{float(s['type_accuracy']):.4f}",
        ],
    ]
    _render_table(col_labels, row_data,
                  "Table 4: Adjacency Graph Extraction Performance", save_path)


# ══════════════════════════════════════════════════════════════════════════════
# Table 5: Ablation Results
# Source: results/ablation/ablation_results.csv
# ══════════════════════════════════════════════════════════════════════════════

def table5_ablation(save_path, csv_path):
    col_labels = ["Variant", "mIoU", "Edge F1", "GED", "Frobenius"]
    row_data = []

    with open(csv_path) as f:
        for r in csv.DictReader(f):
            row_data.append([
                r["variant"],
                f"{float(r['mIoU']):.4f}",
                f"{float(r['edge_f1']):.4f}",
                f"{float(r['ged']):.2f}",
                f"{float(r['frobenius']):.4f}",
            ])

    _render_table(col_labels, row_data,
                  "Table 5: Ablation Study Results", save_path)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=str(_SCRIPT_DIR))
    parser.add_argument("--out", type=str, default=str(_SCRIPT_DIR / "tables"))
    args = parser.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Table 1: no CSV needed
    table1_dataset(out / "table1_dataset.pdf")

    # Table 2: class_iou.csv
    class_iou_csv = root / "class_iou.csv"
    if not class_iou_csv.exists():
        class_iou_csv = root / "results" / "eval_20260403_155913" / "class_iou.csv"
    if class_iou_csv.exists():
        table2_class_iou(out / "table2_class_iou.pdf", class_iou_csv)
    else:
        print(f"[skip] Table 2 — class_iou.csv not found")

    # Table 3: backbone comparison (reads history_deeplab.json if available)
    deeplab_json = root / "results" / "history_deeplab.json"
    if not deeplab_json.exists():
        deeplab_json = root / "history_deeplab.json"
    table3_backbone(out / "table3_backbone.pdf",
                    deeplab_json=deeplab_json if deeplab_json.exists() else None)

    # Table 4: summary.csv + per_image.csv
    summary_csv = root / "summary.csv"
    per_image_csv = root / "per_image.csv"
    if not summary_csv.exists():
        summary_csv = root / "results" / "eval_20260403_155913" / "summary.csv"
    if not per_image_csv.exists():
        per_image_csv = root / "results" / "eval_20260403_155913" / "per_image.csv"
    if summary_csv.exists() and per_image_csv.exists():
        table4_graph(out / "table4_graph.pdf", summary_csv, per_image_csv)
    else:
        print(f"[skip] Table 4 — summary.csv or per_image.csv not found")

    # Table 5: ablation_results.csv
    ablation_csv = root / "results" / "ablation" / "ablation_results.csv"
    if ablation_csv.exists():
        table5_ablation(out / "table5_ablation.pdf", ablation_csv)
    else:
        print(f"[skip] Table 5 — ablation_results.csv not found")

    print(f"\nAll tables saved to: {out}")


if __name__ == "__main__":
    main()
