"""
fig_tables.py - Generate all 5 paper tables as publication-quality PDFs
=======================================================================
Each table is rendered as a matplotlib figure and saved as a PDF.

Required CSV files:
    Table 1: No CSV needed - values hardcoded from compute_split_stats.py output
    Table 2: class_iou.csv (from inference.py)
    Table 3: history_segformer.json + history_deeplab.json (val mIoU + GPU-hours)
    Table 4: summary.csv (from inference.py) + per_image.csv (for GED std dev)
    Table 5: results/ablation_FINAL/ablation_results.csv (from run_ablation.py)

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
    print(f"Saved -> {save_path}")


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
        ["Total",  "17,107", "140,649", "119,099", " - ",    " - "],
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
            # classes that never occur in the split (e.g. Pool, n_plans=0)
            # have blank precision/recall/iou cells - skip them rather than
            # crash on float("").
            if not (r.get("precision") or "").strip():
                continue
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
        f"{sum(precs)/len(precs):.4f}" if precs else " - ",
        f"{sum(recs)/len(recs):.4f}" if recs else " - ",
        f"{sum(f1s)/len(f1s):.4f}" if f1s else " - ",
        f"{mean_iou:.4f}",
    ])

    _render_table(col_labels, row_data,
                  "Table 2: Per-Class Segmentation Performance (Test Set)", save_path)


# ══════════════════════════════════════════════════════════════════════════════
# Table 3: Backbone Comparison
# Source: history_segformer.json + history_deeplab.json (params are model specs)
# ══════════════════════════════════════════════════════════════════════════════

def table3_backbone(save_path, segformer_json=None, deeplab_json=None):
    # Both mIoU values are validation macro (mean-over-class) mIoU, so the two
    # backbones are compared on the same metric (an earlier version compared
    # SegFormer's test per-image mIoU against DeepLab's val macro mIoU - not the
    # same quantity). DeepLabV3+ is marginally higher. FLOPs / inference-ms
    # columns were removed: those numbers were not measured anywhere in this
    # repo. Parameter counts are exact (47,236,304 / 59,452,560).
    col_labels = ["Method", "Params (M)", "Val mIoU (macro)", "GPU-hours"]

    def _read(js):
        miou, hours = " - ", " - "
        if js is not None and Path(js).exists():
            import json
            with open(js) as f:
                h = json.load(f)
            best = max(h, key=lambda x: x.get("val_mIoU", -1))
            miou = f"{best['val_mIoU']:.4f}"
            if h and "elapsed_s" in h[0]:
                hours = f"~{sum(e.get('elapsed_s', 0) for e in h) / 3600:.1f} (A100)"
        return miou, hours

    sf_miou, _sf_hours = _read(segformer_json)
    dl_miou, dl_hours = _read(deeplab_json)
    # SegFormer training is reported as ~3 GPU-hours (2.72 h wall-clock) in the
    # paper/README; keep that rounding here for consistency.
    sf_hours = "~3 (A100)"

    row_data = [
        ["SegFormer-B3",      "47.2", sf_miou, sf_hours],
        ["DeepLabV3+ (R101)", "59.5", dl_miou, dl_hours],
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

    # compute GED mean/std from per_image, skipping blank/NaN cells: GED is
    # only recorded for graphs that converged within the timeout, so some rows
    # have an empty ged. Report the count that actually contributed (n).
    geds = []
    with open(per_image_csv) as f:
        for r in csv.DictReader(f):
            v = (r.get("ged") or "").strip()
            if v == "" or v.lower() in ("nan", "na"):
                continue
            try:
                g = float(v)
            except ValueError:
                continue
            if g != g:  # NaN
                continue
            geds.append(g)

    n_ged = len(geds)
    if n_ged:
        ged_mean = sum(geds) / n_ged
        ged_std = (sum((g - ged_mean) ** 2 for g in geds) / n_ged) ** 0.5
        ged_str = f"{ged_mean:.2f} ± {ged_std:.2f} (n={n_ged})"
    else:
        ged_str = "n/a (n=0)"

    col_labels = ["Method", "Edge Prec.", "Edge Rec.", "Edge F1",
                  "GED (mean ± SD)", "Edge-type Acc."]
    row_data = [
        [
            "Dilation-based (ours)",
            f"{float(s['edge_precision']):.4f}",
            f"{float(s['edge_recall']):.4f}",
            f"{float(s['edge_f1']):.4f}",
            ged_str,
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
            # GED is a mean over only the graphs that converged within the
            # timeout; the CSV leaves the cell blank when none did. Show the
            # converged count (n=k/N) and n/a when nothing converged, instead
            # of crashing on float("").
            ged_cell = (r.get("ged") or "").strip()
            ged_n = (r.get("ged_n") or "").strip()
            n_imgs = (r.get("n_images") or "").strip()
            n_note = f" (n={ged_n}/{n_imgs})" if ged_n and n_imgs else ""
            try:
                n_conv = int(float(ged_n)) if ged_n else 0
            except ValueError:
                n_conv = 0
            # A single converged plan is not a meaningful mean; report n/a for
            # n<2, matching the manuscript's Table 7 convention.
            if ged_cell == "" or ged_cell.lower() in ("nan", "na") or n_conv < 2:
                ged_str = "n/a" + n_note
            else:
                ged_str = f"{float(ged_cell):.2f}" + n_note
            row_data.append([
                r["variant"],
                f"{float(r['mIoU']):.4f}",
                f"{float(r['edge_f1']):.4f}",
                ged_str,
                f"{float(r['frobenius']):.4f}",
            ])

    _render_table(col_labels, row_data,
                  "Table 5: Ablation Study Results", save_path)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def _latest_eval_file(root: Path, filename: str):
    """Most recently written results/eval_*/<filename>, or None. eval dirs are
    named eval_YYYYMMDD_HHMMSS, so a reverse lexicographic sort is chronological
    (same convention fig_failures.py uses)."""
    cands = sorted((root / "results").glob(f"eval_*/{filename}"), reverse=True)
    return cands[0] if cands else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=str(_SCRIPT_DIR.parent))
    parser.add_argument("--out", type=str, default=str(_SCRIPT_DIR / "tables"))
    args = parser.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Table 1: no CSV needed
    table1_dataset(out / "table1_dataset.pdf")

    # Table 2: class_iou.csv (root copy, else the latest eval run)
    class_iou_csv = root / "class_iou.csv"
    if not class_iou_csv.exists():
        class_iou_csv = _latest_eval_file(root, "class_iou.csv")
    if class_iou_csv and class_iou_csv.exists():
        table2_class_iou(out / "table2_class_iou.pdf", class_iou_csv)
    else:
        print(f"[skip] Table 2 - class_iou.csv not found")

    # Table 3: backbone comparison (reads both history_*.json if available)
    segformer_json = root / "results" / "history_segformer.json"
    if not segformer_json.exists():
        segformer_json = root / "history_segformer.json"
    deeplab_json = root / "results" / "history_deeplab.json"
    if not deeplab_json.exists():
        deeplab_json = root / "history_deeplab.json"
    table3_backbone(out / "table3_backbone.pdf",
                    segformer_json=segformer_json if segformer_json.exists() else None,
                    deeplab_json=deeplab_json if deeplab_json.exists() else None)

    # Table 4: summary.csv + per_image.csv (root copies, else the latest eval run)
    summary_csv = root / "summary.csv"
    per_image_csv = root / "per_image.csv"
    if not summary_csv.exists():
        summary_csv = _latest_eval_file(root, "summary.csv")
    if not per_image_csv.exists():
        per_image_csv = _latest_eval_file(root, "per_image.csv")
    if summary_csv and per_image_csv and summary_csv.exists() and per_image_csv.exists():
        table4_graph(out / "table4_graph.pdf", summary_csv, per_image_csv)
    else:
        print(f"[skip] Table 4 - summary.csv or per_image.csv not found")

    # Table 5: ablation_results.csv (the FINAL, corrected-ground-truth run)
    ablation_csv = root / "results" / "ablation_FINAL" / "ablation_results.csv"
    if ablation_csv.exists():
        table5_ablation(out / "table5_ablation.pdf", ablation_csv)
    else:
        print(f"[skip] Table 5 - ablation_results.csv not found")

    print(f"\nAll tables saved to: {out}")


if __name__ == "__main__":
    main()
