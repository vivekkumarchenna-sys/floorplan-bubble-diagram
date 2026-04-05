"""
run_ablation.py — Automated ablation study (6 variants)
========================================================
Runs inference for each ablation variant by patching the graph extraction
behaviour, collects metrics, and generates the ablation bar chart (Fig 4).

Variants:
    1. Full pipeline       — baseline, all components active
    2. w/o door edges      — doors ignored in edge classification
    3. w/o room types      — all rooms labelled as generic "Room"
    4. w/o corridor adj.   — no second-order corridor edges
    5. w/o post-processing — dilation_px=0, min thresholds=0
    6. w/o edge typing     — all edges labelled "adjacent" (binary)

Usage:
    !python run_ablation.py
    !python run_ablation.py --ckpt path/to/model.pth --limit 500
    !python run_ablation.py --limit 0    # all test images (0 = no limit)
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
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

from build_graph import (
    CLASS_NAMES, ROOM_CLASSES, DOOR_CLASSES,
    build_graph_from_segmentation,
)
from build_gt_graph import mask_to_plan_dict, build_gt_graph_from_polygons
from proximity import compute_proximity_matrix
from evaluate import compute_miou, edge_metrics, graph_edit_distance, frobenius_norm
from inference import load_model, predict_mask, IMG_SIZE, NUM_CLASSES
from fig_ablation import plot_ablation

import networkx as nx


# ══════════════════════════════════════════════════════════════════════════════
# Variant graph builders
# ══════════════════════════════════════════════════════════════════════════════

def _build_graph_full(pred_mask):
    """Variant 1: Full pipeline (baseline)."""
    return build_graph_from_segmentation(pred_mask)


def _build_graph_no_doors(pred_mask):
    """Variant 2: w/o door edges — doors ignored in classification."""
    G = build_graph_from_segmentation(pred_mask, door_classes=set())
    return G


def _build_graph_no_room_types(pred_mask):
    """Variant 3: w/o room types — all nodes get generic 'Room' label."""
    G = build_graph_from_segmentation(pred_mask)
    for nid in G.nodes():
        G.nodes[nid]["class_name"] = "Room"
    return G


def _build_graph_no_corridor(pred_mask):
    """Variant 4: w/o corridor adj. — remove edges involving corridor rooms."""
    G = build_graph_from_segmentation(pred_mask)
    corridor_ids = [6]  # Corridor/Storage class
    edges_to_remove = []
    for u, v in G.edges():
        if G.nodes[u]["class_id"] in corridor_ids or G.nodes[v]["class_id"] in corridor_ids:
            edges_to_remove.append((u, v))
    G.remove_edges_from(edges_to_remove)
    return G


def _build_graph_no_postprocess(pred_mask):
    """Variant 5: w/o post-processing — no dilation, minimal thresholds."""
    return build_graph_from_segmentation(
        pred_mask,
        dilation_px=0,
        door_min=1,
        wall_min=1,
        arch_min=1,
        min_room_area=10,
    )


def _build_graph_no_edge_typing(pred_mask):
    """Variant 6: w/o edge typing — all edges become 'adjacent'."""
    G = build_graph_from_segmentation(pred_mask)
    for u, v in G.edges():
        G.edges[u, v]["edge_type"] = "adjacent"
    return G


VARIANTS = [
    ("Full pipeline",       _build_graph_full),
    ("w/o door edges",      _build_graph_no_doors),
    ("w/o room types",      _build_graph_no_room_types),
    ("w/o corridor adj.",   _build_graph_no_corridor),
    ("w/o post-processing", _build_graph_no_postprocess),
    ("w/o edge typing",     _build_graph_no_edge_typing),
]


# ══════════════════════════════════════════════════════════════════════════════
# Evaluate a single variant across all test images
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_variant(
    variant_name: str,
    graph_builder,
    stems: list[str],
    model,
    device,
    img_dir: Path,
    mask_dir: Path,
    ged_timeout: float,
) -> dict:
    """Run one ablation variant and return aggregated metrics."""
    mious, edge_f1s, geds, frobs = [], [], [], []

    for i, stem in enumerate(stems):
        img_path  = img_dir / f"{stem}.png"
        mask_path = mask_dir / f"{stem}_mask.png"

        if not img_path.exists() or not mask_path.exists():
            continue

        image_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        gt_mask   = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE).astype(np.int64)
        pred_mask = predict_mask(model, image_bgr, device, IMG_SIZE)

        if gt_mask.shape != pred_mask.shape:
            gt_mask = cv2.resize(gt_mask.astype(np.uint8), pred_mask.shape[::-1],
                                 interpolation=cv2.INTER_NEAREST).astype(np.int64)

        # pixel metrics
        df_iou = compute_miou(pred_mask, gt_mask, num_classes=NUM_CLASSES)
        miou = df_iou[df_iou.class_name == "mean"]["iou"].iloc[0]
        mious.append(miou)

        # build graphs
        try:
            G_pred = graph_builder(pred_mask)
        except Exception:
            G_pred = nx.Graph()

        try:
            G_gt = build_gt_graph_from_polygons(mask_to_plan_dict(gt_mask))
        except Exception:
            G_gt = nx.Graph()

        # edge metrics
        df_e = edge_metrics(G_pred, G_gt)
        edge_f1s.append(df_e["edge_f1"].iloc[0])

        # GED
        df_ged = graph_edit_distance(G_pred, G_gt, timeout=ged_timeout)
        geds.append(df_ged["ged"].iloc[0])

        # frobenius
        A_pred, _ = compute_proximity_matrix(G_pred) if G_pred.number_of_nodes() > 0 else (np.zeros((0, 0)), [])
        A_gt, _   = compute_proximity_matrix(G_gt)   if G_gt.number_of_nodes() > 0   else (np.zeros((0, 0)), [])
        if A_pred.size == 0 and A_gt.size == 0:
            frobs.append(0.0)
        else:
            df_f = frobenius_norm(A_pred, A_gt)
            frobs.append(df_f["normalized"].iloc[0])

        if (i + 1) % 100 == 0:
            print(f"    [{i+1}/{len(stems)}]")

    return {
        "variant":     variant_name,
        "mIoU":        round(float(np.mean(mious)), 4)    if mious    else 0.0,
        "edge_f1":     round(float(np.mean(edge_f1s)), 4) if edge_f1s else 0.0,
        "ged":         round(float(np.mean(geds)), 2)     if geds     else 0.0,
        "frobenius":   round(float(np.mean(frobs)), 4)    if frobs    else 0.0,
        "n_images":    len(mious),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Run ablation study (6 variants)")
    parser.add_argument("--root", type=str, default=str(_SCRIPT_DIR))
    parser.add_argument("--ckpt", type=str,
                        default=str(_SCRIPT_DIR / "checkpoints" / "segformer" / "best_model.pth"))
    parser.add_argument("--limit", type=int, default=300,
                        help="Max test images per variant (0 = all). Default 300 for speed.")
    parser.add_argument("--ged-timeout", type=float, default=5.0,
                        help="GED timeout per image in seconds")
    parser.add_argument("--out", type=str, default=str(_SCRIPT_DIR / "results" / "ablation"))
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # load test stems
    split_file = root / "data" / "splits" / "test.txt"
    stems = [l.strip() for l in split_file.read_text().splitlines() if l.strip()]
    if args.limit > 0:
        stems = stems[:args.limit]
    print(f"[data] Test images per variant: {len(stems)}")

    img_dir  = root / "data" / "resplan_raster"
    mask_dir = root / "data" / "resplan_masks"

    # load model once
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.ckpt, device)

    # run all variants
    results = []
    t_total = time.time()

    for vi, (name, builder) in enumerate(VARIANTS):
        print(f"\n{'='*60}")
        print(f"  Variant {vi+1}/6: {name}")
        print(f"{'='*60}")
        t0 = time.time()

        row = evaluate_variant(
            name, builder, stems, model, device,
            img_dir, mask_dir, args.ged_timeout,
        )
        elapsed = time.time() - t0
        row["time_s"] = round(elapsed, 1)
        results.append(row)

        print(f"  → mIoU={row['mIoU']:.4f}  edge_f1={row['edge_f1']:.4f}  "
              f"ged={row['ged']:.2f}  frob={row['frobenius']:.4f}  ({elapsed:.0f}s)")

        # save after each variant (crash-safe)
        df = pd.DataFrame(results)
        df.to_csv(out_dir / "ablation_results.csv", index=False)

    # final summary
    df = pd.DataFrame(results)
    df.to_csv(out_dir / "ablation_results.csv", index=False)

    total_time = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"  ABLATION COMPLETE  ({total_time:.0f}s)")
    print(f"{'='*60}")
    print(df.to_string(index=False))

    # generate the bar chart
    variant_labels = [
        "Full\npipeline",
        "w/o door\nedges",
        "w/o room\ntypes",
        "w/o corridor\nadj.",
        "w/o post-\nprocessing",
        "w/o edge\ntyping",
    ]
    plot_ablation(
        variants=variant_labels,
        edge_f1=df["edge_f1"].tolist(),
        miou=df["mIoU"].tolist(),
        save_path=str(out_dir / "fig4_ablation.pdf"),
    )

    print(f"\nResults → {out_dir / 'ablation_results.csv'}")
    print(f"Figure  → {out_dir / 'fig4_ablation.pdf'}")


if __name__ == "__main__":
    main()
