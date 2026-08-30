"""
recompute_class_balance.py - corrected Table D.4 / class-balance recomputation
================================================================================
Compares, across every plan with both a released mask and a ResPlan.pkl
record:
    1. ResPlan's own edge-type distribution (the real ground truth)
    2. M2's reconstruction (build_graph_from_segmentation) run on the
       *ground-truth* mask, i.e. the same "M2 on GT masks" condition as
       eval_gt_upper_bound.py, so segmentation error plays no part
    3. A confusion matrix GT-type -> M2-predicted-type for every room pair
       that both graphs agree is adjacent (this reproduces the Table D.4
       "ground-truth arch 5,165, of which 5,150 predicted as shared wall"
       style breakdown, now against the corrected ground truth)

Usage:
    python src/recompute_class_balance.py --root "<project root>" [--limit N]
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from build_graph import build_graph_from_segmentation
from build_gt_graph import build_gt_graph_from_resplan, load_resplan_records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(_SCRIPT_DIR.parent))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(args.root)
    mask_dir = root / "data" / "resplan_masks"
    out = Path(args.out) if args.out else root / "results" / "class_balance_corrected.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    print("[gt] loading ResPlan.pkl ...")
    records = load_resplan_records(root / "data" / "resplan_raw" / "ResPlan.pkl")
    print(f"[gt] {len(records)} plan records")

    ids = sorted(records.keys())
    if args.limit:
        ids = ids[:args.limit]

    gt_type_counts = Counter()
    pred_type_counts = Counter()
    confusion = Counter()          # (gt_type, pred_type) -> count, over matched room pairs
    gt_edge_total = 0
    pred_edge_total = 0
    n_scored = 0

    rows = []

    for plan_id in tqdm(ids, desc="plans"):
        mask_path = mask_dir / f"{plan_id}_mask.png"
        if not mask_path.exists():
            continue
        gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if gt_mask is None:
            continue
        gt_mask = gt_mask.astype(np.int64)

        try:
            G_gt = build_gt_graph_from_resplan(records[plan_id])
        except Exception:
            continue
        try:
            G_pred = build_graph_from_segmentation(gt_mask)
        except Exception:
            continue

        # room-pair -> type, keyed by "ClassName[canonical_id]" like evaluate.py
        def edges_by_label(G):
            idx, lab = {}, {}
            for n, d in G.nodes(data=True):
                c = d["class_name"]
                idx[c] = idx.get(c, 0)
                lab[n] = f"{c}[{idx[c]}]"
                idx[c] += 1
            return {tuple(sorted((lab[u], lab[v]))): d.get("edge_type")
                    for u, v, d in G.edges(data=True)}

        gt_edges = edges_by_label(G_gt)
        pred_edges = edges_by_label(G_pred)

        for et in gt_edges.values():
            gt_type_counts[et] += 1
        for et in pred_edges.values():
            pred_type_counts[et] += 1
        gt_edge_total += len(gt_edges)
        pred_edge_total += len(pred_edges)

        for key, gt_et in gt_edges.items():
            if key in pred_edges:
                confusion[(gt_et, pred_edges[key])] += 1

        n_scored += 1
        rows.append({
            "plan_id": plan_id,
            "n_gt_edges": len(gt_edges),
            "n_pred_edges": len(pred_edges),
        })

    pd.DataFrame(rows).to_csv(out, index=False)

    print(f"\n{'='*70}")
    print(f"  Plans scored: {n_scored}")
    print(f"  Total GT edges (ResPlan's own graph)  : {gt_edge_total}")
    print(f"  Total M2-reconstruction edges (on GT masks): {pred_edge_total}")
    print(f"{'='*70}")

    print("\n  ResPlan's own edge-type distribution (ground truth):")
    for et, c in gt_type_counts.most_common():
        print(f"    {et:<14s} {c:>7d}  ({100*c/gt_edge_total:.2f}%)")

    print("\n  M2 reconstruction's edge-type distribution (on GT masks):")
    for et, c in pred_type_counts.most_common():
        print(f"    {et:<14s} {c:>7d}  ({100*c/pred_edge_total:.2f}%)")

    print("\n  Confusion (GT type -> M2-predicted type), matched room pairs only:")
    gt_types_seen = sorted({k[0] for k in confusion} | set(gt_type_counts))
    pred_types_seen = sorted({k[1] for k in confusion} | set(pred_type_counts))
    header = "    {:<14s}".format("GT \\ pred") + "".join(f"{t:>14s}" for t in pred_types_seen) + f"{'total':>10s}"
    print(header)
    for gt_et in gt_types_seen:
        row_total = sum(confusion[(gt_et, p)] for p in pred_types_seen)
        line = "    {:<14s}".format(gt_et) + "".join(f"{confusion[(gt_et, p)]:>14d}" for p in pred_types_seen) + f"{row_total:>10d}"
        print(line)

    conf_path = out.with_name(out.stem + "_confusion.csv")
    conf_rows = [{"gt_type": g, "pred_type": p, "count": c} for (g, p), c in confusion.items()]
    pd.DataFrame(conf_rows).to_csv(conf_path, index=False)
    print(f"\nSaved -> {out}\nSaved -> {conf_path}")


if __name__ == "__main__":
    main()
