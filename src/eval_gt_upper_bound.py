"""
eval_gt_upper_bound.py - Upper-bound evaluation: M2-M4 on GT masks
====================================================================
Runs graph construction (M2), proximity matrix (M3), and evaluation
using ground-truth segmentation masks as input instead of predicted masks.

This isolates graph extraction error from segmentation error, giving
an upper-bound Edge F1 for modules M2-M4.

Usage:
    python eval_gt_upper_bound.py
    python eval_gt_upper_bound.py --limit 500
"""

from __future__ import annotations

import argparse
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
from tqdm import tqdm

from build_graph import build_graph_from_segmentation
from build_gt_graph import build_gt_graph_from_resplan, load_resplan_records
from proximity import compute_proximity_matrix
from evaluate import edge_metrics, graph_edit_distance, frobenius_norm


def main():
    parser = argparse.ArgumentParser(
        description="Upper-bound eval: run M2-M4 on GT masks")
    parser.add_argument("--root", type=str, default=str(_SCRIPT_DIR))
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max images (0 = all)")
    parser.add_argument("--ged-timeout", type=float, default=5.0,
                        help="Now a real hard cutoff (daemon-thread based, "
                             "see evaluate.graph_edit_distance) rather than a "
                             "cooperative one, so this is safe to raise.")
    parser.add_argument("--skip-ged", action="store_true",
                        help="Skip GED computation (faster)")
    parser.add_argument("--mask-dir", type=str, default=None,
                        help="Override mask directory (e.g. /tmp/resplan_masks)")
    parser.add_argument("--out", type=str, default=None,
                        help="Output CSV path")
    args = parser.parse_args()

    root = Path(args.root)
    mask_dir = Path(args.mask_dir) if args.mask_dir else root / "data" / "resplan_masks"
    split_file = root / "data" / "splits" / f"{args.split}.txt"

    stems = [l.strip() for l in split_file.read_text().splitlines() if l.strip()]
    if args.limit > 0:
        stems = stems[:args.limit]

    resplan_pkl = root / "data" / "resplan_raw" / "ResPlan.pkl"
    print(f"[gt] loading ResPlan's own graphs from {resplan_pkl}")
    resplan_records = load_resplan_records(resplan_pkl)

    print(f"[eval] Upper-bound evaluation: M2 (on GT masks) vs ResPlan's own graph")
    print(f"[eval] Split: {args.split}  Images: {len(stems)}")

    records = []

    for stem in tqdm(stems, desc="Evaluating GT masks"):
        mask_path = mask_dir / f"{stem}_mask.png"
        gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if gt_mask is None:
            continue
        gt_mask = gt_mask.astype(np.int64)

        # M2: build graph from GT mask using dilation-based method (same as pipeline)
        try:
            G_pred = build_graph_from_segmentation(gt_mask)
        except Exception:
            continue

        # Ground truth: ResPlan's own typed graph, not a reconstruction.
        plan_id = int(stem)
        if plan_id not in resplan_records:
            continue
        try:
            G_gt = build_gt_graph_from_resplan(resplan_records[plan_id])
        except Exception:
            continue

        # Edge metrics
        try:
            edge_df = edge_metrics(G_pred, G_gt)
            e = edge_df.iloc[0]
            ep, er, ef1 = e["edge_precision"], e["edge_recall"], e["edge_f1"]
            ta = e["type_accuracy"]
        except Exception:
            ep = er = ef1 = ta = 0.0

        # GED
        if args.skip_ged:
            ged = float("nan")
        else:
            try:
                ged_df = graph_edit_distance(G_pred, G_gt, timeout=args.ged_timeout)
                ged = ged_df.iloc[0]["ged"]
            except Exception:
                ged = float("nan")

        # Frobenius
        try:
            A_pred, _ = compute_proximity_matrix(G_pred) if G_pred.number_of_nodes() > 0 else (np.zeros((0, 0)), [])
            A_gt, _ = compute_proximity_matrix(G_gt) if G_gt.number_of_nodes() > 0 else (np.zeros((0, 0)), [])
            frob_df = frobenius_norm(A_pred, A_gt)
            frob = frob_df.iloc[0]["frobenius"]
            frob_norm = frob_df.iloc[0]["normalized"]
        except Exception:
            frob = frob_norm = float("nan")

        records.append({
            "stem": stem,
            "edge_precision": ep,
            "edge_recall": er,
            "edge_f1": ef1,
            "type_accuracy": ta,
            "ged": ged,
            "frobenius": frob,
            "frob_normalized": frob_norm,
            "n_rooms_pred": G_pred.number_of_nodes(),
            "n_rooms_gt": G_gt.number_of_nodes(),
            "n_edges_pred": G_pred.number_of_edges(),
            "n_edges_gt": G_gt.number_of_edges(),
        })

    df = pd.DataFrame(records)

    # Guard: if nothing was scored (e.g. masks or ResPlan.pkl missing for this
    # split), df has no columns and every df['...'].mean() below would raise a
    # KeyError. Report and bail out cleanly instead.
    if df.empty:
        print("\n[warn] No images were successfully evaluated (empty result "
              "set) - nothing to summarise. Check that the masks and "
              "ResPlan.pkl exist for this split.")
        out_path = args.out or str(root / "gt_upper_bound.csv")
        df.to_csv(out_path, index=False)
        print(f"Saved -> {out_path}")
        return

    # Summary
    print(f"\n{'='*60}")
    print(f"  UPPER-BOUND RESULTS (M2-M4 on GT masks)")
    print(f"  Images evaluated: {len(df)}")
    print(f"{'='*60}")
    print(f"  Edge Precision : {df['edge_precision'].mean():.4f}")
    print(f"  Edge Recall    : {df['edge_recall'].mean():.4f}")
    print(f"  Edge F1        : {df['edge_f1'].mean():.4f}")
    print(f"  Type Accuracy  : {df['type_accuracy'].mean():.4f}")
    print(f"  GED            : {df['ged'].mean():.2f} +/- {df['ged'].std():.2f}")
    print(f"  Frobenius (n)  : {df['frob_normalized'].mean():.4f}")
    print(f"{'='*60}")

    # Compare with pipeline results. Read the most recent full-pipeline
    # inference summary.csv dynamically rather than hardcoding a number here:
    # a hardcoded value has gone stale at least twice before (once as a
    # copy-paste of an old run, once as the literal 0.6589 pre-ground-truth-fix
    # published figure). Anything that silently reuses a magic number here is
    # exactly the kind of staleness this correction pass was about.
    pipeline_ef1 = None
    results_dir = root / "results"
    if results_dir.exists():
        candidates = sorted(results_dir.glob("eval_*/summary.csv"), reverse=True)
        for c in candidates:
            try:
                pipeline_ef1 = float(pd.read_csv(c)["edge_f1"].iloc[0])
                print(f"  (using full-pipeline Edge F1 {pipeline_ef1:.4f} from {c})")
                break
            except Exception:
                continue
    upper_ef1 = df['edge_f1'].mean()
    if pipeline_ef1 is None:
        print(f"\n  No results/eval_*/summary.csv found - skipping the")
        print(f"  pipeline-vs-upper-bound comparison below (run inference.py first).")
        df.to_csv(args.out or str(root / "gt_upper_bound.csv"), index=False)
        print(f"\nSaved -> {args.out or str(root / 'gt_upper_bound.csv')}")
        return
    m1_error_pct = (1 - pipeline_ef1) / (1 - 0) * 100  # total error from pipeline
    m2m4_error = 1 - upper_ef1  # error from M2-M4 alone
    m1_contribution = ((upper_ef1 - pipeline_ef1) / (1 - pipeline_ef1)) * 100 if upper_ef1 > pipeline_ef1 else 0

    print(f"\n  Pipeline Edge F1     : {pipeline_ef1:.4f}")
    print(f"  Upper-bound Edge F1  : {upper_ef1:.4f}")
    print(f"  Error from M1 (seg)  : {m1_contribution:.1f}%")
    print(f"  Error from M2-M4     : {100 - m1_contribution:.1f}%")
    print(f"{'='*60}")

    # Save
    out_path = args.out or str(root / "gt_upper_bound.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
