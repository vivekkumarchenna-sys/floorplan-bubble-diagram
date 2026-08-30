"""
validate_rule2_fix.py - before/after check for the door_dilation_px Rule-2 fix
================================================================================
Runs M2 on ground-truth masks (no GPU needed) for the full test split, twice:
once with door_dilation_px=15 (== old, single-dilation behavior) and once with
the new default (12), scored against the corrected ResPlan ground truth
(build_gt_graph_from_resplan). Reports pooled edge metrics, a GT-type ->
pred-type confusion matrix for both runs, and the delta, so the fix can be
judged on real data rather than a single anecdote before it is adopted as the
new default for the manuscript's numbers.
"""
from __future__ import annotations

import sys
from pathlib import Path
from collections import Counter

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from build_graph import build_graph_from_segmentation
from build_gt_graph import build_gt_graph_from_resplan, load_resplan_records
from evaluate import edge_metrics


def confusion_and_metrics(root: Path, door_dilation_px: int, test_ids: list[int], records):
    mask_dir = root / "data" / "resplan_masks"
    rows = []
    confusion = Counter()
    for pid in tqdm(test_ids, desc=f"door_dilation_px={door_dilation_px}"):
        mask_path = mask_dir / f"{pid}_mask.png"
        if not mask_path.exists() or pid not in records:
            continue
        gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if gt_mask is None:
            continue
        gt_mask = gt_mask.astype(np.int64)
        try:
            G_gt = build_gt_graph_from_resplan(records[pid])
            G_pred = build_graph_from_segmentation(gt_mask, door_dilation_px=door_dilation_px)
        except Exception:
            continue
        m = edge_metrics(G_pred, G_gt).iloc[0].to_dict()
        m["stem"] = pid
        rows.append(m)

        # confusion over matched pairs (same convention as evaluate._extract_edges)
        def edges_by_label(G):
            out = {}
            for u, v, d in G.edges(data=True):
                un = G.nodes[u].get("class_name", str(u))
                vn = G.nodes[v].get("class_name", str(v))
                key = tuple(sorted((f"{un}[{u}]", f"{vn}[{v}]")))
                out[key] = d.get("edge_type")
            return out

        gt_e = edges_by_label(G_gt)
        pred_e = edges_by_label(G_pred)
        for k, gt_type in gt_e.items():
            if k in pred_e:
                confusion[(gt_type, pred_e[k])] += 1

    df = pd.DataFrame(rows)
    return df, confusion


def main():
    root = _SCRIPT_DIR.parent
    records = load_resplan_records(root / "data" / "resplan_raw" / "ResPlan.pkl")
    test_ids = [int(x.strip()) for x in (root / "data" / "splits" / "test.txt").read_text().splitlines() if x.strip()]
    print(f"test split: {len(test_ids)} plan ids")

    for label, radius in [("OLD (=15, old shared-dilation behavior)", 15), ("NEW (default=12)", 12)]:
        print(f"\n{'='*70}\n{label}\n{'='*70}")
        df, confusion = confusion_and_metrics(root, radius, test_ids, records)
        print(f"n={len(df)}")
        for c in ["edge_precision", "edge_recall", "edge_f1", "type_accuracy"]:
            print(f"  {c}: {df[c].mean():.4f}")
        print("  confusion (gt_type -> pred_type):")
        for (gt_t, pred_t), cnt in sorted(confusion.items()):
            print(f"    {gt_t:12s} -> {pred_t:12s} : {cnt}")
        # shared-wall recall specifically
        sw_total = sum(v for (g, p), v in confusion.items() if g == "shared-wall")
        sw_correct = confusion.get(("shared-wall", "shared-wall"), 0)
        door_total = sum(v for (g, p), v in confusion.items() if g == "door")
        door_correct = confusion.get(("door", "door"), 0)
        print(f"  shared-wall recall: {sw_correct}/{sw_total} = {sw_correct/sw_total:.4f}" if sw_total else "  no shared-wall support")
        print(f"  door recall: {door_correct}/{door_total} = {door_correct/door_total:.4f}" if door_total else "  no door support")
        df.to_csv(root / "results" / f"rule2_validation_r{radius}.csv", index=False)


if __name__ == "__main__":
    main()
