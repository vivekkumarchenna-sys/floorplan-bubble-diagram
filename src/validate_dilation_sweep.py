"""
validate_dilation_sweep.py - does the wide 15px Rule-1 existence test also
manufacture false-positive edges (not just mistyped ones)?
================================================================================
For a sample of the test split (GT masks, no GPU needed), sweeps
`dilation_px` (Rule 1's adjacency-existence test) and reports edge
precision/recall/F1 at each value, plus - for false-positive edges
specifically - the true (undilated) minimum pixel distance between the two
room masks, to see whether FPs are "genuinely nearby" (small true gap) or
"bridged only by the wide dilation" (large true gap, e.g. two rooms either
side of an unmodelled circulation space).
"""
from __future__ import annotations

import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm

from build_graph import build_graph_from_segmentation, _extract_room_instances, ROOM_CLASSES
from build_gt_graph import build_gt_graph_from_resplan, load_resplan_records
from evaluate import edge_metrics

SAMPLE_N = 500


def true_min_distance(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Minimum pixel distance between two boolean masks (0 if touching/overlapping)."""
    if (mask_a & mask_b).any():
        return 0.0
    dt = distance_transform_edt(~mask_b)
    d = dt[mask_a]
    return float(d.min()) if d.size else float("inf")


def main():
    root = _SCRIPT_DIR.parent
    records = load_resplan_records(root / "data" / "resplan_raw" / "ResPlan.pkl")
    test_ids = [int(x.strip()) for x in (root / "data" / "splits" / "test.txt").read_text().splitlines() if x.strip()]
    sample = test_ids[:SAMPLE_N]
    mask_dir = root / "data" / "resplan_masks"

    dilation_values = [15, 12, 10, 8]
    metric_rows = {d: [] for d in dilation_values}
    fp_gap_samples = {d: [] for d in dilation_values}

    for pid in tqdm(sample, desc="plans"):
        mask_path = mask_dir / f"{pid}_mask.png"
        if not mask_path.exists() or pid not in records:
            continue
        gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if gt_mask is None:
            continue
        gt_mask = gt_mask.astype(np.int64)
        try:
            G_gt = build_gt_graph_from_resplan(records[pid])
        except Exception:
            continue

        rooms = None
        for d in dilation_values:
            try:
                G_pred = build_graph_from_segmentation(gt_mask, dilation_px=d)
            except Exception:
                continue
            m = edge_metrics(G_pred, G_gt).iloc[0]
            metric_rows[d].append(dict(m))

            # sample a couple of false-positive edges' true gap at the widest
            # setting only (15, the current default) to keep this fast
            if d == 15 and len(fp_gap_samples[d]) < 400:
                if rooms is None:
                    rooms = {r.instance_id: r.mask for r in _extract_room_instances(gt_mask, ROOM_CLASSES, min_area=100)}
                pred_edges = {}
                for u, v, dd in G_pred.edges(data=True):
                    un = G_pred.nodes[u]["class_name"]; vn = G_pred.nodes[v]["class_name"]
                    pred_edges[(u, v)] = None
                gt_labels = {}
                idx = {}
                for n, dd in G_gt.nodes(data=True):
                    c = dd["class_name"]; idx[c] = idx.get(c, 0)
                    gt_labels[n] = f"{c}[{idx[c]}]"
                    idx[c] += 1
                gt_key_set = {tuple(sorted((gt_labels[a], gt_labels[b]))) for a, b in G_gt.edges()}
                pidx = {}
                pred_labels = {}
                for n, dd in G_pred.nodes(data=True):
                    c = dd["class_name"]; pidx[c] = pidx.get(c, 0)
                    pred_labels[n] = f"{c}[{pidx[c]}]"
                    pidx[c] += 1
                for u, v in list(G_pred.edges())[:6]:
                    key = tuple(sorted((pred_labels[u], pred_labels[v])))
                    if key not in gt_key_set and u in rooms and v in rooms:
                        gap = true_min_distance(rooms[u], rooms[v])
                        fp_gap_samples[d].append(gap)

    print(f"\n{'dilation_px':>12}{'precision':>12}{'recall':>10}{'f1':>10}")
    for d in dilation_values:
        df = pd.DataFrame(metric_rows[d])
        print(f"{d:>12}{df['edge_precision'].mean():>12.4f}{df['edge_recall'].mean():>10.4f}{df['edge_f1'].mean():>10.4f}")

    print(f"\nFalse-positive edges at dilation_px=15: true (undilated) min-gap distribution (n={len(fp_gap_samples[15])})")
    gaps = np.array(fp_gap_samples[15])
    if len(gaps):
        for pct in [10, 25, 50, 75, 90]:
            print(f"  p{pct}: {np.percentile(gaps, pct):.1f}px")
        print(f"  fraction with true gap > 15px (i.e. NOT touching before dilation): {(gaps > 15).mean():.3f}")
        print(f"  fraction with true gap > 30px (i.e. would need > double the dilation to bridge): {(gaps > 30).mean():.3f}")


if __name__ == "__main__":
    main()
