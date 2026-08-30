"""
validate_weight_sensitivity.py - sensitivity analysis for the proximity weights
================================================================================
The proximity-matrix weights (door=1.0, arch=0.8, window=0.5, shared-wall=0.3)
are a disclosed, hand-set design choice, not something claimed to be fit to
data. A reviewer can reasonably ask whether the paper's Frobenius-distance
findings are an artefact of that specific choice. This script re-derives the
normalised Frobenius distance (GT-mask-upper-bound condition, M2 on GT masks
vs ResPlan's own graph) under many resampled weight vectors that respect the
only ordering constraint the paper actually asserts (door is the strongest
connection, shared-wall the weakest: door > arch, window > shared-wall) and
reports how much the headline statistic moves, to show whether the reported
number is fragile to this choice or not.
"""
from __future__ import annotations

import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from build_graph import build_graph_from_segmentation
from build_gt_graph import build_gt_graph_from_resplan, load_resplan_records
from proximity import compute_proximity_matrix
from evaluate import frobenius_norm

N_PLANS = 400
N_TRIALS = 200
SEED = 20260727  # fixed, arbitrary - reproducibility, not randomness-as-a-feature


def main():
    root = _SCRIPT_DIR.parent
    records = load_resplan_records(root / "data" / "resplan_raw" / "ResPlan.pkl")
    test_ids = [int(x.strip()) for x in (root / "data" / "splits" / "test.txt").read_text().splitlines() if x.strip()]
    sample = test_ids[:N_PLANS]
    mask_dir = root / "data" / "resplan_masks"

    # pre-build graphs once (expensive part), then just re-score with
    # different weight dicts many times (cheap: a matrix diff per trial)
    pairs = []
    for pid in tqdm(sample, desc="building graphs"):
        mask_path = mask_dir / f"{pid}_mask.png"
        if not mask_path.exists() or pid not in records:
            continue
        gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if gt_mask is None:
            continue
        gt_mask = gt_mask.astype(np.int64)
        try:
            G_gt = build_gt_graph_from_resplan(records[pid])
            G_pred = build_graph_from_segmentation(gt_mask)
        except Exception:
            continue
        pairs.append((G_gt, G_pred))
    print(f"n plans usable: {len(pairs)}")

    rng = np.random.default_rng(SEED)

    def score(weights):
        vals = []
        for G_gt, G_pred in pairs:
            A_gt, _ = compute_proximity_matrix(G_gt, weights=weights)
            A_pred, _ = compute_proximity_matrix(G_pred, weights=weights)
            f = frobenius_norm(A_pred, A_gt)
            vals.append(f.iloc[0]["normalized"])
        return float(np.mean(vals))

    default_weights = {"door": 1.0, "arch": 0.8, "window": 0.5, "shared-wall": 0.3}
    default_score = score(default_weights)
    print(f"\ndefault weights {default_weights} -> mean normalised Frobenius = {default_score:.4f}")

    trial_scores = []
    for _ in tqdm(range(N_TRIALS), desc="weight trials"):
        # sample respecting: door strongest, shared-wall weakest, arch/window in between (either order)
        door = rng.uniform(0.85, 1.0)
        shared = rng.uniform(0.05, 0.35)
        arch = rng.uniform(shared + 0.05, door - 0.05)
        window = rng.uniform(shared + 0.05, door - 0.05)
        w = {"door": door, "arch": arch, "window": window, "shared-wall": shared}
        trial_scores.append(score(w))

    trial_scores = np.array(trial_scores)
    print(f"\n{N_TRIALS} random weight vectors (door>arch/window>shared-wall respected):")
    print(f"  mean   = {trial_scores.mean():.4f}")
    print(f"  std    = {trial_scores.std():.4f}")
    print(f"  min    = {trial_scores.min():.4f}")
    print(f"  max    = {trial_scores.max():.4f}")
    print(f"  range as %% of default value: {(trial_scores.max()-trial_scores.min())/default_score*100:.1f}%%")

    # also: extreme weight settings (max plausible spread)
    extremes = {
        "all equal (1,1,1,1)":        {"door": 1.0, "arch": 1.0, "window": 1.0, "shared-wall": 1.0},
        "door-only (1,0,0,0)":        {"door": 1.0, "arch": 0.0, "window": 0.0, "shared-wall": 0.0},
        "linear (1.0,0.67,0.33,0.0)": {"door": 1.0, "arch": 0.67, "window": 0.33, "shared-wall": 0.0},
        "published default":          default_weights,
    }
    print("\nSpot checks at named extreme/alternative weight schemes:")
    for name, w in extremes.items():
        print(f"  {name:<30s} -> {score(w):.4f}")


if __name__ == "__main__":
    main()
