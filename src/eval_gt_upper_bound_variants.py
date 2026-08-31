"""
eval_gt_upper_bound_variants.py - corrected upper-bound + threshold sweep
==========================================================================
Runs M2 on ground-truth masks (no segmentation, no GPU needed) and scores
against ResPlan's own typed graph (build_gt_graph_from_resplan), under both
edge-matching conventions and a sweep of arch_min values. This replaces, with
the corrected ground truth, the numbers behind:
    - Table 5, "Upper bound" and "Upper bound, id-independent edge matching"
    - Table 7 / Table D.5, the theta_arch recalibration sweep
    - Table D.4, the edge-type confusion matrix (id-independent, arch_min=30)

Usage:
    python src/eval_gt_upper_bound_variants.py --root "<root>" [--limit N]
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
from eval_pipeline_variants import classreset_edges, idfree_edges, score

ARCH_MINS = [30, 14, 10, 8, 7, 6, 5, 4]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(_SCRIPT_DIR.parent))
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(args.root)
    mask_dir = root / "data" / "resplan_masks"
    out = Path(args.out) if args.out else root / "results" / f"gt_upper_bound_variants_{args.split}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    stems = [l.strip() for l in (root / "data" / "splits" / f"{args.split}.txt").read_text().splitlines() if l.strip()]
    if args.limit:
        stems = stems[:args.limit]

    print("[gt] loading ResPlan.pkl ...")
    records = load_resplan_records(root / "data" / "resplan_raw" / "ResPlan.pkl")

    rows = []
    confusion_30 = Counter()               # id-independent, arch_min=30 -> Table D.4
    confusion_by_am = {am: Counter() for am in ARCH_MINS}   # id-independent, all thresholds -> Table 6/7

    for stem in tqdm(stems, desc=f"{args.split} plans"):
        plan_id = int(stem)
        if plan_id not in records:
            continue
        mask_path = mask_dir / f"{stem}_mask.png"
        gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if gt_mask is None:
            continue
        gt_mask = gt_mask.astype(np.int64)

        try:
            G_gt = build_gt_graph_from_resplan(records[plan_id])
        except Exception:
            continue
        if G_gt.number_of_nodes() == 0:
            continue

        rec = {"stem": stem}
        for am in ARCH_MINS:
            try:
                G_pred = build_graph_from_segmentation(gt_mask, arch_min=am)
            except Exception:
                continue
            tag = f"a{am:02d}"

            p, r, f1, ta, _ = score(classreset_edges(G_pred), classreset_edges(G_gt))
            rec[f"classreset_{tag}_prec"], rec[f"classreset_{tag}_rec"], rec[f"classreset_{tag}_f1"], rec[f"classreset_{tag}_type"] = p, r, f1, ta

            pe, ge = idfree_edges(G_pred, G_gt)
            p, r, f1, ta, ntp = score(pe, ge)
            rec[f"free_{tag}_prec"], rec[f"free_{tag}_rec"], rec[f"free_{tag}_f1"], rec[f"free_{tag}_type"] = p, r, f1, ta

            for key, gt_et in ge.items():
                if key in pe:
                    confusion_by_am[am][(gt_et, pe[key])] += 1
                    if am == 30:
                        confusion_30[(gt_et, pe[key])] += 1

        rows.append(rec)

    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)

    print(f"\n{'='*90}")
    print(f"  UPPER-BOUND VARIANTS (M2 on GT masks vs ResPlan's own graph), {args.split}, n={len(df)}")
    print(f"{'='*90}")
    print(f"{'matching':<18}{'arch_min':>9}{'Edge P':>9}{'Edge R':>9}{'Edge F1':>9}{'type acc':>10}")
    print("-" * 90)
    for mtag, mname in (("classreset", "class-reset id (non-canonical)"), ("free", "id-independent")):
        for am in ARCH_MINS:
            tag = f"a{am:02d}"
            cols = [f"{mtag}_{tag}_prec", f"{mtag}_{tag}_rec", f"{mtag}_{tag}_f1", f"{mtag}_{tag}_type"]
            if cols[0] not in df.columns:
                continue
            print(f"{mname:<18}{am:>9}"
                  f"{df[cols[0]].mean():>9.4f}{df[cols[1]].mean():>9.4f}"
                  f"{df[cols[2]].mean():>9.4f}{df[cols[3]].mean():>10.4f}")

    print(f"\n  Table D.4 confusion matrix (id-independent, arch_min=30):")
    gt_types = sorted({k[0] for k in confusion_30})
    pred_types = sorted({k[1] for k in confusion_30})
    header = "    {:<14s}".format("GT \\ pred") + "".join(f"{t:>12s}" for t in pred_types) + f"{'support':>10s}{'recall':>9s}"
    print(header)
    for gt_et in gt_types:
        row_counts = [confusion_30[(gt_et, p)] for p in pred_types]
        support = sum(row_counts)
        correct = confusion_30[(gt_et, gt_et)] if gt_et in pred_types else 0
        recall = correct / support if support else float("nan")
        line = "    {:<14s}".format(gt_et) + "".join(f"{c:>12d}" for c in row_counts) + f"{support:>10d}{recall:>9.4f}"
        print(line)

    conf_rows = [{"gt_type": g, "pred_type": p, "count": c} for (g, p), c in confusion_30.items()]
    conf_path = out.with_name(out.stem + "_confusion_a30.csv")
    pd.DataFrame(conf_rows).to_csv(conf_path, index=False)

    print(f"\n  Threshold sweep, per-type recall (id-independent matching), Table 6/7:")
    sweep_rows = []
    print(f"    {'arch_min':>9}{'door_rec':>10}{'arch_rec':>10}{'wall_rec':>10}{'macro_rec':>11}{'pooled_acc':>12}")
    for am in ARCH_MINS:
        conf = confusion_by_am[am]
        gt_types = sorted({k[0] for k in conf})
        recalls = {}
        for gt_et in gt_types:
            support = sum(v for k, v in conf.items() if k[0] == gt_et)
            correct = conf.get((gt_et, gt_et), 0)
            recalls[gt_et] = correct / support if support else float("nan")
        door_r = recalls.get("door", float("nan"))
        arch_r = recalls.get("arch", float("nan"))
        wall_r = recalls.get("shared-wall", float("nan"))
        macro_r = np.nanmean([door_r, arch_r, wall_r])
        total = sum(conf.values())
        correct_total = sum(v for k, v in conf.items() if k[0] == k[1])
        pooled = correct_total / total if total else float("nan")
        print(f"    {am:>9}{door_r:>10.4f}{arch_r:>10.4f}{wall_r:>10.4f}{macro_r:>11.4f}{pooled:>12.4f}")
        sweep_rows.append({"arch_min": am, "door_recall": door_r, "arch_recall": arch_r,
                            "shared_wall_recall": wall_r, "macro_recall": macro_r, "pooled_accuracy": pooled})
    sweep_path = out.with_name(out.stem + "_threshold_sweep.csv")
    pd.DataFrame(sweep_rows).to_csv(sweep_path, index=False)

    print(f"\nSaved -> {out}\nSaved -> {conf_path}\nSaved -> {sweep_path}")


if __name__ == "__main__":
    main()
