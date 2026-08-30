"""
recompute_per_class_presence.py - presence-conditioned per-class segmentation
metrics on the full test split (Table C.1's methodology): a class's
IoU/precision/recall are averaged only over the plans where it actually occurs
in the ground truth, not over every plan including ones where it's absent and
would contribute a misleading 0.
"""
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import cv2
    import numpy as np
    import pandas as pd
    import torch
    from tqdm import tqdm
    from build_graph import CLASS_NAMES
    from inference import load_model, predict_mask, NUM_CLASSES
    from evaluate import compute_miou

    root = Path(args.root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.ckpt, device)

    stems = [s.strip() for s in (root / "data" / "splits" / "test.txt").read_text().splitlines() if s.strip()]
    img_dir = root / "data" / "resplan_raster"
    mask_dir = root / "data" / "resplan_masks"

    # accumulate per-class confusion PER PLAN, so we can later average only
    # over plans where the class has ground-truth support
    per_class_rows = {c: [] for c in range(NUM_CLASSES)}
    gt_pixel_totals = {c: 0 for c in range(NUM_CLASSES)}

    for stem in tqdm(stems, desc="per-class presence"):
        img = cv2.imread(str(img_dir / f"{stem}.png"), cv2.IMREAD_COLOR)
        gt = cv2.imread(str(mask_dir / f"{stem}_mask.png"), cv2.IMREAD_GRAYSCALE)
        if img is None or gt is None:
            continue
        gt = gt.astype(np.int64)
        pred = predict_mask(model, img, device)
        if gt.shape != pred.shape:
            gt = cv2.resize(gt.astype(np.uint8), pred.shape[::-1], interpolation=cv2.INTER_NEAREST).astype(np.int64)
        pred = pred.astype(np.int64)

        # single vectorised confusion-matrix pass (bincount) instead of a
        # 16x per-class boolean-mask loop; class absent in this plan's
        # ground truth (support == 0) is excluded, not zero-padded
        plan_df = compute_miou(pred, gt, num_classes=NUM_CLASSES, class_names=CLASS_NAMES)
        for _, row in plan_df[plan_df["class_id"] >= 0].iterrows():
            c = int(row["class_id"])
            support = int(row["support"])
            if support == 0:
                continue
            per_class_rows[c].append({"iou": row["iou"], "precision": row["precision"], "recall": row["recall"]})
            gt_pixel_totals[c] += support

    rows = []
    for c in range(NUM_CLASSES):
        recs = per_class_rows[c]
        n_plans = len(recs)
        if n_plans == 0:
            rows.append({"class_id": c, "class_name": CLASS_NAMES.get(c, f"class_{c}"),
                         "n_plans": 0, "precision": float("nan"), "recall": float("nan"),
                         "iou": float("nan"), "gt_pixels": 0})
            continue
        df = pd.DataFrame(recs)
        rows.append({
            "class_id": c, "class_name": CLASS_NAMES.get(c, f"class_{c}"),
            "n_plans": n_plans,
            "precision": df["precision"].mean(),
            "recall": df["recall"].mean(),
            "iou": df["iou"].mean(),
            "gt_pixels": gt_pixel_totals[c],
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False)
    print(out_df.to_string(index=False))

    # macro average over classes that occur at all (exclude Pool/Column/Other = 0 support)
    occurring = out_df[out_df["n_plans"] > 0]
    print("\nMacro (occurring classes):")
    print(f"  n classes: {len(occurring)}")
    print(f"  precision: {occurring['precision'].mean():.4f}")
    print(f"  recall:    {occurring['recall'].mean():.4f}")
    print(f"  iou:       {occurring['iou'].mean():.4f}")
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
