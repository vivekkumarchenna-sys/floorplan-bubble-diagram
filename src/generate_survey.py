"""
generate_survey.py - Generate paired stimuli for user survey
=============================================================
Picks 10 diverse floor plans from the test set and generates:
  - Floor plan image
  - Pipeline bubble diagram (from predicted segmentation)
  - GT bubble diagram (from GT mask)
  - Stimulus mapping CSV (randomized presentation order)

Usage:
    python generate_survey.py
    python generate_survey.py --n 10 --out survey_stimuli/
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from build_graph import build_graph_from_segmentation
from build_gt_graph import build_gt_graph_from_resplan, load_resplan_records
from visualize import draw_bubble_diagram
from inference import load_model, predict_mask, IMG_SIZE

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


def _load_pixel_scale(root):
    path = root / "pixel_scale.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def _pick_diverse_samples(csv_path, n=10, seed=42):
    """
    Pick n diverse samples spanning different edge_f1 and room counts.
    Strategy: sort by edge_f1, split into n bins, pick one per bin.
    """
    df = pd.read_csv(csv_path)
    df = df.sort_values("edge_f1").reset_index(drop=True)

    bin_size = len(df) // n
    picks = []
    for i in range(n):
        start = i * bin_size
        end = start + bin_size if i < n - 1 else len(df)
        chunk = df.iloc[start:end]
        # pick the middle sample from each bin
        mid = len(chunk) // 2
        picks.append(chunk.iloc[mid])

    return pd.DataFrame(picks)


def generate_survey(n, root, ckpt_path, csv_path, out_dir, seed=42):
    random.seed(seed)
    out_dir = Path(out_dir)
    (out_dir / "floorplans").mkdir(parents=True, exist_ok=True)
    (out_dir / "bubbles").mkdir(parents=True, exist_ok=True)

    img_dir = root / "data" / "resplan_raster"
    mask_dir = root / "data" / "resplan_masks"
    pixel_scale_map = _load_pixel_scale(root)
    resplan_records = load_resplan_records(root / "data" / "resplan_raw" / "ResPlan.pkl")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(ckpt_path, device)

    # pick diverse samples
    df_picks = _pick_diverse_samples(csv_path, n=n, seed=seed)

    stimuli = []
    for idx, (_, row) in enumerate(df_picks.iterrows()):
        stem = str(int(row["stem"]) if isinstance(row["stem"], float) else row["stem"])
        num = f"{idx + 1:02d}"

        print(f"[{num}/{n}] stem={stem}  edge_f1={row['edge_f1']:.4f}  "
              f"rooms={int(row['n_rooms_pred'])}")

        # load images
        image_bgr = cv2.imread(str(img_dir / f"{stem}.png"), cv2.IMREAD_COLOR)
        gt_mask = cv2.imread(str(mask_dir / f"{stem}_mask.png"),
                             cv2.IMREAD_GRAYSCALE).astype(np.int64)
        pred_mask = predict_mask(model, image_bgr, device, IMG_SIZE)

        if gt_mask.shape != pred_mask.shape:
            gt_mask = cv2.resize(gt_mask.astype(np.uint8), pred_mask.shape[::-1],
                                 interpolation=cv2.INTER_NEAREST).astype(np.int64)

        scale = pixel_scale_map.get(stem)

        # pipeline bubble diagram
        G_pred = build_graph_from_segmentation(pred_mask, pixel_scale=scale)
        fig_pred, ax_pred = plt.subplots(figsize=(8, 6))
        draw_bubble_diagram(G_pred, ax=ax_pred, title="")
        fig_pred.tight_layout()
        pred_path = out_dir / "bubbles" / f"bubble_pred_{num}.png"
        fig_pred.savefig(pred_path, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig_pred)

        # GT bubble diagram (ResPlan's own typed graph, not a reconstruction)
        G_gt = build_gt_graph_from_resplan(resplan_records[int(stem)])
        # Room-level "area" here is in ResPlan's own polygon coordinate units,
        # NOT the 512x512 raster pixel units pixel_scale.json was built from
        # (the two are not the same frame - do not multiply by `scale`, that
        # would silently mislabel these diagrams). Instead distribute the
        # plan's own reported net_area (m^2) across rooms in proportion to
        # their ResPlan polygon area, which needs no external scale factor.
        net_area = resplan_records[int(stem)].get("net_area")
        interior_area_raw = sum(
            d["area"] for _, d in G_gt.nodes(data=True)
            if d["class_id"] in {1, 2, 3, 4, 6, 7}  # Bedroom/Bathroom/Kitchen/Living/Storage/Stair
        )
        if net_area and interior_area_raw > 0:
            for nid in G_gt.nodes():
                G_gt.nodes[nid]["area_sqm"] = round(
                    G_gt.nodes[nid]["area"] / interior_area_raw * net_area, 2
                )
        fig_gt, ax_gt = plt.subplots(figsize=(8, 6))
        draw_bubble_diagram(G_gt, ax=ax_gt, title="")
        fig_gt.tight_layout()
        gt_path = out_dir / "bubbles" / f"bubble_gt_{num}.png"
        fig_gt.savefig(gt_path, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig_gt)

        # save floor plan image
        fp_path = out_dir / "floorplans" / f"floorplan_{num}.png"
        cv2.imwrite(str(fp_path), image_bgr)

        # add paired stimuli (pipeline and GT for same floor plan)
        stimuli.append({
            "Stimulus_ID": f"S{2 * idx + 1:02d}",
            "Source": "Pipeline",
            "Floor_Plan_File": f"floorplan_{num}.png",
            "Bubble_Diagram_File": f"bubble_pred_{num}.png",
            "stem": stem,
            "edge_f1": round(row["edge_f1"], 4),
            "n_rooms": int(row["n_rooms_pred"]),
        })
        stimuli.append({
            "Stimulus_ID": f"S{2 * idx + 2:02d}",
            "Source": "GT",
            "Floor_Plan_File": f"floorplan_{num}.png",
            "Bubble_Diagram_File": f"bubble_gt_{num}.png",
            "stem": stem,
            "edge_f1": 1.0,
            "n_rooms": int(row["n_rooms_gt"]),
        })

    # randomize presentation order
    random.shuffle(stimuli)
    # reassign stimulus IDs after shuffle
    for i, s in enumerate(stimuli):
        s["Stimulus_ID"] = f"S{i + 1:02d}"

    # save CSV
    csv_out = out_dir / "survey_stimuli.csv"
    with open(csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "Stimulus_ID", "Source", "Floor_Plan_File",
            "Bubble_Diagram_File", "stem", "edge_f1", "n_rooms",
        ])
        writer.writeheader()
        writer.writerows(stimuli)

    print(f"\nGenerated {len(stimuli)} stimuli from {n} floor plans")
    print(f"  Floor plans : {out_dir / 'floorplans'}")
    print(f"  Bubbles     : {out_dir / 'bubbles'}")
    print(f"  CSV mapping : {csv_out}")

    # print summary
    print(f"\nPresentation order (randomized):")
    for s in stimuli:
        print(f"  {s['Stimulus_ID']}  {s['Source']:<10s}  {s['Floor_Plan_File']}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate paired survey stimuli from test set")
    parser.add_argument("--n", type=int, default=10,
                        help="Number of floor plans to sample (default: 10)")
    parser.add_argument("--root", type=str, default=str(_SCRIPT_DIR))
    parser.add_argument("--ckpt", type=str,
                        default=str(_SCRIPT_DIR / "checkpoints" / "segformer" / "best_model.pth"))
    parser.add_argument("--csv", type=str,
                        default=str(_SCRIPT_DIR / "results" / "eval_20260403_155913" / "per_image.csv"))
    parser.add_argument("--out", type=str,
                        default=str(_SCRIPT_DIR / "survey_stimuli"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # fallback CSV paths
    csv_path = Path(args.csv)
    if not csv_path.exists():
        csv_path = Path(args.root) / "per_image.csv"

    generate_survey(
        n=args.n,
        root=Path(args.root),
        ckpt_path=args.ckpt,
        csv_path=csv_path,
        out_dir=args.out,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
