"""
eval_backbone.py - evaluate any trained SegFormer backbone on the test split.
=============================================================================
Reports the two numbers the paper compares backbones on:

  1. per-image mIoU - the mean, over test plans, of the macro IoU across the
     classes present in that plan (the definition used in Section 6.2);
  2. the end-to-end typed-graph scores - module M2 (truegraph_builder) run on
     the predicted masks and scored against the geometry-derived reference,
     exactly as src/step4_rescore_newM2.py does for SegFormer-B3.

Unlike the DeepLabV3+ sanity check in Appendix B.4, this runs on the *test*
split, so a lighter backbone can be compared with SegFormer-B3 on the same
terms as every other number in the paper.

usage:
    python src/eval_backbone.py --ckpt run/checkpoints/segformer/best_model.pth \
                                --pretrained nvidia/segformer-b0-finetuned-ade-512-512
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation

sys.path.insert(0, str(Path(__file__).resolve().parent))
import truegraph_builder as tg

NUM_CLASSES = 16
IMG_SIZE = 512
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_any(ckpt_path: str, pretrained: str, device: torch.device):
    id2label = {i: str(i) for i in range(NUM_CLASSES)}
    model = SegformerForSemanticSegmentation.from_pretrained(
        pretrained, num_labels=NUM_CLASSES, id2label=id2label,
        label2id={v: k for k, v in id2label.items()}, ignore_mismatched_sizes=True)
    # load onto CPU first: a checkpoint saved from CUDA cannot be mapped onto a
    # device that is visible but has zero count (CUDA_VISIBLE_DEVICES="")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"[model] {pretrained}  epoch={ckpt.get('epoch','?')}  "
          f"val_mIoU={ckpt.get('val_mIoU','?')}  params={n:,}")
    return model, n


@torch.no_grad()
def predict(model, image_bgr, device):
    img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    if (h, w) != (IMG_SIZE, IMG_SIZE):
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    x = (img.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(device)
    logits = model(pixel_values=x).logits
    logits = F.interpolate(logits, size=(IMG_SIZE, IMG_SIZE),
                           mode="bilinear", align_corners=False)
    pred = logits.argmax(1).squeeze(0).cpu().numpy()
    if (h, w) != (IMG_SIZE, IMG_SIZE):
        pred = cv2.resize(pred.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    return pred.astype(np.int64)


def per_image_miou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Macro IoU across the classes present in this plan's ground truth."""
    ious = []
    for c in np.unique(gt):
        p, g = pred == c, gt == c
        union = np.logical_or(p, g).sum()
        if union:
            ious.append(np.logical_and(p, g).sum() / union)
    return float(np.mean(ious)) if ious else float("nan")


def centroids(rooms):
    return {i: (np.where(cm)[1].mean(), np.where(cm)[0].mean()) for i, (c, cm) in rooms.items()}


def match(Rp, Rg, cp, cg, tol2=900):
    """Predicted room -> reference room: same class, nearest centroid within 30 px."""
    out = {}
    for i, (ci, _) in Rp.items():
        best, bd = None, tol2
        for j, (cj, _) in Rg.items():
            if cj != ci:
                continue
            d = (cp[i][0] - cg[j][0]) ** 2 + (cp[i][1] - cg[j][1]) ** 2
            if d < bd:
                bd, best = d, j
        if best is not None:
            out[i] = best
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pretrained", default="nvidia/segformer-b3-finetuned-ade-512-512")
    ap.add_argument("--data", default="data", help="folder holding resplan_raster/, resplan_masks/, splits/")
    ap.add_argument("--limit", type=int, default=0, help="evaluate only the first N test plans")
    a = ap.parse_args()

    data = Path(a.data)
    use_cuda = torch.cuda.is_available() and torch.cuda.device_count() > 0
    device = torch.device("cuda" if use_cuda else "cpu")
    model, n_params = load_any(a.ckpt, a.pretrained, device)

    stems = [l.strip() for l in open(data / "splits" / "test.txt") if l.strip()]
    if a.limit:
        stems = stems[:a.limit]

    mious, f1s, tacc = [], [], []
    agg = Counter()
    n_plans = skipped = 0
    t0 = time.time()

    for k, stem in enumerate(stems):
        gm = cv2.imread(str(data / "resplan_masks" / f"{stem}_mask.png"), 0)
        raster = cv2.imread(str(data / "resplan_raster" / f"{stem}.png"))
        if gm is None or raster is None:
            skipped += 1
            continue
        gt_mask = gm.astype(np.int64)
        pred = predict(model, raster, device)

        m = per_image_miou(pred, gt_mask)
        if not np.isnan(m):
            mious.append(m)

        Rg, ge = tg.build_true_graph(gt_mask)
        if not ge:
            continue
        Rp, pe = tg.build_true_graph(pred)
        mp = match(Rp, Rg, centroids(Rp), centroids(Rg))

        pred_edges = {}
        for (x, y), t in pe.items():
            if x in mp and y in mp and mp[x] != mp[y]:
                pred_edges[frozenset({mp[x], mp[y]})] = t
        ref = {frozenset({x, y}): t for (x, y), t in ge.items()}

        inter = set(pred_edges) & set(ref)
        P = len(inter) / len(pred_edges) if pred_edges else 0.0
        R = len(inter) / len(ref)
        f1s.append(2 * P * R / (P + R) if (P + R) else 0.0)
        if inter:
            tacc.append(sum(1 for e in inter if pred_edges[e] == ref[e]) / len(inter))
        for e, t in ref.items():
            agg["ref_" + t] += 1
            if pred_edges.get(e) == t:
                agg["rec_" + t] += 1
        for e in inter:
            agg["mtot"] += 1
            agg["mcorrect"] += int(pred_edges[e] == ref[e])
        n_plans += 1

        if (k + 1) % 500 == 0:
            print(f"  {k+1}/{len(stems)}  mIoU={np.mean(mious):.4f}  "
                  f"EdgeF1={np.mean(f1s):.4f}  ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n=== {a.pretrained} on the test split  (n={n_plans}, skipped {skipped}) ===")
    print(f"  parameters                : {n_params:,}")
    print(f"  per-image mIoU            : {np.mean(mious):.4f} +/- {np.std(mious):.4f}")
    print(f"  Edge F1 (M2 vs reference) : {np.mean(f1s):.4f}")
    print(f"  edge-type acc (matched)   : {np.mean(tacc):.4f}")
    for t in ("door", "open", "shared-wall"):
        g, r = agg["ref_" + t], agg["rec_" + t]
        print(f"  {t:12s} recall        : {r}/{g} = {r/g:.4f}" if g else f"  {t:12s} recall        : n/a")
    print(f"  pooled type acc (matched) : {agg['mcorrect']}/{agg['mtot']} = "
          f"{agg['mcorrect']/max(1, agg['mtot']):.4f}")
    print(f"  wall-clock                : {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
