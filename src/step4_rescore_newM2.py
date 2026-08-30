"""STEP 4 - re-score the NEW M2 (build_true_graph) END-TO-END on PREDICTED masks
against the corrected GT (build_true_graph on GT masks).

Loads SegFormer once (GPU), predicts each test raster, builds the typed graph on
the prediction, matches rooms to the GT-mask graph by class+centroid, and scores
Edge P/R/F1, per-type recall (door/open/shared-wall) and type accuracy.
This isolates segmentation-induced error (seg mIoU ~0.997) and shows the method
works end-to-end.
"""
import sys; sys.path.insert(0, "src")
import cv2, numpy as np, torch, time
from collections import Counter
import truegraph_builder as tg
from inference import load_model, predict_mask

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def centroids(R):
    return {i: (np.where(cm)[1].mean(), np.where(cm)[0].mean()) for i, (c, cm) in R.items()}

def match_pred_to_gt(Rp, Rg, cp, cg, tol2=900):
    """pred room -> gt room by same class + nearest centroid within tol (30 px)."""
    mp = {}
    for i, (ci, cmi) in Rp.items():
        best, bd = None, tol2
        for j, (cj, cmj) in Rg.items():
            if cj != ci:
                continue
            d = (cp[i][0] - cg[j][0]) ** 2 + (cp[i][1] - cg[j][1]) ** 2
            if d < bd:
                bd, best = d, j
        if best is not None:
            mp[i] = best
    return mp

def main():
    stems = [l.strip() for l in open("data/splits/test.txt") if l.strip()]
    model = load_model("checkpoints/segformer/best_model.pth", DEV)
    t0 = time.time()
    f1s = []; tas = []
    agg = Counter()   # gt_<t>, correct_<t>, matched-type counters
    npl = 0; skipped = 0
    for k, stem in enumerate(stems):
        gm = cv2.imread(f"data/resplan_masks/{stem}_mask.png", 0)
        raster = cv2.imread(f"data/resplan_raster/{stem}.png")
        if gm is None or raster is None:
            skipped += 1; continue
        gt_mask = gm.astype(np.int64)
        pred = predict_mask(model, raster, DEV)
        Rg, ge = tg.build_true_graph(gt_mask)
        Rp, pe = tg.build_true_graph(pred)
        if not ge:
            continue
        cg = centroids(Rg); cp = centroids(Rp)
        mp = match_pred_to_gt(Rp, Rg, cp, cg)
        # translate predicted edges into GT room-id space
        pred_edges = {}
        for (a, b), t in pe.items():
            if a in mp and b in mp and mp[a] != mp[b]:
                pred_edges[frozenset({mp[a], mp[b]})] = t
        gt = {frozenset({a, b}): t for (a, b), t in ge.items()}
        inter = set(pred_edges) & set(gt)
        P = len(inter) / len(pred_edges) if pred_edges else 0.0
        Rc = len(inter) / len(gt)
        F1 = 2 * P * Rc / (P + Rc) if (P + Rc) else 0.0
        f1s.append(F1)
        if inter:
            tas.append(sum(1 for e in inter if pred_edges[e] == gt[e]) / len(inter))
        for e, t in gt.items():
            agg["gt_" + t] += 1
            if pred_edges.get(e) == t:
                agg["rec_" + t] += 1     # same pair AND same type
        for e in inter:
            agg["mtot"] += 1
            if pred_edges[e] == gt[e]:
                agg["mcorrect"] += 1
        npl += 1
        if (k + 1) % 500 == 0:
            print(f"  {k+1}/{len(stems)}  F1={np.mean(f1s):.4f}  ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n=== NEW M2 end-to-end (PREDICTED masks) vs corrected GT   n={npl}  (skipped {skipped}) ===")
    print(f"  Edge-existence F1 (mean/plan): {np.mean(f1s):.4f}")
    print(f"  Edge-type acc among matched  : {np.mean(tas):.4f}")
    for t in ["door", "open", "shared-wall"]:
        g = agg["gt_" + t]; r = agg["rec_" + t]
        print(f"  {t:12s} recall: {r}/{g} = {r/g:.4f}" if g else f"  {t:12s} recall: n/a")
    print(f"  Pooled type acc (matched): {agg['mcorrect']}/{agg['mtot']} = {agg['mcorrect']/max(1,agg['mtot']):.4f}")
    print(f"  time {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
