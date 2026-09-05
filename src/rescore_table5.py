# -*- coding: utf-8 -*-
"""Table 5 rows 2 and 3 on PREDICTED masks, with precision and recall retained.

One SegFormer pass per test plan; the predicted mask is scored against the
geometry-derived reference (build_true_graph on the ground-truth mask) under
both graph-construction stages:

  row 2  dilation-based M2 (build_graph.build_graph_from_segmentation), matched
         to reference rooms by nearest centroid within 20 px (as batch_rescore.py)
  row 3  geometric M2 (truegraph_builder.build_true_graph), matched by same class
         + nearest centroid within 30 px (as step4_rescore_newM2.py)

Also re-runs row 2 on GROUND-TRUTH masks (the condition batch_rescore.py used)
so the two can be compared.  Writes results/table5_rescore.csv (per plan) and
results/table5_rescore.json (aggregates).
"""
import sys; sys.path.insert(0, "src")
import cv2, numpy as np, torch, time, json, os
from collections import Counter
import truegraph_builder as tg
from build_graph import build_graph_from_segmentation
from inference import load_model, predict_mask

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TYPES = ["door", "open", "shared-wall"]

def centroids(R):
    return {i: (np.where(cm)[1].mean(), np.where(cm)[0].mean()) for i, (c, cm) in R.items()}

def score(pred_edges, gt):
    inter = set(pred_edges) & set(gt)
    P = len(inter) / len(pred_edges) if pred_edges else 0.0
    Rc = len(inter) / len(gt)
    F1 = 2 * P * Rc / (P + Rc) if (P + Rc) else 0.0
    ta = (sum(1 for e in inter if pred_edges[e] == gt[e]) / len(inter)) if inter else None
    return P, Rc, F1, ta, inter

def row2_edges(mask, R, cents):
    """dilation-based M2 on `mask`, mapped to reference room ids (batch_rescore convention)."""
    Gp = build_graph_from_segmentation(mask)
    mp = {}
    for u in Gp.nodes():
        cu = Gp.nodes[u].get("centroid")
        if cu is None: continue
        ux, uy = cu[1], cu[0]
        best = min(R, key=lambda i: (cents[i][0]-ux)**2 + (cents[i][1]-uy)**2)
        if (cents[best][0]-ux)**2 + (cents[best][1]-uy)**2 < 400: mp[u] = best
    pred = {}
    for u, v, d in Gp.edges(data=True):
        if u in mp and v in mp and mp[u] != mp[v]:
            pred[frozenset({mp[u], mp[v]})] = d.get("edge_type")
    return pred

def row3_edges(mask, Rg, cg):
    """geometric M2 on `mask`, mapped to reference room ids (step4 convention)."""
    Rp, pe = tg.build_true_graph(mask)
    cp = centroids(Rp)
    mp = {}
    for i, (ci, cmi) in Rp.items():
        best, bd = None, 900
        for j, (cj, cmj) in Rg.items():
            if cj != ci: continue
            d = (cp[i][0]-cg[j][0])**2 + (cp[i][1]-cg[j][1])**2
            if d < bd: bd, best = d, j
        if best is not None: mp[i] = best
    pred = {}
    for (a, b), t in pe.items():
        if a in mp and b in mp and mp[a] != mp[b]:
            pred[frozenset({mp[a], mp[b]})] = t
    return pred

def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    stems = [l.strip() for l in open("data/splits/test.txt") if l.strip()]
    if limit: stems = stems[:limit]
    model = load_model("checkpoints/segformer/best_model.pth", DEV)
    os.makedirs("results", exist_ok=True)
    conds = {"row2_pred": {}, "row2_gtmask": {}, "row3_pred": {}}
    acc = {k: {"P": [], "R": [], "F1": [], "TA": [], "agg": Counter()} for k in conds}
    rows = []; t0 = time.time(); n = 0
    for k, stem in enumerate(stems):
        gm = cv2.imread(f"data/resplan_masks/{stem}_mask.png", 0)
        raster = cv2.imread(f"data/resplan_raster/{stem}.png")
        if gm is None or raster is None: continue
        gt_mask = gm.astype(np.int64)
        Rg, ge = tg.build_true_graph(gt_mask)
        if not ge: continue
        gt = {frozenset({a, b}): t for (a, b), t in ge.items()}
        cg = centroids(Rg)
        with torch.no_grad():
            pred = predict_mask(model, raster, DEV)
        preds = {"row2_pred": row2_edges(pred, Rg, cg),
                 "row2_gtmask": row2_edges(gt_mask, Rg, cg),
                 "row3_pred": row3_edges(pred, Rg, cg)}
        rec = {"stem": stem, "n_gt_edges": len(gt)}
        for c, pe in preds.items():
            P, Rc, F1, ta, inter = score(pe, gt)
            a = acc[c]; a["P"].append(P); a["R"].append(Rc); a["F1"].append(F1)
            if ta is not None: a["TA"].append(ta)
            for e, t in gt.items():
                a["agg"]["gt_" + t] += 1
                if pe.get(e) == t: a["agg"]["rec_" + t] += 1
            for e in inter:
                a["agg"]["mtot"] += 1
                if pe[e] == gt[e]: a["agg"]["mcorrect"] += 1
            a["agg"]["n_pred"] += len(pe); a["agg"]["n_inter"] += len(inter); a["agg"]["n_gt"] += len(gt)
            rec.update({f"{c}_P": P, f"{c}_R": Rc, f"{c}_F1": F1, f"{c}_TA": ta, f"{c}_npred": len(pe)})
        rows.append(rec); n += 1
        if (k + 1) % 250 == 0:
            print(f"  {k+1}/{len(stems)}  row2 F1={np.mean(acc['row2_pred']['F1']):.4f}  row3 F1={np.mean(acc['row3_pred']['F1']):.4f}  ({time.time()-t0:.0f}s)", flush=True)
    import pandas as pd
    pd.DataFrame(rows).to_csv("results/table5_rescore.csv", index=False)
    out = {"n_plans": n}
    for c, a in acc.items():
        g = a["agg"]
        out[c] = {
            "edge_precision_mean": float(np.mean(a["P"])), "edge_precision_sd": float(np.std(a["P"])),
            "edge_recall_mean": float(np.mean(a["R"])), "edge_recall_sd": float(np.std(a["R"])),
            "edge_f1_mean": float(np.mean(a["F1"])), "edge_f1_sd": float(np.std(a["F1"])),
            "type_acc_mean": float(np.mean(a["TA"])), "type_acc_sd": float(np.std(a["TA"])),
            "pooled_precision": g["n_inter"] / max(1, g["n_pred"]),
            "pooled_recall": g["n_inter"] / max(1, g["n_gt"]),
            "pooled_type_acc": g["mcorrect"] / max(1, g["mtot"]),
            "per_type_recall": {t: (g["rec_" + t] / g["gt_" + t] if g["gt_" + t] else None) for t in TYPES},
            "per_type_support": {t: g["gt_" + t] for t in TYPES},
            "unmatched_share": 1 - g["n_inter"] / max(1, g["n_gt"]),
        }
    json.dump(out, open("results/table5_rescore.json", "w"), indent=2)
    print(json.dumps(out, indent=2)); print(f"time {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
