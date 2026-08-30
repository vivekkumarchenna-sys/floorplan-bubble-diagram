"""3-type confusion (door/open/shared-wall) for geometry M2 on PREDICTED masks
vs geometry-based GT (GT masks). Full test split. For Table D.4."""
import sys; sys.path.insert(0, "src")
import cv2, numpy as np, torch
from collections import Counter
import truegraph_builder as tg
from inference import load_model, predict_mask
from step4_rescore_newM2 import centroids, match_pred_to_gt
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TYPES = ["door", "open", "shared-wall"]
conf = Counter()  # (gt_type, pred_type) -> count; pred_type 'absent' if missing
model = load_model("checkpoints/segformer/best_model.pth", DEV)
stems = [l.strip() for l in open("data/splits/test.txt") if l.strip()]
for k, stem in enumerate(stems):
    gm = cv2.imread(f"data/resplan_masks/{stem}_mask.png", 0)
    raster = cv2.imread(f"data/resplan_raster/{stem}.png")
    if gm is None or raster is None: continue
    pred = predict_mask(model, raster, DEV)
    Rg, ge = tg.build_true_graph(gm.astype(np.int64)); Rp, pe = tg.build_true_graph(pred)
    if not ge: continue
    cg = centroids(Rg); cp = centroids(Rp); mp = match_pred_to_gt(Rp, Rg, cp, cg)
    pmap = {}
    for (a, b), t in pe.items():
        if a in mp and b in mp and mp[a] != mp[b]: pmap[frozenset({mp[a], mp[b]})] = t
    for (a, b), t in ge.items():
        conf[(t, pmap.get(frozenset({a, b}), "absent"))] += 1
    if (k+1) % 500 == 0: print(f"  {k+1}/{len(stems)}", flush=True)
print("\n=== 3-type confusion (rows=GT, cols=pred) ===")
hdr = TYPES + ["absent"]
print("gt\\pred," + ",".join(hdr) + ",support,recall")
for g in TYPES:
    row = [conf[(g, p)] for p in hdr]; sup = sum(row); rec = conf[(g, g)] / sup if sup else 0
    print(f"{g}," + ",".join(str(x) for x in row) + f",{sup},{rec:.4f}")
