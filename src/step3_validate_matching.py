"""Validate the ResPlan matching on architect-confirmed plans + quantify the
multi-room-hub contribution to the door-count gap (publication safety check)."""
import sys; sys.path.insert(0, "src")
import cv2, numpy as np, json
from collections import Counter, defaultdict
import truegraph_builder as tg
from build_gt_graph import build_gt_graph_from_resplan, load_resplan_records

recs = load_resplan_records("data/resplan_raw/ResPlan.pkl")

def analyze(stem):
    mask = cv2.imread(f"data/resplan_masks/{stem}_mask.png", 0).astype(np.int64)
    R, edges = tg.build_true_graph(mask)
    nm = tg.name_map(R, mask)
    ce = {frozenset({a, b}): t for (a, b), t in edges.items()}
    # match ResPlan by class+area rank
    rid_cls = {i: c for i, (c, cm) in R.items()}
    rid_area = {i: int(cm.sum()) for i, (c, cm) in R.items()}
    mask_by_cls = defaultdict(list)
    for i in R: mask_by_cls[rid_cls[i]].append((i, rid_area[i]))
    for c in mask_by_cls: mask_by_cls[c].sort(key=lambda t: t[1])
    Gr = build_gt_graph_from_resplan(recs[int(stem)])
    rp_by_cls = defaultdict(list)
    for u in Gr.nodes(): rp_by_cls[Gr.nodes[u]["class_id"]].append((u, Gr.nodes[u].get("area", 0)))
    for c in rp_by_cls: rp_by_cls[c].sort(key=lambda t: t[1])
    rp2mask = {}
    for c in rp_by_cls:
        for k, (u, _) in enumerate(rp_by_cls[c]):
            if c in mask_by_cls and k < len(mask_by_cls[c]): rp2mask[u] = mask_by_cls[c][k][0]
    rpe = {}
    for u, v, d in Gr.edges(data=True):
        if u in rp2mask and v in rp2mask and rp2mask[u] != rp2mask[v]:
            rpe[frozenset({rp2mask[u], rp2mask[v]})] = d.get("edge_type")
    print(f"\n=== plan {stem} ===")
    doors = [e for e, t in ce.items() if t == "door"]
    mis = []
    for e in doors:
        rt = rpe.get(e, "absent")
        a, b = sorted(e, key=lambda i: nm[i])
        tag = "OK" if rt == "door" else ("-> shared-wall MISLABEL" if rt == "shared-wall" else f"-> {rt}")
        print(f"  door  {nm[a]:11s}-- {nm[b]:11s}: ResPlan {rt:12s} {tag}")
        if rt != "door": mis.append((nm[a], nm[b], rt))
    print(f"  corrected doors={len(doors)}  ResPlan-shared={sum(1 for e in doors if rpe.get(e)=='shared-wall')}"
          f"  ResPlan-absent={sum(1 for e in doors if e not in rpe)}")

for stem in ["16649", "15389", "13388", "5674"]:
    analyze(stem)

# multi-room hub contribution across full dataset
print("\n=== multi-room-hub door-blob contribution (full dataset) ===")
CG = json.load(open("corrected_gt.json"))
# recompute door-blob multiplicities via geometry on a sample for speed
import os, random
random.seed(7)
stems = sorted({f[:-9] for f in os.listdir("data/resplan_masks") if f.endswith("_mask.png")}, key=lambda s: int(s))
samp = random.sample(stems, 500)
from scipy import ndimage
blob_edges = Counter()   # how many door edges per physical door blob
nblob = 0; ndooredge = 0
for stem in samp:
    mask = cv2.imread(f"data/resplan_masks/{stem}_mask.png", 0).astype(np.int64)
    R = tg.rooms_of(mask)
    door = (mask == 11); lab, n = ndimage.label(door)
    for k in range(1, n + 1):
        dc = (lab == k)
        if dc.sum() < 15: continue
        dd = ndimage.binary_dilation(dc, iterations=8)
        touched = [(i, (dd & cm).sum()) for i, (c, cm) in R.items() if (dd & cm).sum() > 20]
        if len(touched) < 2: continue
        touched.sort(key=lambda t: -t[1]); hub, hov = touched[0]
        e = sum(1 for j, ov in touched[1:] if ov >= 0.4 * hov)
        if e >= 1:
            nblob += 1; ndooredge += e; blob_edges[e] += 1
print(f"  sample plans={len(samp)}  physical door blobs={nblob}  door edges={ndooredge}")
print(f"  edges-per-blob distribution: {dict(sorted(blob_edges.items()))}")
print(f"  mean edges/blob = {ndooredge/nblob:.3f}  (1.0 = every door joins exactly one pair)")
multi = sum(v for k, v in blob_edges.items() if k >= 2)
print(f"  multi-room door blobs (>=2 edges): {multi}/{nblob} = {100*multi/nblob:.1f}% of door blobs")
print(f"  => extra edges from multi-room rule: {ndooredge-nblob} of {ndooredge} door edges = {100*(ndooredge-nblob)/ndooredge:.1f}%")
