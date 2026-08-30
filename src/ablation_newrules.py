"""Ablation of the new M2 rules: disable each component, score vs the full corrected
GT (on GT masks, so M1 is held fixed). Shows each rule's contribution."""
import sys; sys.path.insert(0, "src")
import cv2, numpy as np
from scipy import ndimage
from collections import Counter
import truegraph_builder as tg

def build_variant(mask, door_precedence=True, open_detect=True, hub_multiroom=True,
                  adj_dil=9, min_share=25, door_dil=8, min_wall_contact=250, door_frac=0.4, open_min=45):
    R = tg.rooms_of(mask)
    wall = (mask == 10)
    dils = {i: ndimage.binary_dilation(cm, iterations=adj_dil) for i, (c, cm) in R.items()}
    door = (mask == 11); lab, n = ndimage.label(door)
    door_edges = {}
    for k in range(1, n + 1):
        dc = (lab == k)
        if dc.sum() < 15: continue
        dd = ndimage.binary_dilation(dc, iterations=door_dil)
        touched = [(i, (dd & cm).sum()) for i, (c, cm) in R.items() if (dd & cm).sum() > 20]
        if len(touched) < 2: continue
        touched.sort(key=lambda t: -t[1]); hub, hov = touched[0]
        rest = touched[1:]
        if not hub_multiroom:
            rest = rest[:1]                      # only the single best partner
        for j, ov in rest:
            if ov >= door_frac * hov:
                a, b = sorted([hub, j]); door_edges[(a, b)] = 'door'
    edges = dict(door_edges)
    ids = list(R)
    tight = {i: ndimage.binary_dilation(cm, iterations=3) for i, (c, cm) in R.items()}
    for ii in range(len(ids)):
        for jj in range(ii + 1, len(ids)):
            a, b = ids[ii], ids[jj]
            has_door = (a, b) in door_edges
            zone = dils[a] & dils[b]
            if zone.sum() < min_share:
                continue
            wallc = (zone & wall).sum()
            opening = (tight[a] & R[b][1]).sum()
            if has_door:
                if door_precedence:
                    continue                      # door wins (default)
                # no precedence: a solid wall between the pair overrides the door
                if wallc >= min_wall_contact and opening < open_min:
                    edges[(a, b)] = 'shared-wall'
                continue
            if open_detect and opening >= open_min:
                edges[(a, b)] = 'open'
            elif wallc >= min_wall_contact:
                edges[(a, b)] = 'shared-wall'
    return R, edges

def score(ref, pred):
    refs = {frozenset(k): v for k, v in ref.items()}
    prd = {frozenset(k): v for k, v in pred.items()}
    inter = set(refs) & set(prd)
    P = len(inter)/len(prd) if prd else 0; Rc = len(inter)/len(refs) if refs else 0
    F1 = 2*P*Rc/(P+Rc) if (P+Rc) else 0
    ta = sum(1 for e in inter if prd[e]==refs[e])/len(inter) if inter else 0
    rec = Counter()
    for e,t in refs.items():
        rec['gt_'+t]+=1
        if prd.get(e)==t: rec['ok_'+t]+=1
    return F1, ta, rec

if __name__ == "__main__":
    import random
    random.seed(3)
    stems = [l.strip() for l in open("data/splits/test.txt") if l.strip()]
    stems = random.sample(stems, 800)
    variants = {
        "Full rules": dict(),
        "− door precedence": dict(door_precedence=False),
        "− open-passage detection": dict(open_detect=False),
        "− multi-room hub rule": dict(hub_multiroom=False),
    }
    agg = {k: (Counter(), []) for k in variants}
    for stem in stems:
        m = cv2.imread(f"data/resplan_masks/{stem}_mask.png", 0)
        if m is None: continue
        mask = m.astype(np.int64)
        R, ref = tg.build_true_graph(mask)          # full corrected GT
        if not ref: continue
        for name, kw in variants.items():
            _, pe = build_variant(mask, **kw)
            F1, ta, rec = score(ref, pe)
            agg[name][0].update(rec); agg[name][1].append((F1, ta))
    print(f"Ablation of the new M2 rules on {len(stems)} test plans (GT masks; scored vs full corrected GT)")
    print(f"{'variant':26s} {'EdgeF1':>7s} {'typeAcc':>8s} {'doorRec':>8s} {'openRec':>8s} {'swRec':>7s}")
    for name in variants:
        c, fts = agg[name]
        f1 = np.mean([x[0] for x in fts]); ta = np.mean([x[1] for x in fts])
        dr = c['ok_door']/max(1,c['gt_door']); orc = c['ok_open']/max(1,c['gt_open']); sw = c['ok_shared-wall']/max(1,c['gt_shared-wall'])
        print(f"{name:26s} {f1:7.3f} {ta:8.3f} {dr:8.3f} {orc:8.3f} {sw:7.3f}")
