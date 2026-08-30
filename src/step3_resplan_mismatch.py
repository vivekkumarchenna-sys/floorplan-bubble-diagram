"""STEP 3 - quantify the ResPlan-GT mismatch (the headline finding).

For every plan: match corrected-GT edges (room pairs) to ResPlan's own graph by
class + area-rank, and cross-tabulate corrected-type x ResPlan-type. Reports:
  (a) the door-count gap: corrected door-edges/plan vs ResPlan door-edges/plan;
  (b) of corrected DOORS, what ResPlan calls them (the ~1/3 -> shared-wall claim);
  (c) that ResPlan has no valid room-to-room arch/open.
Runs single-process (loads ResPlan.pkl once). Optional arg = split file to restrict.
"""
import sys; sys.path.insert(0, "src")
import json, numpy as np
from collections import Counter, defaultdict
from build_gt_graph import build_gt_graph_from_resplan, load_resplan_records

CG = json.load(open("corrected_gt.json"))   # {stem: {rooms:[(rid,cls,name,apx)], edges:[(a,b,t)]}}

restrict = None
if len(sys.argv) > 1:
    restrict = set(l.strip() for l in open(sys.argv[1]) if l.strip())

print("loading ResPlan.pkl ...", flush=True)
recs = load_resplan_records("data/resplan_raw/ResPlan.pkl")
print(f"loaded {len(recs)} ResPlan records", flush=True)

# aggregate counters
corr_edges = Counter()      # corrected edge-type totals
rp_edges = Counter()        # ResPlan edge-type totals (room-to-room, mapped)
rp_edges_all = Counter()    # ResPlan edge-type totals (all edges in graph)
xtab = defaultdict(Counter) # corrected_type -> Counter(resplan_type incl 'absent')
nplans = 0
per_plan_corr_door = []; per_plan_rp_door = []; per_plan_rp_sw = []

for stem, data in CG.items():
    if restrict is not None and stem not in restrict:
        continue
    rid_cls = {rid: cls for (rid, cls, name, apx) in data["rooms"]}
    rid_area = {rid: apx for (rid, cls, name, apx) in data["rooms"]}
    ce = {frozenset({a, b}): t for (a, b, t) in data["edges"]}
    if not data["rooms"]:
        continue
    try:
        Gr = build_gt_graph_from_resplan(recs[int(stem)])
    except Exception:
        continue
    nplans += 1
    # --- match ResPlan nodes -> corrected rooms by class + area rank ---
    mask_by_cls = defaultdict(list)
    for rid, cls in rid_cls.items():
        mask_by_cls[cls].append((rid, rid_area[rid]))
    for c in mask_by_cls:
        mask_by_cls[c].sort(key=lambda t: t[1])
    rp_by_cls = defaultdict(list)
    for u in Gr.nodes():
        rp_by_cls[Gr.nodes[u]["class_id"]].append((u, Gr.nodes[u].get("area", 0)))
    for c in rp_by_cls:
        rp_by_cls[c].sort(key=lambda t: t[1])
    rp2mask = {}
    for c in rp_by_cls:
        for k, (u, _) in enumerate(rp_by_cls[c]):
            if c in mask_by_cls and k < len(mask_by_cls[c]):
                rp2mask[u] = mask_by_cls[c][k][0]
    # ResPlan edges mapped to corrected room pairs
    rpe = {}
    for u, v, d in Gr.edges(data=True):
        rp_edges_all[d.get("edge_type")] += 1
        if u in rp2mask and v in rp2mask and rp2mask[u] != rp2mask[v]:
            rpe[frozenset({rp2mask[u], rp2mask[v]})] = d.get("edge_type")
    # tallies
    for t in ce.values():
        corr_edges[t] += 1
    for t in rpe.values():
        rp_edges[t] += 1
    # cross-tab over corrected edges
    for pair, ct in ce.items():
        rt = rpe.get(pair, "absent")
        xtab[ct][rt] += 1
    per_plan_corr_door.append(sum(1 for t in ce.values() if t == "door"))
    per_plan_rp_door.append(sum(1 for t in rpe.values() if t == "door"))
    per_plan_rp_sw.append(sum(1 for t in rpe.values() if t == "shared-wall"))

print(f"\n=== ResPlan-GT mismatch over {nplans} plans"
      + (f" (restricted to {sys.argv[1]})" if restrict else " (FULL dataset)") + " ===\n")

print("Per-plan average edge counts:")
print(f"  corrected GT : door {corr_edges['door']/nplans:.2f}  open {corr_edges['open']/nplans:.2f}  "
      f"shared-wall {corr_edges['shared-wall']/nplans:.2f}")
print(f"  ResPlan graph: door {rp_edges['door']/nplans:.2f}  shared-wall {rp_edges['shared-wall']/nplans:.2f}  "
      f"arch {rp_edges['arch']/nplans:.2f}  window {rp_edges['window']/nplans:.2f}   (mapped room-to-room)")
print(f"  ResPlan (all edges in graph, unmapped): "
      + "  ".join(f"{k} {v/nplans:.2f}" for k, v in rp_edges_all.most_common()))
print()

print("Cross-tab: corrected-GT edge type  ->  what ResPlan calls the SAME room pair")
for ct in ["door", "open", "shared-wall"]:
    row = xtab[ct]; tot = sum(row.values())
    if not tot:
        continue
    print(f"\n  corrected '{ct}'  (n={tot}):")
    for rt, cnt in row.most_common():
        print(f"      -> ResPlan {rt:12s}: {cnt:6d}  ({100*cnt/tot:5.1f}%)")

# headline
dtot = sum(xtab["door"].values())
d_sw = xtab["door"]["shared-wall"]
d_door = xtab["door"]["door"]
d_absent = xtab["door"]["absent"]
print("\n--- HEADLINE ---")
print(f"  corrected doors matched to a ResPlan edge or gap: {dtot}")
print(f"  ResPlan labels {d_sw} of them 'shared-wall'  = {100*d_sw/dtot:.1f}% of real doors mislabelled")
print(f"  ResPlan labels {d_door} correctly as 'door'   = {100*d_door/dtot:.1f}%")
print(f"  ResPlan has NO edge for {d_absent}            = {100*d_absent/dtot:.1f}% (absent)")
print(f"  door-count gap: corrected {np.mean(per_plan_corr_door):.2f}/plan  vs  "
      f"ResPlan {np.mean(per_plan_rp_door):.2f}/plan   (ResPlan shared-wall {np.mean(per_plan_rp_sw):.2f}/plan)")
print(f"  ResPlan room-to-room 'arch': {rp_edges['arch']} total  ({rp_edges['arch']/nplans:.3f}/plan) -> effectively none")

# save machine-readable
out = {"nplans": nplans, "restrict": (sys.argv[1] if restrict else None),
       "corr_edges_per_plan": {k: corr_edges[k]/nplans for k in ["door", "open", "shared-wall"]},
       "rp_edges_per_plan": {k: rp_edges[k]/nplans for k in ["door", "shared-wall", "arch", "window"]},
       "xtab": {ct: dict(xtab[ct]) for ct in xtab},
       "door_mislabel_rate_shared": d_sw/dtot, "door_correct_rate": d_door/dtot}
json.dump(out, open("step3_mismatch.json", "w"), indent=2)
print("\nsaved step3_mismatch.json")
