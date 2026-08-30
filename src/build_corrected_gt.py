"""STEP 2 - build the corrected ground truth (typed edges + rooms) for every plan.
Saves consolidated CSVs + a JSON keyed by stem. Parallel, <=4 workers (handoff)."""
import sys; sys.path.insert(0, "src")
import cv2, numpy as np, json, os, csv, time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

MASK_DIR = "data/resplan_masks"

def work(stem):
    import truegraph_builder as tg
    m = cv2.imread(f"{MASK_DIR}/{stem}_mask.png", 0)
    if m is None:
        return None
    mask = m.astype(np.int64)
    R, edges = tg.build_true_graph(mask)
    if not R:
        return (stem, [], [])
    nm = tg.name_map(R, mask)
    rooms = [(i, c, nm[i], int(cm.sum())) for i, (c, cm) in R.items()]
    elist = [(int(a), int(b), t) for (a, b), t in edges.items()]
    return (stem, rooms, elist)

if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    stems = sorted({f[:-9] for f in os.listdir(MASK_DIR) if f.endswith("_mask.png")}, key=lambda s: int(s))
    if limit:
        stems = stems[:limit]
    scale = json.load(open("pixel_scale.json"))
    t0 = time.time()
    results = {}
    agg = Counter(); nplans = 0
    with ProcessPoolExecutor(max_workers=4) as ex:
        for i, r in enumerate(ex.map(work, stems, chunksize=16)):
            if r is None:
                continue
            stem, rooms, elist = r
            results[stem] = {"rooms": rooms, "edges": elist}
            nplans += 1
            for _, _, t in elist:
                agg[t] += 1
            agg["rooms"] += len(rooms)
            if (i + 1) % 2000 == 0:
                print(f"  {i+1}/{len(stems)}  ({time.time()-t0:.0f}s)", flush=True)

    # write edges CSV
    with open("corrected_gt_edges.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stem", "room_a", "room_b", "edge_type"])
        for stem in sorted(results, key=lambda s: int(s)):
            for a, b, t in results[stem]["edges"]:
                w.writerow([stem, a, b, t])
    # write rooms CSV
    with open("corrected_gt_rooms.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stem", "room_id", "class", "name", "area_px", "area_sqm"])
        for stem in sorted(results, key=lambda s: int(s)):
            sc = scale.get(str(stem))
            for i, c, name, apx in results[stem]["rooms"]:
                w.writerow([stem, i, c, name, apx, round(apx * sc, 3) if sc else ""])
    # write JSON
    json.dump(results, open("corrected_gt.json", "w"))

    print(f"\n=== corrected GT built for {nplans} plans in {time.time()-t0:.0f}s ===")
    print(f"total: door={agg['door']} open={agg['open']} shared-wall={agg['shared-wall']} rooms={agg['rooms']}")
    print(f"per plan: {agg['door']/nplans:.2f} door, {agg['open']/nplans:.2f} open, "
          f"{agg['shared-wall']/nplans:.2f} shared-wall, {agg['rooms']/nplans:.2f} rooms")
    print("saved corrected_gt_edges.csv, corrected_gt_rooms.csv, corrected_gt.json")
