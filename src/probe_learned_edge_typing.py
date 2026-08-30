"""
probe_learned_edge_typing.py - does a learned classifier beat the rule for
edge TYPING (door / shared-wall / window), given the same features Rule 2
already computes?

This is a bounded probe, not a replacement for M2. It answers one question a
reviewer is entitled to ask: is the residual shared-wall/door confusion
(Section 6.5) a limitation of a fixed threshold, recoverable by a learned
classifier over the same geometric evidence, or does it persist regardless of
classifier? M2 stays rule-based in the reported pipeline; this only tests
whether the same three numbers (overlap_px, door_px, opening_width) plus room
context support a better typing decision than a single arch_min threshold.

Runs on ground-truth masks (isolates typing from segmentation error, same
condition as the arch_min sweep in Section 6.5) across the full train/val/test
split. For every M2 candidate edge that both exists in and is matched (by the
same id-independent room assignment used elsewhere in this project) to a real
ResPlan edge, records:

    overlap_px, door_px, opening_width,
    class_a, class_b (as one-hot / ordinal), area_a, area_b, centroid_dist

label = ResPlan's own edge_type for that matched pair (door / shared-wall /
window; arch excluded, zero support - see build_gt_graph.py).

Usage
-----
    python src/probe_learned_edge_typing.py --root .
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))


def _available_memory_bytes() -> int | None:
    """Best-effort free-RAM lookup with no extra dependency (no psutil in
    requirements.txt). Returns None if it can't be determined, so callers
    must have a conservative fallback."""
    try:
        if sys.platform == "win32":
            import ctypes

            class _MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = _MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return int(stat.ullAvailPhys)
            return None
        else:
            page_size = os.sysconf("SC_PAGE_SIZE")
            avail_pages = os.sysconf("SC_AVPHYS_PAGES")
            return int(page_size * avail_pages)
    except Exception:
        return None


def _safe_worker_count(records_by_id, requested: int) -> int:
    """Cap the process-pool size so ``records_by_id`` (pickled once per
    worker spawn by ProcessPoolExecutor's initargs) can't outrun free RAM.

    records_by_id unpickles from a ~297MB ResPlan.pkl; a naive
    cpu_count()-2 worker count (22+ on a many-core box) previously caused a
    MemoryError here, since every worker gets its own pickled copy on spawn
    plus its own live, deserialised copy held for the pool's lifetime.
    Measured directly rather than guessed, since "a few workers" isn't a
    number - free RAM divided by the actual pickle size is.
    """
    import pickle

    try:
        pickle_size = len(pickle.dumps(records_by_id, protocol=pickle.HIGHEST_PROTOCOL))
    except Exception:
        pickle_size = 297 * 1024 * 1024  # fallback: known ResPlan.pkl size

    avail = _available_memory_bytes()
    if avail is None:
        avail = 4 * 1024 * 1024 * 1024  # conservative 4GB fallback

    # Each worker needs roughly 2x pickle_size (the pickled bytes in flight
    # during spawn plus the live deserialised object held afterwards); only
    # spend 60% of currently-free RAM on this so the main process and each
    # worker's own mask/graph processing still have headroom.
    per_worker = max(pickle_size * 2, 1)
    budget = int(avail * 0.6)
    by_memory = max(budget // per_worker, 1)

    safe = max(1, min(requested, by_memory))
    if safe < requested:
        print(f"[workers] capping pool at {safe} (requested {requested}): "
              f"records_by_id pickles to {pickle_size / 1e6:.0f}MB, "
              f"{avail / 1e9:.1f}GB free RAM")
    return safe


def _find_root(start: Path) -> Path:
    for cand in (start.parent, start, *start.parents):
        if (cand / "data" / "splits").is_dir():
            return cand
    return start.parent


# ── parallel worker (module-level: required for Windows "spawn" pickling) ──
# Each worker process loads ResPlan.pkl and imports the heavy modules once,
# via _init_worker, rather than once per task.
_worker_root = None
_worker_records = None


def _init_worker(root_str, records_by_id):
    global _worker_root, _worker_records
    _worker_root = Path(root_str)
    _worker_records = records_by_id


def _process_stem(stem):
    import cv2
    import numpy as np
    from build_graph import build_graph_from_segmentation
    from build_gt_graph import build_gt_graph_from_resplan
    from eval_pipeline_variants import assign_rooms

    mask_dir = _worker_root / "data" / "resplan_masks"
    gt_mask = cv2.imread(str(mask_dir / f"{stem}_mask.png"), cv2.IMREAD_GRAYSCALE)
    if gt_mask is None:
        return []
    try:
        G_pred = build_graph_from_segmentation(gt_mask)
    except Exception:
        return []
    rec = _worker_records.get(int(stem))
    if rec is None:
        return []
    try:
        G_gt = build_gt_graph_from_resplan(rec)
    except Exception:
        return []
    if G_gt.number_of_nodes() == 0 or G_pred.number_of_nodes() == 0:
        return []

    room_map = assign_rooms(G_pred, G_gt)
    gt_edges = {tuple(sorted((u, v))): d.get("edge_type")
                for u, v, d in G_gt.edges(data=True)}

    rows = []
    for u, v, d in G_pred.edges(data=True):
        if u not in room_map or v not in room_map:
            continue
        gu, gv = room_map[u], room_map[v]
        key = tuple(sorted((gu, gv)))
        label = gt_edges.get(key)
        if label is None or label == "arch":
            continue  # not matched to a real edge, or arch (zero support)
        nu, nv = G_pred.nodes[u], G_pred.nodes[v]
        # order the pair once (by class name) and keep every feature tied
        # to that same room throughout, rather than sorting each feature
        # independently - which would let class_a/area_a/centroid describe
        # different rooms
        if nu["class_name"] <= nv["class_name"]:
            first, second = nu, nv
        else:
            first, second = nv, nu
        class_a, class_b = first["class_name"], second["class_name"]
        area_a, area_b = first["area_px"], second["area_px"]
        row_a, col_a = first["centroid"]
        row_b, col_b = second["centroid"]
        dist = float(np.hypot(row_a - row_b, col_a - col_b))
        rows.append({
            "stem": stem,
            "overlap_px": d["overlap_px"],
            "door_px": d["door_px"],
            "opening_width": d["opening_width"],
            "rule_pred_type": d["edge_type"],
            "class_a": class_a, "class_b": class_b,
            "area_a": area_a, "area_b": area_b,
            "centroid_dist": dist,
            "label": label,
        })
    return rows


def extract_features_for_split(root, stems, records_by_id, max_workers=None, executor=None):
    """
    If ``executor`` is given (a live ProcessPoolExecutor already initialised
    with this ``root``/``records_by_id``), reuse it instead of paying pool
    startup and records-pickling cost again - main() reuses one pool across
    all three splits rather than recreating it per split.
    """
    import pandas as pd
    from tqdm import tqdm

    rows = []
    if executor is not None:
        for stem_rows in tqdm(executor.map(_process_stem, stems, chunksize=8),
                              total=len(stems), desc="extracting features"):
            rows.extend(stem_rows)
        return pd.DataFrame(rows)

    from concurrent.futures import ProcessPoolExecutor
    requested = max_workers or max((os.cpu_count() or 1) - 2, 1)
    max_workers = _safe_worker_count(records_by_id, requested)
    with ProcessPoolExecutor(
        max_workers=max_workers, initializer=_init_worker, initargs=(str(root), records_by_id),
    ) as ex:
        for stem_rows in tqdm(ex.map(_process_stem, stems, chunksize=8),
                              total=len(stems), desc="extracting features"):
            rows.extend(stem_rows)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--out-prefix", default="results/probe_edge_typing")
    ap.add_argument("--workers", type=int, default=None,
                    help="process pool size (default: cpu_count() - 2)")
    args = ap.parse_args()

    root = Path(args.root).resolve() if args.root else _find_root(_SCRIPT_DIR)
    from build_gt_graph import load_resplan_records
    from concurrent.futures import ProcessPoolExecutor

    records_by_id = load_resplan_records(root / "data" / "resplan_raw" / "ResPlan.pkl")
    requested = args.workers or max((os.cpu_count() or 1) - 2, 1)
    max_workers = _safe_worker_count(records_by_id, requested)
    print(f"Using {max_workers} worker processes")

    with ProcessPoolExecutor(
        max_workers=max_workers, initializer=_init_worker, initargs=(str(root), records_by_id),
    ) as ex:
        for split in ("train", "val", "test"):
            stems = [s.strip() for s in
                     (root / "data" / "splits" / f"{split}.txt").read_text().splitlines() if s.strip()]
            df = extract_features_for_split(root, stems, records_by_id, executor=ex)
            out_path = root / f"{args.out_prefix}_{split}.csv"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_path, index=False)
            print(f"{split}: {len(df)} matched edges -> {out_path}")


if __name__ == "__main__":
    main()
