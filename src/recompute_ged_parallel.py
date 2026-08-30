"""
recompute_ged_parallel.py - GED recompute across BOTH conditions, in parallel
================================================================================
Uses evaluate.graph_edit_distance's process-based hard timeout (one
subprocess per GED call, terminated if it outlives the timeout - see that
function's docstring). Each *image* is additionally processed by a pool of
worker threads, so multiple images' GED subprocesses run concurrently rather
than one at a time. Threads (not processes) are used at this outer level
because each task mostly blocks on .join() waiting for its own GED
subprocess - the GIL is released during that wait, so this achieves real
concurrency without needing to pickle models/graphs across an outer process
boundary. Segmentation (GPU) and GED (CPU, one subprocess per call) both
happen inside each worker, so GPU and CPU work from different images overlap.

Usage:
    python src/recompute_ged_parallel.py --root <root> --condition upper_bound --ged-timeout 3.0 --workers 10 --out results/ged_upperbound.csv
    python src/recompute_ged_parallel.py --root <root> --condition pipeline --ckpt <ckpt> --ged-timeout 3.0 --workers 6 --out results/ged_pipeline.csv
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--condition", choices=["upper_bound", "pipeline"], required=True)
    ap.add_argument("--ckpt", default=None, help="required for --condition pipeline")
    ap.add_argument("--stems-from", default=None, help="CSV with a 'stem' column; defaults to data/splits/test.txt")
    ap.add_argument("--ged-timeout", type=float, default=3.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import cv2
    import numpy as np
    import pandas as pd
    from build_graph import build_graph_from_segmentation
    from build_gt_graph import build_gt_graph_from_resplan, load_resplan_records
    from evaluate import graph_edit_distance

    root = Path(args.root)
    records = load_resplan_records(root / "data" / "resplan_raw" / "ResPlan.pkl")

    if args.stems_from:
        stems = [str(s) for s in pd.read_csv(args.stems_from)["stem"].tolist()]
    else:
        stems = [s.strip() for s in (root / "data" / "splits" / "test.txt").read_text().splitlines() if s.strip()]

    model = None
    device = None
    model_lock = threading.Lock()
    if args.condition == "pipeline":
        import torch
        from inference import load_model
        assert args.ckpt, "--ckpt required for --condition pipeline"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_model(args.ckpt, device)

    mask_dir = root / "data" / "resplan_masks"
    img_dir = root / "data" / "resplan_raster"

    def process_one(stem):
        pid = int(stem)
        if pid not in records:
            return None
        try:
            G_gt = build_gt_graph_from_resplan(records[pid])
            if args.condition == "upper_bound":
                m = cv2.imread(str(mask_dir / f"{stem}_mask.png"), cv2.IMREAD_GRAYSCALE)
                if m is None:
                    return None
                m = m.astype(np.int64)
                G_pred = build_graph_from_segmentation(m)
            else:
                img = cv2.imread(str(img_dir / f"{stem}.png"), cv2.IMREAD_COLOR)
                if img is None:
                    return None
                from inference import predict_mask
                # PyTorch inference (eval mode, no grad inside predict_mask) is
                # safe to call concurrently from multiple threads on the same
                # CUDA device; the model_lock just serialises the forward pass
                # itself to avoid any shared-buffer surprises, GED (the real
                # bottleneck) still runs fully concurrently outside the lock.
                with model_lock:
                    pred_mask = predict_mask(model, img, device)
                G_pred = build_graph_from_segmentation(pred_mask)
            df = graph_edit_distance(G_pred, G_gt, timeout=args.ged_timeout)
            return {"stem": stem, "ged": df.iloc[0]["ged"]}
        except Exception:
            return {"stem": stem, "ged": float("nan")}

    rows = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process_one, s): s for s in stems}
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            if r is not None:
                rows.append(r)
            if i % 100 == 0:
                pd.DataFrame(rows).to_csv(args.out, index=False)
                elapsed = time.time() - t0
                n_valid = sum(1 for r in rows if r["ged"] == r["ged"])  # not-nan
                print(f"[{i}/{len(stems)}] {elapsed:.0f}s elapsed, "
                      f"{n_valid}/{len(rows)} valid so far", flush=True)

    result = pd.DataFrame(rows)
    result.to_csv(args.out, index=False)
    n_valid = result["ged"].notna().sum()
    print(f"Done: {n_valid}/{len(result)} valid GED ({n_valid/len(result)*100:.1f}%), "
          f"total time {time.time()-t0:.0f}s")
    print(f"Mean GED (valid only): {result['ged'].dropna().mean():.2f} +/- {result['ged'].dropna().std():.2f}")
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
