"""
benchmark_runtime.py - real per-plan wall-clock timing and GPU memory for the
deployed pipeline (M1 GPU inference + M2-M4 CPU graph construction), excluding
GED (an evaluation-only computation, never part of deployment/inference).
"""
import argparse
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))


def _find_root(start: Path) -> Path:
    for cand in (start.parent, start, *start.parents):
        if (cand / "data" / "splits").is_dir():
            return cand
    return start.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()

    root = Path(args.root).resolve() if args.root else _find_root(_SCRIPT_DIR)
    ckpt = Path(args.ckpt) if args.ckpt else root / "checkpoints" / "segformer" / "best_model.pth"

    import cv2
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    from build_graph import build_graph_from_segmentation
    from proximity import compute_proximity_matrix
    from visualize import draw_bubble_diagram
    from inference import load_model, predict_mask

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")
    model = load_model(ckpt, device)

    stems = [s.strip() for s in (root / "data" / "splits" / "test.txt").read_text().splitlines() if s.strip()]
    stems = stems[: args.n + args.warmup]
    img_dir = root / "data" / "resplan_raster"

    m1_times, m2_times, m3_times, m4_times, total_times = [], [], [], [], []
    m4_failures = 0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for i, stem in enumerate(stems):
        img = cv2.imread(str(img_dir / f"{stem}.png"), cv2.IMREAD_COLOR)
        if img is None:
            continue

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        mask = predict_mask(model, img, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        G = build_graph_from_segmentation(mask)
        t2 = time.perf_counter()

        M = compute_proximity_matrix(G)
        t3 = time.perf_counter()

        render_failed = False
        try:
            draw_bubble_diagram(G)
        except Exception:
            render_failed = True
        t4 = time.perf_counter()
        plt.close("all")  # cleanup only, kept outside the timed M4 window

        if i < args.warmup:
            continue  # discard warmup iterations (first-call CUDA kernel compilation, cache fill)

        if render_failed:
            m4_failures += 1
            continue  # don't count a failed render as timed work

        m1_times.append(t1 - t0)
        m2_times.append(t2 - t1)
        m3_times.append(t3 - t2)
        m4_times.append(t4 - t3)
        total_times.append(t4 - t0)

    peak_mem_mb = None
    if device.type == "cuda":
        peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    n = np.array(m1_times)
    print(f"\nn = {len(n)} plans (after {args.warmup} warmup iterations discarded, "
          f"{m4_failures} M4 render failures excluded)")
    for name, arr in [("M1 (segmentation, GPU)", m1_times),
                      ("M2 (graph construction, CPU)", m2_times),
                      ("M3 (proximity matrix, CPU)", m3_times),
                      ("M4 (bubble diagram render, CPU)", m4_times),
                      ("Total (M1-M4)", total_times)]:
        arr = np.array(arr) * 1000  # ms
        print(f"  {name:35s}: {arr.mean():7.2f} +/- {arr.std():6.2f} ms  (median {np.median(arr):7.2f} ms)")

    if peak_mem_mb is not None:
        print(f"\nPeak GPU memory allocated: {peak_mem_mb:.1f} MB")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,} ({n_params / 1e6:.1f}M)")


if __name__ == "__main__":
    main()
