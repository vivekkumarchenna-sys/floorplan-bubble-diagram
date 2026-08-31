"""
eval_pipeline_variants.py - complete the two measurements left open in the paper
=================================================================================

Runs the FULL predicted-mask pipeline (M1 -> M2) over the test split and scores
every plan under a 2 x 2 design:

    edge matching : canonical instance-id   vs   id-independent assignment
    arch threshold: arch_min = 30 (published) vs arch_min = 6

Note on the arch_min sweep: against ResPlan's own ground truth (this script
uses build_gt_graph_from_resplan, not the old polygon reconstruction),
"recalibrating" arch_min does not help - the released data gives the arch
category zero ground-truth support at any threshold, so this comparison mainly
serves to confirm that finding also holds for the full predicted-mask pipeline,
not just the ground-truth-mask condition. It is not evidence for a fix.

Usage
-----
    python src/eval_pipeline_variants.py                       # full test split
    python src/eval_pipeline_variants.py --limit 200           # quick smoke test
    python src/eval_pipeline_variants.py --device cpu

Output
------
    results/pipeline_variants.csv   one row per plan, resumable
    a printed summary block with exactly the four numbers the manuscript needs

Runtime: a few minutes on a GPU. Safe to interrupt and re-run; it resumes.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))


def _find_root(start: Path) -> Path:
    """Locate the project root, i.e. the directory containing data/splits.

    Works whether this file sits in <root>/src/ (the GitHub layout) or directly
    in <root>/ (the flat archive layout).
    """
    for cand in (start.parent, start, *start.parents):
        if (cand / "data" / "splits").is_dir():
            return cand
    return start.parent

from build_graph import build_graph_from_segmentation
from build_gt_graph import build_gt_graph_from_resplan, load_resplan_records
from inference import load_model, predict_mask

BIG = 1e6
# "window" (from ResPlan's via_window) is a 4th ground-truth type the M2
# predictor never emits; see build_gt_graph.RESPLAN_EDGE_TYPE_MAP.
TYPES = ("door", "arch", "shared-wall", "window")


# ──────────────────────────────────────────────────────────────────────────────
# The two edge-matching conventions
# ──────────────────────────────────────────────────────────────────────────────

def classreset_edges(G):
    """
    Edges keyed by ClassName[instance_id] pair, where instance_id resets to 0
    for every class (Bathroom[0], Bathroom[1], ... regardless of how many
    bedrooms precede them in node-iteration order).

    This is NOT the convention the manuscript documents or that Table 5 was
    produced with (see Appendix D.3's own definition: "instance ids follow
    iteration order over classes and connected components" - i.e. a single
    global counter, never reset - which is what ``evaluate._extract_edges``
    implements). The two conventions happened to give statistically
    indistinguishable aggregate numbers against the old, wrong ground truth,
    which is presumably why the mislabeling here went unnoticed; against the
    corrected ResPlan ground truth they diverge visibly (e.g. 0.510 vs 0.664
    Edge F1 for the same upper-bound condition).

    Kept as a distinct, explicitly-labelled third matching convention
    (reported as "class-reset id", never as "canonical id") rather than
    deleted, since a full run's worth of results already exists under it and
    it is still a coherent, if non-standard, choice. Do not use this
    function's output for any number reported under the "canonical id"
    label - use ``evaluate.edge_metrics`` (via its ``_extract_edges``) for
    that instead.
    """
    idx, lab = {}, {}
    for n, d in G.nodes(data=True):
        c = d["class_name"]
        idx[c] = idx.get(c, 0)
        lab[n] = f"{c}[{idx[c]}]"
        idx[c] += 1
    return {tuple(sorted((lab[u], lab[v]))): d.get("edge_type")
            for u, v, d in G.edges(data=True)}


def _normalized_centroids(G):
    """
    Per-graph min-max normalised (row, col) in [0, 1], so centroid distance is
    comparable between two graphs that live in different coordinate systems
    (same scale/translation invariance a Hungarian assignment needs).

    G_pred's centroids come from ``build_graph_from_segmentation`` in the
    512x512 raster mask's (row, col) frame (row increases downward). G_gt's
    centroids, when G_gt is built by ``build_gt_graph_from_resplan``, come
    from ResPlan.pkl's own polygon geometry, which is a *different*,
    plan-specific coordinate system (no fixed scale or origin shared with the
    raster masks, and no rasterisation script in this repo records the
    transform between them). Measured over 397 single-kitchen test plans,
    pkl x correlates with mask col at r=0.83 (no flip), pkl y correlates with
    mask row at r=-0.71 (i.e. pkl y increases upward, mask row increases
    downward - a y-axis flip, not a coincidence). We therefore flip the
    y-coordinate for nodes carrying a ``resplan_name`` attribute (the marker
    ``build_gt_graph_from_resplan`` sets) before normalising, and otherwise
    assume the layout is axis-aligned (no rotation) between the two frames,
    which is consistent with both being top-down, wall-aligned floor plans.
    """
    is_resplan = any("resplan_name" in d for _, d in G.nodes(data=True))
    rows, cols = [], []
    for _, d in G.nodes(data=True):
        r, c = d["centroid"]
        if is_resplan:
            r = -r  # undo the y-up vs row-down convention mismatch
        rows.append(r)
        cols.append(c)
    if not rows:
        return {}
    rows, cols = np.array(rows), np.array(cols)
    r_span = rows.max() - rows.min() or 1.0
    c_span = cols.max() - cols.min() or 1.0
    rows = (rows - rows.min()) / r_span
    cols = (cols - cols.min()) / c_span
    return {n: (rows[i], cols[i]) for i, (n, _) in enumerate(G.nodes(data=True))}


def assign_rooms(G_pred, G_gt):
    """One-to-one room correspondence: min normalised-centroid distance, same class only."""
    P, Q = list(G_pred.nodes()), list(G_gt.nodes())
    if not P or not Q:
        return {}
    pred_xy = _normalized_centroids(G_pred)
    gt_xy = _normalized_centroids(G_gt)
    C = np.full((len(P), len(Q)), BIG)
    for i, p in enumerate(P):
        dp = G_pred.nodes[p]
        pr, pc = pred_xy[p]
        for j, g in enumerate(Q):
            dg = G_gt.nodes[g]
            if dp["class_name"] != dg["class_name"]:
                continue
            gr, gc = gt_xy[g]
            C[i, j] = float(np.hypot(pr - gr, pc - gc))
    r, c = linear_sum_assignment(C)
    return {P[i]: Q[j] for i, j in zip(r, c) if C[i, j] < BIG}


def idfree_edges(G_pred, G_gt):
    """Predicted edges relabelled into ground-truth node ids."""
    m = assign_rooms(G_pred, G_gt)
    pe = {}
    for k, (u, v, d) in enumerate(G_pred.edges(data=True)):
        if u in m and v in m:
            pe[tuple(sorted((m[u], m[v])))] = d.get("edge_type")
        else:
            pe[("UNMATCHED", k)] = d.get("edge_type")      # cannot correspond -> FP
    ge = {tuple(sorted((u, v))): d.get("edge_type")
          for u, v, d in G_gt.edges(data=True)}
    return pe, ge


def score(pe, ge):
    tp = set(pe) & set(ge)
    fp, fn = len(set(pe) - set(ge)), len(set(ge) - set(pe))
    eps = 1e-8
    p = len(tp) / (len(tp) + fp + eps)
    r = len(tp) / (len(tp) + fn + eps)
    f1 = 2 * p * r / (p + r + eps)
    ta = (sum(1 for k in tp if pe[k] == ge[k]) / len(tp)) if tp else 0.0
    return p, r, f1, ta, len(tp)


# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="project root; auto-detected if omitted")
    ap.add_argument("--ckpt", default=None, help="default: checkpoints/segformer/best_model.pth")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(args.root) if args.root else _find_root(_SCRIPT_DIR)
    print(f"[paths] project root: {root}")
    if args.ckpt:
        ckpt = Path(args.ckpt)
    else:
        ckpt = root / "checkpoints" / "segformer" / "best_model.pth"
        if not ckpt.exists():
            found = list(root.glob("checkpoints/**/best_model.pth"))
            if found:
                ckpt = found[0]
    if not ckpt.exists():
        sys.exit(f"[error] checkpoint not found: {ckpt}\n"
                 f"        pass --ckpt /path/to/best_model.pth")
    out = Path(args.out) if args.out else root / "results" / "pipeline_variants.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    stems = [l.strip() for l in (root / "data" / "splits" / f"{args.split}.txt").read_text().splitlines() if l.strip()]
    if args.limit:
        stems = stems[:args.limit]

    resplan_pkl = root / "data" / "resplan_raw" / "ResPlan.pkl"
    print(f"[gt] loading ResPlan's own graphs from {resplan_pkl}")
    resplan_records = load_resplan_records(resplan_pkl)

    done = set()
    if out.exists():
        try:
            done = set(pd.read_csv(out, usecols=["stem"])["stem"].astype(str))
            print(f"[resume] {len(done)} plans already scored in {out}")
        except Exception:
            pass
    todo = [s for s in stems if s not in done]
    if not todo:
        print("[done] nothing left to process")
    else:
        device = torch.device(args.device)
        model = load_model(ckpt, device)
        print(f"[run] {len(todo)} plans on {device}")

        rows, t0 = [], time.time()
        for i, stem in enumerate(todo, 1):
            img = cv2.imread(str(root / "data" / "resplan_raster" / f"{stem}.png"), cv2.IMREAD_COLOR)
            gtm = cv2.imread(str(root / "data" / "resplan_masks" / f"{stem}_mask.png"), cv2.IMREAD_GRAYSCALE)
            if img is None or gtm is None:
                continue
            try:
                pred_mask = predict_mask(model, img, device)
                plan_id = int(stem)
                if plan_id not in resplan_records:
                    continue
                G_gt = build_gt_graph_from_resplan(resplan_records[plan_id])
                if G_gt.number_of_nodes() == 0:
                    continue
                rec = {"stem": stem}
                for tag, am in (("a30", 30), ("a06", 6)):
                    G_pred = build_graph_from_segmentation(pred_mask, arch_min=am)
                    p, r, f1, ta, _ = score(classreset_edges(G_pred), classreset_edges(G_gt))
                    rec.update({f"classreset_{tag}_prec": p, f"classreset_{tag}_rec": r,
                                f"classreset_{tag}_f1": f1, f"classreset_{tag}_type": ta})
                    p, r, f1, ta, ntp = score(*idfree_edges(G_pred, G_gt))
                    rec.update({f"free_{tag}_prec": p, f"free_{tag}_rec": r,
                                f"free_{tag}_f1": f1, f"free_{tag}_type": ta,
                                f"free_{tag}_matched": ntp})
                rows.append(rec)
            except Exception as e:
                print(f"  [skip] {stem}: {e}")

            if i % 50 == 0 or i == len(todo):
                pd.DataFrame(rows).to_csv(out, mode="a", header=not out.exists(), index=False)
                rows = []
                el = time.time() - t0
                print(f"  {i}/{len(todo)}  {el:.0f}s  ({el/i:.2f}s/plan)", flush=True)
        if rows:
            pd.DataFrame(rows).to_csv(out, mode="a", header=not out.exists(), index=False)

    # ── summary ───────────────────────────────────────────────────────────────
    d = pd.read_csv(out)
    print("\n" + "=" * 78)
    print(f"  FULL PREDICTED-MASK PIPELINE, {args.split} split, n = {len(d)} plans")
    print("=" * 78)
    print(f"{'matching':<22}{'arch_min':>9}{'Edge P':>9}{'Edge R':>9}{'Edge F1':>9}{'type acc':>10}")
    print("-" * 78)
    for mtag, mname in (("classreset", "class-reset id (non-canonical)"), ("free", "id-independent")):
        for atag, am in (("a30", 30), ("a06", 6)):
            print(f"{mname:<22}{am:>9}"
                  f"{d[f'{mtag}_{atag}_prec'].mean():>9.4f}"
                  f"{d[f'{mtag}_{atag}_rec'].mean():>9.4f}"
                  f"{d[f'{mtag}_{atag}_f1'].mean():>9.4f}"
                  f"{d[f'{mtag}_{atag}_type'].mean():>10.4f}")
    print("-" * 78)
    print("\nFor the manuscript (NOTE: the authoritative canonical-id row for Table 5")
    print("  comes from evaluate.edge_metrics via inference.py, NOT from this script's")
    print("  'classreset' columns above - those are a distinct, non-canonical variant,")
    print("  see classreset_edges()'s docstring):")
    print(f"  Table 5, NEW row - pipeline under id-independent matching:")
    print(f"      {d['free_a30_prec'].mean():.4f} / {d['free_a30_rec'].mean():.4f} / "
          f"{d['free_a30_f1'].mean():.4f} / {d['free_a30_type'].mean():.4f}")
    print(f"  Threshold-sweep row (arch_min=6) - id-independent matching:")
    print(f"      edge-type accuracy {d['free_a06_type'].mean():.4f} "
          f"(was {d['free_a30_type'].mean():.4f})")
    print(f"\n  written to {out}")


if __name__ == "__main__":
    main()
