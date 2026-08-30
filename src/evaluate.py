"""
evaluate.py - Evaluation metrics for segmentation + graph pipeline
===================================================================
Four metric groups:

1. ``compute_miou`` - per-class IoU, precision, recall from masks
2. ``edge_metrics`` - edge-level precision/recall/F1 + type accuracy
3. ``graph_edit_distance`` - GED between predicted and ground-truth graphs
4. ``frobenius_norm`` - Frobenius distance between proximity matrices

All functions return pandas DataFrames for easy inspection and export.

Colab usage:
    import sys
    sys.path.insert(0, "/content/drive/MyDrive/bubble_diagram_project")

    from evaluate import compute_miou, edge_metrics, graph_edit_distance, frobenius_norm
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
except NameError:
    sys.path.insert(0, "/content/drive/MyDrive/bubble_diagram_project")

import numpy as np
import networkx as nx
import pandas as pd

from build_graph import CLASS_NAMES

# ══════════════════════════════════════════════════════════════════════════════
# 1.  Segmentation metrics  (per-class IoU, precision, recall)
# ══════════════════════════════════════════════════════════════════════════════

def compute_miou(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int = 16,
    ignore_index: int = 255,
    class_names: dict[int, str] | None = None,
) -> pd.DataFrame:
    """
    Per-class IoU, precision, and recall from two label maps.

    Parameters
    ----------
    pred, target : (H, W) integer arrays with class ids 0..num_classes-1.
    num_classes  : number of semantic classes.
    ignore_index : pixels with this label in *target* are excluded.
    class_names  : optional id→name mapping (defaults to CLASS_NAMES).

    Returns
    -------
    DataFrame with columns [class_id, class_name, iou, precision, recall, support]
    plus a final "mean" row (macro-average over classes with support > 0).
    """
    if class_names is None:
        class_names = CLASS_NAMES

    pred = pred.astype(np.int64).ravel()
    target = target.astype(np.int64).ravel()

    # mask out ignored pixels
    valid = target != ignore_index
    pred = pred[valid]
    target = target[valid]

    # confusion matrix via bincount
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    idx = target * num_classes + pred
    np.add.at(cm.ravel(), idx, 1)

    tp = np.diag(cm)                        # true positives per class
    fp = cm.sum(axis=0) - tp                # false positives
    fn = cm.sum(axis=1) - tp                # false negatives
    support = cm.sum(axis=1)                # pixels per class in target

    eps = 1e-8
    iou       = tp / (tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)

    rows = []
    for c in range(num_classes):
        rows.append({
            "class_id":   c,
            "class_name": class_names.get(c, f"class_{c}"),
            "iou":        float(iou[c]),
            "precision":  float(precision[c]),
            "recall":     float(recall[c]),
            "support":    int(support[c]),
        })

    # macro-average over classes that actually appear in ground truth
    present = support > 0
    rows.append({
        "class_id":   -1,
        "class_name": "mean",
        "iou":        float(iou[present].mean())       if present.any() else 0.0,
        "precision":  float(precision[present].mean())  if present.any() else 0.0,
        "recall":     float(recall[present].mean())     if present.any() else 0.0,
        "support":    int(support.sum()),
    })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Edge metrics  (precision / recall / F1 + edge-type accuracy)
# ══════════════════════════════════════════════════════════════════════════════

def _edge_key(u_name: str, v_name: str) -> tuple[str, str]:
    """Canonical (sorted) room-name pair so A-B == B-A."""
    return tuple(sorted((u_name, v_name)))


def _extract_edges(G: nx.Graph) -> dict[tuple[str, str], str]:
    """Return {(roomA, roomB): edge_type} from a room-adjacency graph."""
    edges = {}
    for u, v, data in G.edges(data=True):
        u_name = G.nodes[u].get("class_name", str(u))
        v_name = G.nodes[v].get("class_name", str(v))
        # append instance id to handle multiple rooms of same type
        u_label = f"{u_name}[{u}]"
        v_label = f"{v_name}[{v}]"
        key = _edge_key(u_label, v_label)
        edges[key] = data.get("edge_type", "shared-wall")
    return edges


def edge_metrics(
    G_pred: nx.Graph,
    G_gt: nx.Graph,
) -> pd.DataFrame:
    """
    Edge-level precision, recall, F1 and edge-type accuracy.

    Edges are matched by their canonical room-label pair (ClassName[id]).

    Returns
    -------
    DataFrame with one row containing:
        edge_precision, edge_recall, edge_f1,
        type_accuracy  (fraction of matched edges with correct edge_type),
        tp, fp, fn, matched_correct_type, matched_total
    """
    pred_edges = _extract_edges(G_pred)
    gt_edges   = _extract_edges(G_gt)

    pred_keys = set(pred_edges.keys())
    gt_keys   = set(gt_edges.keys())

    tp_keys = pred_keys & gt_keys
    fp_keys = pred_keys - gt_keys
    fn_keys = gt_keys - pred_keys

    tp = len(tp_keys)
    fp = len(fp_keys)
    fn = len(fn_keys)

    eps = 1e-8
    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)

    # edge-type accuracy among matched edges
    correct_type = sum(
        1 for k in tp_keys if pred_edges[k] == gt_edges[k]
    )
    type_acc = correct_type / tp if tp > 0 else 0.0

    return pd.DataFrame([{
        "edge_precision":       round(precision, 4),
        "edge_recall":          round(recall, 4),
        "edge_f1":              round(f1, 4),
        "type_accuracy":        round(type_acc, 4),
        "tp":                   tp,
        "fp":                   fp,
        "fn":                   fn,
        "matched_correct_type": correct_type,
        "matched_total":        tp,
    }])


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Graph Edit Distance
# ══════════════════════════════════════════════════════════════════════════════

import multiprocessing as _mp
import threading as _threading


def _ged_mp_worker(q, G_pred, G_gt, node_cost, edge_cost):
    """
    Module-level (picklable) worker for the process-based GED search below.
    Must live at module scope, not as a nested closure, because Windows'
    ``spawn`` start method pickles the target and re-imports the defining
    module in the child process.
    """
    def node_subst(n1, n2):
        return 0.0 if n1.get("class_name") == n2.get("class_name") else node_cost

    def node_del(n):
        return node_cost

    def node_ins(n):
        return node_cost

    def edge_subst(e1, e2):
        return 0.0 if e1.get("edge_type") == e2.get("edge_type") else edge_cost

    def edge_del(e):
        return edge_cost

    def edge_ins(e):
        return edge_cost

    try:
        best = None
        for ged in nx.optimize_graph_edit_distance(
            G_pred, G_gt,
            node_subst_cost=node_subst,
            node_del_cost=node_del,
            node_ins_cost=node_ins,
            edge_subst_cost=edge_subst,
            edge_del_cost=edge_del,
            edge_ins_cost=edge_ins,
        ):
            best = ged
            q.put(best)   # progressively better values; main process keeps the last
    except Exception:
        pass


# Whether the process-based path has already failed once in this process
# (e.g. a sandboxed environment that blocks process spawning). Once it does,
# stop retrying it for every subsequent call - fall back to the thread-based
# path directly, which is slower per call in aggregate but always available.
_ged_mp_broken = False


def graph_edit_distance(
    G_pred: nx.Graph,
    G_gt: nx.Graph,
    node_cost: float = 1.0,
    edge_cost: float = 1.0,
    timeout: float = 30.0,
    max_nodes: int = 20,
) -> pd.DataFrame:
    """
    Approximate graph edit distance (GED) between two room graphs.

    Uses NetworkX's ``optimize_graph_edit_distance`` with class-aware
    node substitution: substituting a node with the same class_name
    costs 0, different class_name costs ``node_cost``.

    Parameters
    ----------
    node_cost : cost for inserting/deleting/substituting (different class) a node.
    edge_cost : cost for inserting/deleting/substituting (different type) an edge.
    timeout   : seconds before returning the best-so-far approximation - a
                **hard** cutoff, not a cooperative one: certain graph pairs
                (confirmed reproducible on a 10-vs-12-node pair, stem 4058)
                can take minutes to yield even their *first* candidate from
                ``optimize_graph_edit_distance`` - likely because many nodes
                share the same class, so the substitution search is highly
                combinatorial - and a cooperative check between yields never
                fires in that case.
    max_nodes : if either graph has more nodes than this, GED is not attempted
                and ``nan`` is returned immediately, as a fast first-line
                defense before paying process/thread-spawn overhead. ResPlan
                plans average 8.2 rooms, so this excludes rare outliers
                rather than typical plans.

    Implementation note - hard timeout via a real subprocess, not a thread
    ------------------------------------------------------------------------
    Two approaches were tried before this one, in order, both documented
    here because the reasoning that ruled each out matters for anyone
    revisiting this function:

    1. A cooperative check inside the search loop - abandoned, because it
       only fires between yields, and the stem-4058 case above can take
       minutes to yield even once.
    2. A ``threading.Thread(daemon=True)`` with a queue and
       ``queue.get(timeout=...)`` - works for the individual-call timeout,
       but a **second-order problem** showed up at batch scale: in a tight
       loop over thousands of plans in one process, if a meaningful fraction
       hit the timeout, the abandoned threads accumulate, and because they
       are pure-Python and hold the GIL in turn, the *calling* process's own
       throughput degrades sharply as more pile up (measured directly: a
       batch run's rate went from ~2s/plan to effectively stalled after
       ~100 plans, GPU utilisation near zero throughout - not a hang, GIL
       contention from its own earlier orphaned GED threads). Capping the
       number of outstanding threads bounded the slowdown but created a
       *coverage* problem instead: once genuinely slow pairs occupy the cap
       early in a long run, every later plan is skipped for as long as they
       remain stuck, which for a full 2,567-plan run left only ~2% of plans
       with any GED value at all - too thin to report with confidence.

    This function now spawns a real ``multiprocessing.Process`` per call and
    calls ``.terminate()`` on it if it outlives the timeout. This was
    originally avoided: an earlier attempt at repeated ``Process`` spawning
    in this environment reportedly hit ``PermissionError: [WinError 5]`` /
    ``BrokenPipeError``. Re-tested directly against this codebase's own
    graphs - 200 real spawn-and-possibly-terminate cycles, zero errors - and
    it does not reproduce; whatever caused it before is not present here.
    A terminated OS process is actually gone (unlike an abandoned thread),
    so neither the throughput problem nor the coverage problem above occurs:
    measured on the same 200-call sample, 159/200 (79.5%) converged within a
    3-second timeout, against roughly 2% for the thread-based path at batch
    scale. If process creation fails for any reason (e.g. a sandboxed
    environment that blocks it, matching what an earlier session reported),
    this function falls back to the thread-based implementation
    automatically and remembers not to retry the process path for the rest
    of the run, so a single blocked environment costs one failed attempt,
    not a slowdown on every call.

    Returns
    -------
    DataFrame with columns [ged, n_pred, n_gt, e_pred, e_gt, timed_out].
    ``ged`` is ``nan`` if either graph exceeds ``max_nodes``, or if the
    timeout fires before the search yields even one candidate.
    """
    if G_pred.number_of_nodes() > max_nodes or G_gt.number_of_nodes() > max_nodes:
        return pd.DataFrame([{
            "ged":    float("nan"),
            "n_pred": G_pred.number_of_nodes(),
            "n_gt":   G_gt.number_of_nodes(),
            "e_pred": G_pred.number_of_edges(),
            "e_gt":   G_gt.number_of_edges(),
            "timed_out": False,
        }])

    global _ged_mp_broken
    if not _ged_mp_broken:
        try:
            return _graph_edit_distance_mp(G_pred, G_gt, node_cost, edge_cost, timeout)
        except Exception:
            _ged_mp_broken = True   # don't keep paying the failed-spawn cost every call

    return _graph_edit_distance_thread_fallback(G_pred, G_gt, node_cost, edge_cost, timeout)


def _graph_edit_distance_mp(G_pred, G_gt, node_cost, edge_cost, timeout):
    """Process-based GED with a real hard kill. See graph_edit_distance's docstring."""
    q = _mp.Queue()
    p = _mp.Process(target=_ged_mp_worker, args=(q, G_pred, G_gt, node_cost, edge_cost))
    p.start()
    p.join(timeout=timeout)
    timed_out = p.is_alive()
    if timed_out:
        p.terminate()
        p.join(timeout=2.0)

    best_ged = float("inf")
    got_any = False
    while True:
        try:
            best_ged = q.get_nowait()
            got_any = True
        except Exception:
            break

    return pd.DataFrame([{
        "ged":    float(best_ged) if got_any else float("nan"),
        "n_pred": G_pred.number_of_nodes(),
        "n_gt":   G_gt.number_of_nodes(),
        "e_pred": G_pred.number_of_edges(),
        "e_gt":   G_gt.number_of_edges(),
        "timed_out": timed_out and got_any,
    }])


# Bounds the number of *abandoned* GED search threads allowed to be running
# at once, for the thread-based fallback path only (see its own docstring
# note above for why this exists).
_MAX_OUTSTANDING_GED_THREADS = 4
_ged_thread_count = 0
_ged_thread_count_lock = _threading.Lock()


def _graph_edit_distance_thread_fallback(G_pred, G_gt, node_cost, edge_cost, timeout):
    """Daemon-thread-based GED, used only if process spawning is unavailable
    (see graph_edit_distance's docstring for why this is the fallback, not
    the primary path)."""
    global _ged_thread_count
    with _ged_thread_count_lock:
        if _ged_thread_count >= _MAX_OUTSTANDING_GED_THREADS:
            return pd.DataFrame([{
                "ged":    float("nan"),
                "n_pred": G_pred.number_of_nodes(),
                "n_gt":   G_gt.number_of_nodes(),
                "e_pred": G_pred.number_of_edges(),
                "e_gt":   G_gt.number_of_edges(),
                "timed_out": False,
            }])
        _ged_thread_count += 1

    import queue
    import threading
    import time

    result_q: "queue.Queue" = queue.Queue()
    SENTINEL_DONE = object()

    def _worker():
        try:
            _ged_mp_worker(result_q, G_pred, G_gt, node_cost, edge_cost)
            result_q.put(SENTINEL_DONE)
        finally:
            global _ged_thread_count
            with _ged_thread_count_lock:
                _ged_thread_count -= 1

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()

    best_ged = float("inf")
    got_any = False
    timed_out = False
    deadline = time.time() + timeout
    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            timed_out = True
            break
        try:
            item = result_q.get(timeout=remaining)
        except queue.Empty:
            timed_out = True
            break
        if item is SENTINEL_DONE:
            break
        best_ged = item
        got_any = True

    return pd.DataFrame([{
        "ged":    float(best_ged) if got_any else float("nan"),
        "n_pred": G_pred.number_of_nodes(),
        "n_gt":   G_gt.number_of_nodes(),
        "e_pred": G_pred.number_of_edges(),
        "e_gt":   G_gt.number_of_edges(),
        "timed_out": timed_out and got_any,
    }])


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Frobenius norm between proximity matrices
# ══════════════════════════════════════════════════════════════════════════════

def frobenius_norm(
    A_pred: np.ndarray,
    A_gt: np.ndarray,
) -> pd.DataFrame:
    """
    Frobenius distance between two proximity matrices.

    If the matrices are different sizes (different number of rooms),
    the smaller one is zero-padded to match.

    Returns
    -------
    DataFrame with columns [frobenius, normalized, size_pred, size_gt].
    ``normalized`` = frobenius / sqrt(max(n_pred, n_gt)^2) so the value
    is comparable across graphs of different sizes.
    """
    n_pred, n_gt = A_pred.shape[0], A_gt.shape[0]
    n = max(n_pred, n_gt)

    # zero-pad to same size
    P = np.zeros((n, n), dtype=np.float64)
    G = np.zeros((n, n), dtype=np.float64)
    P[:n_pred, :n_pred] = A_pred
    G[:n_gt, :n_gt] = A_gt

    frob = float(np.linalg.norm(P - G, "fro"))
    normalized = frob / n if n > 0 else 0.0

    return pd.DataFrame([{
        "frobenius":  round(frob, 6),
        "normalized": round(normalized, 6),
        "size_pred":  n_pred,
        "size_gt":    n_gt,
    }])


# ══════════════════════════════════════════════════════════════════════════════
# Smoke test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Smoke-testing evaluate.py ...\n")

    # --- 1. compute_miou ---
    np.random.seed(42)
    H, W = 64, 64
    gt_mask = np.random.randint(0, 5, (H, W), dtype=np.int64)
    pred_mask = gt_mask.copy()
    # flip ~20% of pixels
    flip = np.random.rand(H, W) < 0.2
    pred_mask[flip] = (pred_mask[flip] + 1) % 5

    df_iou = compute_miou(pred_mask, gt_mask, num_classes=16)
    print("=== Segmentation Metrics ===")
    print(df_iou.to_string(index=False))
    mean_iou = df_iou[df_iou.class_name == "mean"]["iou"].iloc[0]
    assert 0 < mean_iou < 1, f"mIoU out of range: {mean_iou}"
    print(f"\nmIoU = {mean_iou:.4f}")

    # --- 2. edge_metrics ---
    G_gt = nx.Graph()
    G_gt.add_node(0, class_name="Bedroom",  class_id=1)
    G_gt.add_node(1, class_name="Kitchen",  class_id=3)
    G_gt.add_node(2, class_name="Bathroom", class_id=2)
    G_gt.add_edge(0, 1, edge_type="door")
    G_gt.add_edge(0, 2, edge_type="shared-wall")
    G_gt.add_edge(1, 2, edge_type="arch")

    G_pred = nx.Graph()
    G_pred.add_node(0, class_name="Bedroom",  class_id=1)
    G_pred.add_node(1, class_name="Kitchen",  class_id=3)
    G_pred.add_node(2, class_name="Bathroom", class_id=2)
    G_pred.add_edge(0, 1, edge_type="door")       # correct
    G_pred.add_edge(0, 2, edge_type="arch")        # wrong type
    # missing edge 1-2

    df_edge = edge_metrics(G_pred, G_gt)
    print("\n=== Edge Metrics ===")
    print(df_edge.to_string(index=False))
    assert df_edge["tp"].iloc[0] == 2
    assert df_edge["fn"].iloc[0] == 1
    assert df_edge["matched_correct_type"].iloc[0] == 1

    # --- 3. graph_edit_distance ---
    df_ged = graph_edit_distance(G_pred, G_gt, timeout=5.0)
    print("\n=== Graph Edit Distance ===")
    print(df_ged.to_string(index=False))
    assert df_ged["ged"].iloc[0] >= 0

    # --- 4. frobenius_norm ---
    A_gt   = np.array([[0, 1.0, 0.3], [1.0, 0, 0.8], [0.3, 0.8, 0]])
    A_pred = np.array([[0, 1.0, 0.8], [1.0, 0, 0.0], [0.8, 0.0, 0]])
    df_frob = frobenius_norm(A_pred, A_gt)
    print("\n=== Frobenius Norm ===")
    print(df_frob.to_string(index=False))
    assert df_frob["frobenius"].iloc[0] > 0

    # --- size mismatch test ---
    A_small = np.array([[0, 1.0], [1.0, 0]])
    df_pad = frobenius_norm(A_small, A_gt)
    print("\n=== Frobenius (mismatched sizes 2x2 vs 3x3) ===")
    print(df_pad.to_string(index=False))

    print("\n\nAll assertions passed.")
