"""
proximity.py - Weighted adjacency (proximity) matrix from a room graph
=======================================================================
Converts a NetworkX graph (from build_graph.py or build_gt_graph.py)
into a NumPy weighted adjacency matrix.

Edge weights:
    door        → 1.0   (strongest connection - direct passage)
    arch        → 0.8   (open passage, no physical door)
    window      → 0.5   (real opening in the wall, not walkable - ResPlan's
                          via_window, see build_gt_graph.RESPLAN_EDGE_TYPE_MAP.
                          Provisional weight, halfway between arch and
                          shared-wall; confirm with the ResPlan authors)
    shared-wall → 0.3   (adjacent but no passage)

Colab usage:
    import sys
    sys.path.insert(0, "/content/drive/MyDrive/bubble_diagram_project")

    from build_graph import build_graph_from_segmentation
    from proximity import compute_proximity_matrix, print_proximity_matrix

    G = build_graph_from_segmentation(seg_mask)
    matrix, labels = compute_proximity_matrix(G)
    print(print_proximity_matrix(matrix, labels))
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


# Default edge-type weights
DEFAULT_WEIGHTS: dict[str, float] = {
    "door":        1.0,
    "arch":        0.8,
    "window":      0.5,
    "shared-wall": 0.3,
}


def compute_proximity_matrix(
    G: nx.Graph,
    weights: dict[str, float] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """
    Build a weighted adjacency matrix from a room-adjacency graph.

    Parameters
    ----------
    G       : NetworkX graph produced by ``build_graph_from_segmentation``
              (predicted side) or ``build_gt_graph_from_resplan`` (ground
              truth - see ``build_gt_graph.py``; ``build_gt_graph_from_polygons``
              is a legacy reconstruction kept only as a second model for
              comparison, not as ground truth). Each edge must have an
              ``edge_type`` attribute ("door", "arch", "shared-wall" or
              "window"). Each node must have a ``class_name`` attribute.
    weights : Optional mapping from edge_type → float weight.
              Defaults to {"door": 1.0, "arch": 0.8, "shared-wall": 0.3}.

    Returns
    -------
    matrix : np.ndarray, shape (N, N), dtype float64.
             Symmetric weighted adjacency matrix.  Diagonal is 0.
             matrix[i][j] = weight of edge between room i and room j,
             or 0.0 if no edge exists.
    labels : list[str], length N.
             Human-readable label for each row/column, formatted as
             "ClassName[id]" (e.g. "Bedroom[0]", "Kitchen[1]").

    Example
    -------
    >>> matrix, labels = compute_proximity_matrix(G)
    >>> print(labels)
    ['Bedroom[0]', 'Kitchen[1]', 'Corridor[2]']
    >>> print(matrix)
    [[0.  1.  0.3]
     [1.  0.  0.3]
     [0.3 0.3 0. ]]
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    # sorted node ids for deterministic ordering
    node_ids = sorted(G.nodes())
    n = len(node_ids)
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    # build labels
    labels = [
        f"{G.nodes[nid]['class_name']}[{nid}]"
        for nid in node_ids
    ]

    # build matrix
    matrix = np.zeros((n, n), dtype=np.float64)

    for u, v, data in G.edges(data=True):
        edge_type = data.get("edge_type", "shared-wall")
        w = weights.get(edge_type, 0.0)
        i, j = id_to_idx[u], id_to_idx[v]
        matrix[i, j] = w
        matrix[j, i] = w       # symmetric

    return matrix, labels


def print_proximity_matrix(
    matrix: np.ndarray,
    labels: list[str],
) -> str:
    """
    Format the proximity matrix as a readable table string.

    Parameters
    ----------
    matrix : (N, N) weighted adjacency matrix.
    labels : length-N list of row/column labels.

    Returns
    -------
    Formatted string ready for print().
    """
    n = len(labels)
    # column width = max label length + padding
    col_w = max(len(l) for l in labels) + 2

    # header row
    header = " " * col_w + "".join(f"{l:>{col_w}}" for l in labels)
    lines = [header, "-" * len(header)]

    # data rows
    for i in range(n):
        row_vals = "".join(
            f"{matrix[i, j]:>{col_w}.1f}" for j in range(n)
        )
        lines.append(f"{labels[i]:<{col_w}}{row_vals}")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Smoke test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Smoke-testing compute_proximity_matrix …\n")

    # build a small graph manually
    G = nx.Graph()
    G.add_node(0, class_name="Bedroom",  class_id=2)
    G.add_node(1, class_name="Kitchen",  class_id=3)
    G.add_node(2, class_name="Corridor", class_id=6)

    G.add_edge(0, 1, edge_type="door")
    G.add_edge(0, 2, edge_type="arch")
    G.add_edge(1, 2, edge_type="shared-wall")

    matrix, labels = compute_proximity_matrix(G)

    print("Labels:", labels)
    print(f"Matrix shape: {matrix.shape}\n")
    print(print_proximity_matrix(matrix, labels))

    # assertions
    assert matrix.shape == (3, 3)
    assert matrix[0, 1] == 1.0,  f"door weight should be 1.0, got {matrix[0, 1]}"
    assert matrix[0, 2] == 0.8,  f"arch weight should be 0.8, got {matrix[0, 2]}"
    assert matrix[1, 2] == 0.3,  f"shared-wall weight should be 0.3, got {matrix[1, 2]}"
    assert (matrix == matrix.T).all(), "Matrix should be symmetric"
    assert (np.diag(matrix) == 0).all(), "Diagonal should be zero"

    # test custom weights
    custom = {"door": 2.0, "arch": 1.5, "shared-wall": 0.5}
    m2, _ = compute_proximity_matrix(G, weights=custom)
    assert m2[0, 1] == 2.0
    assert m2[0, 2] == 1.5
    assert m2[1, 2] == 0.5

    print("\n\nAll assertions passed.")
