"""
visualize.py — Publication-quality bubble diagram from a room-adjacency graph
==============================================================================
Draws a Fruchterman–Reingold layout where:
    - Node size  ∝  room area
    - Node color =  room type
    - Edge style =  connection type (solid / dashed / dotted)

Colab usage:
    import sys
    sys.path.insert(0, "/content/drive/MyDrive/bubble_diagram_project")

    import numpy as np
    from PIL import Image
    from build_graph import build_graph_from_segmentation
    from visualize import draw_bubble_diagram
    import matplotlib.pyplot as plt

    mask = np.array(Image.open(".../0_mask.png"))
    G = build_graph_from_segmentation(mask)
    fig, ax = plt.subplots(figsize=(10, 8))
    draw_bubble_diagram(G, ax, title="Floor Plan — Bubble Diagram")
    plt.show()
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
except NameError:
    sys.path.insert(0, "/content/drive/MyDrive/bubble_diagram_project")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import networkx as nx
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Color palette — one color per room class
# ──────────────────────────────────────────────────────────────────────────────

ROOM_COLORS: dict[str, str] = {
    "Bedroom":   "#84C781",    # green     — matches raster color
    "Bathroom":  "#64B5F6",    # blue
    "Kitchen":   "#FF8A65",    # orange
    "Living":    "#FFD54F",    # yellow/amber
    "Balcony":   "#AED581",    # light green
    "Storage":   "#E0E0E0",    # grey
    "Stair":     "#CE93D8",    # purple
    "Parking":   "#90A4AE",    # blue-grey
    "Pool":      "#4DD0E1",    # cyan
}

_FALLBACK_COLOR = "#B0BEC5"     # light blue-grey for unknown classes


# ──────────────────────────────────────────────────────────────────────────────
# Edge style mapping
# ──────────────────────────────────────────────────────────────────────────────

EDGE_STYLES: dict[str, dict] = {
    "door":        {"style": "solid",  "color": "#212121", "width": 2.5},
    "arch":        {"style": "dashed", "color": "#616161", "width": 2.0},
    "shared-wall": {"style": "dotted", "color": "#BDBDBD", "width": 1.5},
}

_FALLBACK_EDGE = {"style": "solid", "color": "#9E9E9E", "width": 1.0}


# ──────────────────────────────────────────────────────────────────────────────
# Main drawing function
# ──────────────────────────────────────────────────────────────────────────────

def draw_bubble_diagram(
    G: nx.Graph,
    ax: plt.Axes | None = None,
    title: str = "Bubble Diagram",
    *,
    min_node_size: float = 300,
    max_node_size: float = 3000,
    font_size: int = 9,
    seed: int = 42,
    show_legend: bool = True,
) -> plt.Axes:
    """
    Draw a publication-quality bubble diagram from a room-adjacency graph.

    Parameters
    ----------
    G              : NetworkX graph from ``build_graph_from_segmentation``
                     or ``build_gt_graph_from_polygons``.
    ax             : Matplotlib Axes. Created automatically if None.
    title          : Plot title.
    min_node_size  : Minimum node marker size (smallest room).
    max_node_size  : Maximum node marker size (largest room).
    font_size      : Node label font size.
    seed           : Random seed for Fruchterman–Reingold layout.
    show_legend    : Whether to draw the legend.

    Returns
    -------
    ax : The Matplotlib Axes with the diagram drawn.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8))

    if G.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "Empty graph", ha="center", va="center",
                transform=ax.transAxes, fontsize=14)
        ax.set_title(title)
        return ax

    # ── layout ───────────────────────────────────────────────────────────────
    pos = nx.spring_layout(G, seed=seed, k=2.0 / max(np.sqrt(G.number_of_nodes()), 1))

    # ── node sizes (proportional to area) ────────────────────────────────────
    # works with both build_graph (area_px) and build_gt_graph (area)
    areas = []
    for nid in G.nodes():
        data = G.nodes[nid]
        areas.append(data.get("area_px", data.get("area", 500)))
    areas = np.array(areas, dtype=float)

    if areas.max() > areas.min():
        norm = (areas - areas.min()) / (areas.max() - areas.min())
    else:
        norm = np.full_like(areas, 0.5)
    node_sizes = min_node_size + norm * (max_node_size - min_node_size)

    # ── node colors ──────────────────────────────────────────────────────────
    node_colors = [
        ROOM_COLORS.get(G.nodes[nid]["class_name"], _FALLBACK_COLOR)
        for nid in G.nodes()
    ]

    # ── node labels ──────────────────────────────────────────────────────────
    node_labels = {}
    for nid in G.nodes():
        name = G.nodes[nid]["class_name"]
        area = G.nodes[nid].get("area_px", G.nodes[nid].get("area", 0))
        node_labels[nid] = f"{name}\n{area:.0f}px²"

    # ── draw edges (grouped by type for consistent style) ────────────────────
    for edge_type, style in EDGE_STYLES.items():
        edge_list = [
            (u, v) for u, v, d in G.edges(data=True)
            if d.get("edge_type") == edge_type
        ]
        if edge_list:
            nx.draw_networkx_edges(
                G, pos, edgelist=edge_list, ax=ax,
                style=style["style"],
                edge_color=style["color"],
                width=style["width"],
                alpha=0.9,
            )

    # draw any edges with unknown types
    known_types = set(EDGE_STYLES.keys())
    unknown_edges = [
        (u, v) for u, v, d in G.edges(data=True)
        if d.get("edge_type") not in known_types
    ]
    if unknown_edges:
        nx.draw_networkx_edges(
            G, pos, edgelist=unknown_edges, ax=ax,
            style=_FALLBACK_EDGE["style"],
            edge_color=_FALLBACK_EDGE["color"],
            width=_FALLBACK_EDGE["width"],
        )

    # ── draw nodes ───────────────────────────────────────────────────────────
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_size=node_sizes,
        node_color=node_colors,
        edgecolors="#333333",
        linewidths=1.5,
        alpha=0.92,
    )

    # ── draw labels ──────────────────────────────────────────────────────────
    nx.draw_networkx_labels(
        G, pos, labels=node_labels, ax=ax,
        font_size=font_size,
        font_weight="bold",
        font_color="#212121",
    )

    # ── legend ───────────────────────────────────────────────────────────────
    if show_legend:
        legend_handles = []

        # room type patches
        seen_classes = set()
        for nid in G.nodes():
            cname = G.nodes[nid]["class_name"]
            if cname not in seen_classes:
                seen_classes.add(cname)
                color = ROOM_COLORS.get(cname, _FALLBACK_COLOR)
                legend_handles.append(
                    mpatches.Patch(
                        facecolor=color, edgecolor="#333333",
                        linewidth=1.0, label=cname,
                    )
                )

        # separator
        legend_handles.append(mpatches.Patch(color="none", label=""))

        # edge type lines
        for edge_type, style in EDGE_STYLES.items():
            # only include edge types that exist in the graph
            if any(d.get("edge_type") == edge_type for _, _, d in G.edges(data=True)):
                legend_handles.append(
                    mlines.Line2D(
                        [], [],
                        color=style["color"],
                        linewidth=style["width"],
                        linestyle=style["style"],
                        label=edge_type,
                    )
                )

        ax.legend(
            handles=legend_handles,
            loc="upper left",
            framealpha=0.9,
            fontsize=9,
            title="Room types & edges",
            title_fontsize=10,
        )

    # ── styling ──────────────────────────────────────────────────────────────
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_facecolor("#FAFAFA")
    ax.axis("off")

    return ax


# ──────────────────────────────────────────────────────────────────────────────
# Smoke test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Smoke-testing draw_bubble_diagram …")

    G = nx.Graph()
    G.add_node(0, class_name="Bedroom",   class_id=1, area_px=12000)
    G.add_node(1, class_name="Kitchen",   class_id=3, area_px=8000)
    G.add_node(2, class_name="Living",    class_id=4, area_px=15000)
    G.add_node(3, class_name="Bathroom",  class_id=2, area_px=4000)

    G.add_edge(0, 1, edge_type="door")
    G.add_edge(0, 2, edge_type="arch")
    G.add_edge(1, 2, edge_type="shared-wall")
    G.add_edge(2, 3, edge_type="door")

    fig, ax = plt.subplots(figsize=(10, 8))
    draw_bubble_diagram(G, ax, title="Smoke Test — Bubble Diagram")
    fig.tight_layout()
    fig.savefig("bubble_diagram_test.png", dpi=150, bbox_inches="tight")
    print("Saved → bubble_diagram_test.png")
    plt.close(fig)

    print("Done.")
