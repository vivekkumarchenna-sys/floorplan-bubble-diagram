"""
render_bubble.py - the paper's M2/M3/M4 as a reusable module.
=============================================================
Wraps ``truegraph_builder.build_true_graph`` (Module M2, Section 5.2) into a
NetworkX graph, exposes the categorical typed adjacency matrix (Module M3,
Section 5.3) and draws the geographic typed bubble diagram (Module M4,
Section 5.4): nodes at room centroids, marker area proportional to room floor
area, node colour sampled from the raster, and line style encoding the
connection type.

This is the rendering reported in the paper. ``visualize.draw_bubble_diagram``
is the earlier force-directed rendering with an ``arch`` edge class; it is kept
for the legacy dilation-based path (``build_graph.py``) and because the
user-study stimuli of Section 7.3 were produced with it.
"""
from __future__ import annotations

import re

import numpy as np
import networkx as nx
import matplotlib.lines as mlines

from build_graph import CLASS_NAMES
from truegraph_builder import build_true_graph, name_map

# Section 5.4: a door is a thin solid line, an open passage a thick solid line,
# a shared wall a dotted line - continuity signals that one can pass, thickness
# separates an open passage from a door.
EDGE_STYLES = {
    "door":        {"color": "#111111", "width": 2.6, "linestyle": "solid"},
    "open":        {"color": "#8E24AA", "width": 5.0, "linestyle": "solid"},
    "shared-wall": {"color": "#9E9E9E", "width": 2.2, "linestyle": (0, (1, 3))},
}
EDGE_LABELS = {"door": "door", "open": "open passage", "shared-wall": "shared wall"}


def _display(name: str) -> str:
    """'Bedroom1' -> 'Bedroom 1'."""
    return re.sub(r"([A-Za-z])(\d)$", r"\1 \2", name)


def build_typed_graph(mask, raster_rgb=None, pixel_scale=None) -> nx.Graph:
    """
    Module M2 (Section 5.2) as a NetworkX graph.

    mask         : 512x512 int label map, 16 classes.
    raster_rgb   : optional RGB raster, used only to sample each room's own
                   colour so a room reads the same in the plan and the diagram.
    pixel_scale  : optional m^2 per pixel for this plan (Section 5.5). Room
                   areas play no part in graph construction.

    Node attributes: class_id, class_name, name, centroid (x, -y in image
    coordinates, so the diagram mirrors the drawing), area_px, area_sqm
    (only when pixel_scale is given) and colour (only when raster_rgb is given).
    Edge attribute: edge_type in {door, open, shared-wall}.
    """
    rooms, edges = build_true_graph(mask)
    names = name_map(rooms, mask)

    G = nx.Graph()
    for i, (cls, cm) in rooms.items():
        ys, xs = np.where(cm)
        px = int(cm.sum())
        attrs = dict(
            class_id=cls,
            class_name=CLASS_NAMES[cls],
            name=_display(names[i]),
            centroid=(float(xs.mean()), -float(ys.mean())),
            area_px=px,
        )
        if pixel_scale:
            attrs["area_sqm"] = round(px * pixel_scale, 2)
        if raster_rgb is not None:
            px_vals = raster_rgb[cm]
            vals, counts = np.unique(px_vals.reshape(-1, 3), axis=0, return_counts=True)
            r, g, b = vals[counts.argmax()]
            attrs["colour"] = (r / 255.0, g / 255.0, b / 255.0)
        G.add_node(i, **attrs)

    for (a, b), etype in edges.items():
        G.add_edge(a, b, edge_type=etype)
    return G


def typed_adjacency(G: nx.Graph):
    """
    Module M3 (Section 5.3): the same graph as a categorical adjacency matrix.

    Returns (matrix, labels). Each off-diagonal entry is 'door', 'open',
    'shared-wall' or '' (not connected); the diagonal is ''. There are no edge
    weights - the graph is purely categorical.
    """
    order = ["Living", "Kitchen", "Balcony", "Bathroom", "Bedroom"]

    def rank(n):
        nm = G.nodes[n]["name"]
        return (next((k for k, o in enumerate(order) if nm.startswith(o)), 99), nm)

    ids = sorted(G.nodes, key=rank)
    labels = [G.nodes[i]["name"] for i in ids]
    n = len(ids)
    matrix = np.full((n, n), "", dtype=object)
    for r in range(n):
        for c in range(n):
            if r == c:
                continue
            d = G.get_edge_data(ids[r], ids[c])
            if d:
                matrix[r, c] = d["edge_type"]
    return matrix, labels


def format_adjacency(matrix, labels) -> str:
    """Printable form of the categorical adjacency matrix."""
    if not labels:
        return "(no rooms)"
    short = {"door": "D", "open": "OP", "shared-wall": "SW", "": "."}
    w = max(len(l) for l in labels)
    head = " " * (w + 2) + " ".join(f"{l[:4]:>4s}" for l in labels)
    rows = [head]
    for i, l in enumerate(labels):
        cells = " ".join(f"{short.get(matrix[i][j], '?'):>4s}" for j in range(len(labels)))
        rows.append(f"{l:<{w}s}  {cells}")
    return "\n".join(rows)


def draw_typed_bubble(G: nx.Graph, ax, title: str = "Typed bubble diagram",
                      show_legend: bool = True, font_size: float = 8.0):
    """
    Module M4 (Section 5.4): the geographic typed bubble diagram.

    Nodes sit at the room centroids recovered from the plan, so the diagram
    mirrors the spatial arrangement of the drawing; no force-directed layout
    and therefore no random seed is involved.
    """
    if G.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "No rooms detected", ha="center", va="center",
                transform=ax.transAxes, fontsize=12)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")
        return ax

    pos = {i: G.nodes[i]["centroid"] for i in G}
    metric = all("area_sqm" in G.nodes[i] for i in G)
    area = {i: (G.nodes[i]["area_sqm"] if metric else G.nodes[i]["area_px"]) for i in G}
    amax = max(area.values()) or 1

    for a, b, d in G.edges(data=True):
        st = EDGE_STYLES.get(d.get("edge_type"), EDGE_STYLES["shared-wall"])
        ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]],
                color=st["color"], lw=st["width"], ls=st["linestyle"], zorder=1)

    cx = np.mean([p[0] for p in pos.values()])
    cy = np.mean([p[1] for p in pos.values()])
    for i in G:
        s = 300 + 2200 * (area[i] / amax)
        colour = G.nodes[i].get("colour", "#90CAF9")
        ax.scatter([pos[i][0]], [pos[i][1]], s=s, c=[colour],
                   edgecolors="#333", linewidths=1.5, zorder=3)
        label = G.nodes[i]["name"]
        if metric:
            label += f"\n{area[i]:.1f} m²"
        dx, dy = pos[i][0] - cx, pos[i][1] - cy
        norm = (dx * dx + dy * dy) ** 0.5 or 1
        ux, uy = dx / norm, dy / norm
        off = (s / np.pi) ** 0.5 + 8
        ha = "left" if ux > 0.35 else ("right" if ux < -0.35 else "center")
        va = "bottom" if uy > 0.35 else ("top" if uy < -0.35 else "center")
        ax.annotate(label, pos[i], xytext=(off * ux, off * uy),
                    textcoords="offset points", ha=ha, va=va,
                    fontsize=font_size, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#bbb",
                              lw=0.5, alpha=0.92), zorder=5)

    if show_legend:
        handles = [mlines.Line2D([], [], color=st["color"], lw=st["width"],
                                 ls=st["linestyle"], label=EDGE_LABELS[et])
                   for et, st in EDGE_STYLES.items()]
        ax.legend(handles=handles, loc="lower center", ncol=3, fontsize=9,
                  frameon=False, bbox_to_anchor=(0.5, -0.10))

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axis("off")
    ax.margins(0.28)
    return ax
