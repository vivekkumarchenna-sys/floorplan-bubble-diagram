"""
visualize.py - Publication-quality bubble diagram from a room-adjacency graph
==============================================================================
Draws a Kamada-Kawai layout (falling back to Fruchterman-Reingold spring
layout only for disconnected graphs) where:
    - Node size  ∝  room area
    - Node color =  room type
    - Edge style =  connection type (solid / dashed / dotted / dash-dot)

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
    draw_bubble_diagram(G, ax, title="Floor Plan - Bubble Diagram")
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


# Node marker area (matplotlib points²) per square metre of floor area.
# Constant across diagrams so bubble sizes are directly comparable between
# plans. Only applied when nodes carry ``area_sqm`` (see build_pixel_scale.py).
# At 58, a 3 m² bathroom draws at 174 pt², clear of the min_node_size floor.
# Raised from 40 -> 58 (~45%) so bubbles read clearly even when a diagram is
# embedded small (e.g. one panel of Fig. 1's four-panel composition).
SQM_TO_POINTS: float = 58.0

# Ceiling for diagrams drawn without a known scale, where node size is
# normalised within the plan rather than in absolute units. Set near the
# dataset median largest room (~34 m²) so that unscaled diagrams read at
# typical size instead of saturating max_node_size and towering over
# scaled ones placed beside them. Raised in step with SQM_TO_POINTS above.
FALLBACK_MAX_POINTS: float = 2000.0


# ──────────────────────────────────────────────────────────────────────────────
# Color palette - one color per room class
# ──────────────────────────────────────────────────────────────────────────────
#
# ROOM_COLORS MUST equal inference.py's / generate_bubble.py's ``_PALETTE``
# (the values used to render the segmentation-mask overlay), so a room drawn as
# e.g. green in the mask panel of a figure is drawn as the same green bubble in
# the adjoining bubble-diagram panel. It is derived directly from that canonical
# BGR palette (BGR -> RGB -> hex) rather than being restated by hand, so it
# cannot drift from it. (An older version of this dict was an unrelated,
# hand-chosen Material-Design palette that matched _PALETTE at zero of the nine
# room classes; deriving it removes any chance of that recurring.)
def _room_colors_from_palette() -> dict[str, str]:
    """Build the room-class -> hex-colour map from inference.py's canonical BGR
    _PALETTE. Room classes are ids 1-9 (see build_graph.CLASS_NAMES)."""
    from build_graph import CLASS_NAMES
    from inference import _PALETTE as _BGR_PALETTE  # (NUM_CLASSES, 3) uint8, BGR
    colors: dict[str, str] = {}
    for cid in range(1, 10):
        b, g, r = (int(c) for c in _BGR_PALETTE[cid])
        colors[CLASS_NAMES[cid]] = "#{:02X}{:02X}{:02X}".format(r, g, b)
    return colors


ROOM_COLORS: dict[str, str] = _room_colors_from_palette()

# Short forms used for on-diagram labels. The full room name is carried by the
# legend and by the matching colour in the adjoining label-map panel, so the
# diagram itself only needs enough text to stay identifiable in greyscale
# print, where colour alone would not encode room type at all. Two-line
# "Bedroom\n12.4 m²" labels were roughly twice this width and collided with
# neighbouring bubbles in dense plans.
ROOM_ABBR: dict[str, str] = {
    "Bedroom": "Bed", "Bathroom": "Bath", "Kitchen": "Kit",
    "Living": "Liv", "Balcony": "Balc", "Storage": "Sto",
    "Stair": "Stair", "Parking": "Park", "Pool": "Pool",
}

_FALLBACK_COLOR ="#B0BEC5"     # light blue-grey for unknown classes


# ──────────────────────────────────────────────────────────────────────────────
# Edge style mapping
# ──────────────────────────────────────────────────────────────────────────────
# One visually distinct (style, color) pair per edge type actually produced by
# either build_gt_graph_from_resplan (door/arch/shared-wall/window) or
# build_graph_from_segmentation (door/arch/shared-wall - M2 does not predict
# "window"). "window" previously had no entry here and silently fell into the
# generic _FALLBACK_EDGE style shared with genuinely-unrecognised edge types -
# indistinguishable from an error case in any rendered figure that includes a
# ground-truth graph with via_window edges (e.g. a corrected Fig. 5/6 GT panel).

EDGE_STYLES: dict[str, dict] = {
    # dashes are (on, off, ...) in points, absolute - not scaled by width.
    "door":        {"color": "#111111", "width": 2.8, "dashes": None},
    "arch":        {"color": "#3F3F3F", "width": 2.3, "dashes": (13, 5)},
    "shared-wall": {"color": "#6E6E6E", "width": 2.6, "dashes": (0.1, 5.0)},
    "window":      {"color": "#9A9A9A", "width": 1.8, "dashes": (9, 3, 1.6, 3)},
}

def _style_line(et):
    """Matplotlib kwargs for one edge type, shared by plot and legend."""
    st = EDGE_STYLES[et]
    kw = dict(color=st["color"], linewidth=st["width"], solid_capstyle="round",
              dash_capstyle="round")
    kw["linestyle"] = "solid" if st["dashes"] is None else (0, st["dashes"])
    return kw

_FALLBACK_EDGE = {"style": "solid", "color": "#9E9E9E", "width": 1.0}


# ──────────────────────────────────────────────────────────────────────────────
# Main drawing function
# ──────────────────────────────────────────────────────────────────────────────

def _layout_graph(G: nx.Graph, seed: int = 42) -> dict:
    """Position nodes for drawing.

    Connected graphs use Kamada-Kawai directly. Disconnected graphs previously
    fell back to a single spring_layout over the whole graph, which packed the
    largest component into a tight knot (its off-node labels then overprinting
    one another) while flinging isolated rooms to the far corners. Instead lay
    out each connected component on its own with the same Kamada-Kawai spacing
    the connected case gets, normalise each to a unit box, then pack the
    components on a grid so none collapses and none overlaps another.
    """
    if G.number_of_nodes() == 1:
        return {next(iter(G.nodes())): np.array([0.0, 0.0])}
    if nx.is_connected(G):
        return nx.kamada_kawai_layout(G)

    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    cols = int(np.ceil(np.sqrt(len(comps))))
    # Each component is scaled so its half-extent grows with sqrt(node count):
    # a large hub-and-spoke component is given proportionally more room than a
    # two-node stub, instead of both being squeezed into the same unit box.
    half = {i: 0.55 * np.sqrt(len(c)) for i, c in enumerate(comps)}
    spacing = 2.0 * max(half.values()) + 1.6
    pos: dict = {}
    for i, comp in enumerate(comps):
        sub = G.subgraph(comp)
        nodes = list(comp)
        if len(nodes) == 1:
            sub_pos = {nodes[0]: np.array([0.0, 0.0])}
        elif len(nodes) > 2:
            sub_pos = nx.kamada_kawai_layout(sub)
        else:
            sub_pos = nx.spring_layout(sub, seed=seed, k=1.5, iterations=200)
        xs = np.array([sub_pos[n][0] for n in nodes], dtype=float)
        ys = np.array([sub_pos[n][1] for n in nodes], dtype=float)
        rx, ry = xs.max() - xs.min(), ys.max() - ys.min()
        rng = max(rx, ry, 1e-9)
        h = half[i]
        xs = (xs - xs.mean()) / rng * (2 * h)
        ys = (ys - ys.mean()) / rng * (2 * h)
        r, c = divmod(i, cols)
        ox, oy = c * spacing, -r * spacing
        for k, nid in enumerate(nodes):
            pos[nid] = np.array([xs[k] + ox, ys[k] + oy])
    return pos


def draw_bubble_diagram(
    G: nx.Graph,
    ax: plt.Axes | None = None,
    title: str = "Bubble Diagram",
    *,
    min_node_size: float = 70,
    max_node_size: float = 8000,
    font_size: int = 9,
    seed: int = 42,
    show_legend: bool = True,
) -> plt.Axes:
    """
    Draw a publication-quality bubble diagram from a room-adjacency graph.

    Parameters
    ----------
    G              : NetworkX graph from ``build_graph_from_segmentation``
                     (predicted side) or ``build_gt_graph_from_resplan``
                     (ground truth - ``build_gt_graph_from_polygons`` is a
                     legacy reconstruction, no longer used as ground truth,
                     see ``build_gt_graph.py``).
    ax             : Matplotlib Axes. Created automatically if None.
    title          : Plot title.
    min_node_size  : Lower clamp on node marker size. Kept small so that
                     absolute area scaling is not flattened for tiny rooms.
    max_node_size  : Upper clamp on node marker size, applied when nodes
                     carry ``area_sqm``. The default clears the 99th
                     percentile room in ResPlan (~137 m²); larger rooms are
                     clipped and are usually segmentation merges. Diagrams
                     without a scale use FALLBACK_MAX_POINTS instead.
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
    # Kamada-Kawai gives a markedly cleaner, less crossing-prone layout than
    # spring_layout for the small, dense graphs typical here (ResPlan plans
    # average ~8 rooms) - it directly minimises a stress function over all
    # pairwise graph distances rather than simulating springs, so it doesn't
    # depend on a random seed or need k/iteration tuning per graph. It
    # requires a connected graph, though (undefined path distance otherwise),
    # so isolated components fall back to spring_layout with wide spacing.
    pos = _layout_graph(G, seed=seed)


    # ── node sizes (proportional to area) ────────────────────────────────────
    # When area_sqm is available, size is absolute (points² per m²) so that
    # bubbles are comparable across diagrams. Otherwise size is proportional to
    # pixel area normalised within the plan, which keeps ratios honest but is
    # not comparable with other diagrams.
    areas_sqm = [G.nodes[nid].get("area_sqm") for nid in G.nodes()]
    if areas_sqm and all(a is not None for a in areas_sqm):
        node_sizes = np.clip(np.array(areas_sqm, dtype=float) * SQM_TO_POINTS,
                             min_node_size, max_node_size)
    else:
        # works with both build_graph (area_px) and build_gt_graph (area)
        areas = np.array(
            [G.nodes[nid].get("area_px", G.nodes[nid].get("area", 500))
             for nid in G.nodes()], dtype=float)
        # Scale relative to the largest room so marker area stays proportional
        # to floor area. Min-max normalisation would instead pin the smallest
        # room to the floor and the largest to the ceiling, exaggerating their
        # ratio whenever the rooms are close in size. Pixel areas are not
        # comparable between plans, so normalising per plan loses nothing.
        peak = areas.max()
        if peak > 0:
            node_sizes = np.clip(areas / peak * FALLBACK_MAX_POINTS,
                                 min_node_size, FALLBACK_MAX_POINTS)
        else:
            node_sizes = np.full_like(areas, min_node_size)

    # Bubble area is set in absolute points^2 per m^2 so diagrams are
    # comparable to each other, but the layout box does not grow with room
    # count: a 14-room plan therefore packs the same canvas as a 5-room one
    # and its bubbles crowd out their own off-node labels. Damp the absolute
    # scale as the graph grows. Ratios within a plan are untouched, so the
    # area encoding stays honest; only the plan-to-plan scale is compressed,
    # which is the trade that keeps dense plans legible.
    _n = len(node_sizes)
    if _n > 7:
        node_sizes = np.clip(node_sizes * (7.0 / _n) ** 0.55,
                             min_node_size, None)

    # ── node colors ──────────────────────────────────────────────────────────
    node_colors = [
        ROOM_COLORS.get(G.nodes[nid]["class_name"], _FALLBACK_COLOR)
        for nid in G.nodes()
    ]

    # ── node labels ──────────────────────────────────────────────────────────
    node_labels = {}
    for nid in G.nodes():
        name = G.nodes[nid]["class_name"]
        short = ROOM_ABBR.get(name, name)
        if "area_sqm" in G.nodes[nid]:
            node_labels[nid] = f"{short} {G.nodes[nid]['area_sqm']:.1f} m²"
        else:
            area = G.nodes[nid].get("area_px", G.nodes[nid].get("area", 0))
            node_labels[nid] = f"{short} {area:.0f}px²"

    # ── draw edges (grouped by type for consistent style) ────────────────────
    # Dash/dot patterns alone (matplotlib's built-in "dashed"/"dotted") read as
    # near-identical at small figure sizes or after downsampling, especially
    # once lines get short in a dense layout - the previous version relied on
    # linestyle name alone and was hard to tell apart in practice. Explicit
    # ``dashes`` tuples (on/off segment lengths in points, scaled by width)
    # plus a wider spread of both width and greyscale value make the four
    # types distinguishable even in a thumbnail or greyscale print.
    # Segment lengths are absolute minimums (in points, before the
    # width-scaling below), not just proportional to line width: at small
    # embed sizes (e.g. one panel of a four-panel composite figure) a 1pt
    # dot on a 1.5pt-wide line all but disappears and the line reads as
    # solid. Every "on" segment here is kept large enough to survive
    # shrinking to roughly a quarter of this function's own reference size.
    for edge_type, style in EDGE_STYLES.items():
        edge_list = [
            (u, v) for u, v, d in G.edges(data=True)
            if d.get("edge_type") == edge_type
        ]
        if edge_list:
            artists = nx.draw_networkx_edges(
                G, pos, edgelist=edge_list, ax=ax,
                edge_color=style["color"],
                width=style["width"],
                alpha=0.95,
                node_size=node_sizes,   # so edges stop at the bubble boundary, not its centre
            )
            dashes = style["dashes"]
            if dashes is not None and artists is not None:
                for a in (artists if isinstance(artists, list) else [artists]):
                    a.set_linestyle((0, dashes))
                    a.set_capstyle("round")

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
            node_size=node_sizes,
        )

    # ── draw nodes ───────────────────────────────────────────────────────────
    # A faint offset "shadow" duplicate reads as subtle depth without needing
    # a real renderer - cheap, robust across matplotlib backends/DPIs.
    shadow_pos = {n: (x + 0.012, y - 0.012) for n, (x, y) in pos.items()}
    nx.draw_networkx_nodes(
        G, shadow_pos, ax=ax,
        node_size=node_sizes,
        node_color="#000000",
        alpha=0.10,
        linewidths=0,
    )
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_size=node_sizes,
        node_color=node_colors,
        edgecolors="#333333",
        linewidths=1.5,
        alpha=0.95,
    )

    # ── draw labels ──────────────────────────────────────────────────────────
    # Placed just below each bubble rather than on top of it: on-node text
    # overflows small rooms (a "Bathroom, 3.5 m²" label is wider than a small
    # bathroom's bubble) and collides with the border/adjacent edges. An
    # off-node label with its own translucent rounded backdrop stays legible
    # regardless of what's behind it (another bubble, a bundle of edges) and
    # scales with a node's own on-screen radius rather than one fixed size,
    # so small and large rooms both read cleanly.
    node_list = list(G.nodes())
    sizes_by_node = dict(zip(node_list, node_sizes))
    # matplotlib node_size is in points^2; approximate on-screen radius in
    # the axes' data units so the label offset scales with the bubble.
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1e-6)
    cx = float(np.mean([p[0] for p in pos.values()]))
    cy = float(np.mean([p[1] for p in pos.values()]))
    for nid in node_list:
        radius_pt = float(np.sqrt(sizes_by_node[nid] / np.pi))
        x, y = pos[nid]
        # Push the label radially outward from the graph's centroid instead of
        # always straight down. In a hub-and-spoke plan (one large living area
        # ringed by small rooms) a downward label on a small outer room lands
        # on top of the hub bubble; outward is the one direction guaranteed to
        # lead away from the crowded middle.
        dx, dy = x - cx, y - cy
        norm = float(np.hypot(dx, dy))
        if norm < 1e-9:
            ux, uy = 0.0, -1.0          # the hub itself: keep it below
        else:
            ux, uy = dx / norm, dy / norm
        gap = radius_pt + 7.0
        _n = G.number_of_nodes()
        label_size = (max(font_size - 3, 6) if _n > 11
                      else max(font_size - 2, 6) if _n > 8
                      else font_size)
        ax.annotate(
            node_labels[nid],
            xy=(x, y), xytext=(ux * gap, uy * gap),
            textcoords="offset points",
            ha="left" if ux > 0.3 else ("right" if ux < -0.3 else "center"),
            va="bottom" if uy > 0.3 else ("top" if uy < -0.3 else "center"),
            fontsize=label_size, fontweight="bold", color="#212121",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="none", alpha=0.75),
            zorder=5,
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
                    mlines.Line2D([], [], label=edge_type,
                                  **_style_line(edge_type))
                )

        # place the legend below the diagram (horizontal strip) so it never
        # overlaps the bubbles. The empty spacer patch is dropped here since a
        # blank column reads oddly in a horizontal layout. Anchored well clear
        # of y=0 (axes-fraction) rather than just below it: node labels are
        # offset below their bubble in points, not data units, and matplotlib's
        # autoscale does not reliably reserve room for that offset, so a node
        # near the bottom of the layout can otherwise collide with the legend.
        row_handles = [h for h in legend_handles if h.get_label()]
        ax.legend(
            handles=row_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.16),
            ncol=min(len(row_handles), 6),
            framealpha=0.9,
            fontsize=9,
            title="Room types & edges",
            title_fontsize=10,
        )

    # ── styling ──────────────────────────────────────────────────────────────
    # Generous margins (rather than the default ~5%) so that off-node labels,
    # whose offset is set in points rather than data units, never clip against
    # the axes edge or the legend below - matplotlib's autoscale accounts for
    # the annotation anchor point but not reliably for the offset text itself.
    # Asymmetric: labels are only ever offset *below* a node, so the extra
    # headroom only needs to go on the bottom, not wasted at the top too.
    ax.margins(x=0.15, y=0.12)
    y0, y1 = ax.get_ylim()
    ax.set_ylim(y0 - 0.30 * (y1 - y0), y1)
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
    draw_bubble_diagram(G, ax, title="Smoke Test - Bubble Diagram")
    fig.tight_layout()
    fig.savefig("bubble_diagram_test.png", dpi=150, bbox_inches="tight")
    print("Saved -> bubble_diagram_test.png")
    plt.close(fig)

    print("Done.")
