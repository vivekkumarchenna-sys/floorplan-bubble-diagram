"""
build_graph.py — Extract a room-adjacency graph from a segmentation mask
=========================================================================
Input  : (H, W) integer segmentation mask with 16 classes (0–15).
Output : NetworkX graph where nodes = room instances, edges = adjacencies.

Class convention (matches train.py / 16-class floor-plan segmentation):
    0  Background
    1  Bedroom
    2  Bathroom
    3  Kitchen
    4  Living
    5  Balcony
    6  Storage
    7  Stair
    8  Parking
    9  Pool
   10  Wall
   11  Door
   12  Window
   13  FrontDoor
   14  Column
   15  Other

Algorithm
---------
1. Extract room instances via connected-component labelling for classes 1–9.
2. Extract door instances via connected-component labelling for class 11.
3. Dilate each room mask by `dilation_px` and test pairwise overlap.
4. Classify each edge:
     - "door"        → door pixels present in the overlap zone (≥ door_min px)
     - "arch"        → no door, but opening width ≥ arch_min px
     - "shared-wall" → overlap exists but no significant opening
5. Return the graph with node / edge attributes.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import cv2
import networkx as nx
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Default class mapping
# ──────────────────────────────────────────────────────────────────────────────

CLASS_NAMES = {
    0:  "Background",
    1:  "Bedroom",
    2:  "Bathroom",
    3:  "Kitchen",
    4:  "Living",
    5:  "Balcony",
    6:  "Storage",
    7:  "Stair",
    8:  "Parking",
    9:  "Pool",
    10: "Wall",
    11: "Door",
    12: "Window",
    13: "FrontDoor",
}

ROOM_CLASSES  = {1, 2, 3, 4, 5, 6, 7, 8, 9}   # all room/space types
DOOR_CLASSES  = {11, 13}                         # Door, FrontDoor


# ──────────────────────────────────────────────────────────────────────────────
# Data containers
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RoomInstance:
    """A single connected room region in the segmentation mask."""
    instance_id : int           # unique id across all rooms
    class_id    : int           # semantic class (1–9)
    class_name  : str
    mask        : np.ndarray    # (H, W) bool — pixels belonging to this room
    area_px     : int           # number of pixels
    centroid    : tuple[float, float]   # (row, col) centroid


# ──────────────────────────────────────────────────────────────────────────────
# Core helpers
# ──────────────────────────────────────────────────────────────────────────────

def _extract_room_instances(
    seg_mask: np.ndarray,
    room_classes: set[int] = ROOM_CLASSES,
    min_area: int = 100,
) -> list[RoomInstance]:
    """
    Find connected components for each room class, discarding tiny fragments.

    Parameters
    ----------
    seg_mask     : (H, W) uint8 or int32 segmentation mask.
    room_classes : set of class ids considered as rooms.
    min_area     : minimum pixel count to keep a component.

    Returns
    -------
    List of RoomInstance objects, sorted by instance_id.
    """
    instances: list[RoomInstance] = []
    inst_id = 0

    for cls in sorted(room_classes):
        cls_mask = (seg_mask == cls).astype(np.uint8)
        if cls_mask.sum() == 0:
            continue

        n_labels, labels = cv2.connectedComponents(cls_mask, connectivity=8)

        for lbl in range(1, n_labels):       # 0 = background
            comp_mask = labels == lbl
            area = int(comp_mask.sum())
            if area < min_area:
                continue

            ys, xs = np.where(comp_mask)
            centroid = (float(ys.mean()), float(xs.mean()))

            instances.append(RoomInstance(
                instance_id=inst_id,
                class_id=cls,
                class_name=CLASS_NAMES.get(cls, f"class_{cls}"),
                mask=comp_mask,
                area_px=area,
                centroid=centroid,
            ))
            inst_id += 1

    return instances


def _extract_door_mask(seg_mask: np.ndarray, door_classes: set[int] = DOOR_CLASSES) -> np.ndarray:
    """Return a boolean mask of all door pixels (Door + FrontDoor)."""
    mask = np.zeros_like(seg_mask, dtype=np.uint8)
    for cls in door_classes:
        mask[seg_mask == cls] = 1
    return mask


def _dilate(mask: np.ndarray, px: int) -> np.ndarray:
    """Dilate a binary mask by `px` pixels (square kernel)."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * px + 1, 2 * px + 1))
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)


def _overlap_width(overlap_mask: np.ndarray) -> int:
    """
    Estimate the width of an opening by finding the largest contour span
    in the overlap region.  Returns 0 if the mask is empty.
    """
    if overlap_mask.sum() == 0:
        return 0
    contours, _ = cv2.findContours(
        overlap_mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return 0
    # take the largest contour and measure its bounding-box minor axis
    largest = max(contours, key=cv2.contourArea)
    _, (w, h), _ = cv2.minAreaRect(largest)
    return int(min(w, h))       # minor axis ≈ "width" of the opening


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────

def build_graph_from_segmentation(
    seg_mask: np.ndarray,
    *,
    room_classes: set[int] = ROOM_CLASSES,
    door_classes: set[int] = DOOR_CLASSES,
    dilation_px: int = 15,
    door_min: int = 15,
    wall_min: int = 20,
    arch_min: int = 30,
    min_room_area: int = 100,
    pixel_scale: float | None = None,
) -> nx.Graph:
    """
    Build a room-adjacency graph from a semantic segmentation mask.

    Parameters
    ----------
    seg_mask      : np.ndarray, shape (H, W), dtype uint8 / int32.
                    Each pixel value is a class id.
    room_classes  : Set of class ids that represent rooms (default {1,2,3,4}).
    door_classes  : Set of class ids for doors (default {5, 8}).
    dilation_px   : Pixels to dilate each room mask when testing adjacency.
    door_min      : Minimum door-pixel overlap to label an edge "door".
    wall_min      : Minimum overlap (px) to consider two rooms adjacent.
    arch_min      : Minimum opening width (px) to label an edge "arch"
                    when no door is present.
    min_room_area : Connected components smaller than this are discarded.
    pixel_scale   : If provided (sq m per pixel), each node also gets an
                    ``area_sqm`` attribute = area_px × pixel_scale.

    Returns
    -------
    G : networkx.Graph
        Nodes carry attributes:
            instance_id, class_id, class_name, area_px, centroid
            area_sqm (only if pixel_scale is provided)
        Edges carry attributes:
            edge_type   : "door" | "arch" | "shared-wall"
            overlap_px  : number of overlapping pixels between dilated masks
            door_px     : number of door pixels in the overlap zone
            opening_width : estimated opening width in pixels

    Raises
    ------
    ValueError  If seg_mask is not 2-D.

    Example
    -------
    >>> import cv2, numpy as np
    >>> mask = cv2.imread("pred_mask.png", cv2.IMREAD_GRAYSCALE)
    >>> G = build_graph_from_segmentation(mask)
    >>> print(G.number_of_nodes(), "rooms,", G.number_of_edges(), "edges")
    >>> for u, v, d in G.edges(data=True):
    ...     print(f"  {G.nodes[u]['class_name']} ↔ {G.nodes[v]['class_name']}  "
    ...           f"[{d['edge_type']}]")
    """
    if seg_mask.ndim != 2:
        raise ValueError(f"seg_mask must be 2-D, got shape {seg_mask.shape}")

    seg_mask = seg_mask.astype(np.int32)

    # ── step 1 : extract room instances ──────────────────────────────────────
    rooms = _extract_room_instances(seg_mask, room_classes, min_area=min_room_area)

    # ── step 2 : extract door mask ───────────────────────────────────────────
    door_mask = _extract_door_mask(seg_mask, door_classes)

    # ── step 3 : build graph nodes ───────────────────────────────────────────
    G = nx.Graph()
    for r in rooms:
        attrs = dict(
            class_id=r.class_id,
            class_name=r.class_name,
            area_px=r.area_px,
            centroid=r.centroid,
        )
        if pixel_scale is not None:
            attrs["area_sqm"] = round(r.area_px * pixel_scale, 2)
        G.add_node(r.instance_id, **attrs)

    # ── step 4 : pre-compute dilated masks ───────────────────────────────────
    dilated = {r.instance_id: _dilate(r.mask, dilation_px) for r in rooms}

    # ── step 5 : pairwise adjacency + edge classification ────────────────────
    for a, b in itertools.combinations(rooms, 2):
        overlap = dilated[a.instance_id] & dilated[b.instance_id]
        overlap_px = int(overlap.sum())

        if overlap_px < wall_min:
            continue        # not adjacent

        # count door pixels in the overlap zone
        door_overlap = overlap & door_mask
        door_px = int(door_overlap.sum())

        # measure opening width (gap in wall between rooms)
        # opening = dilated overlap minus wall pixels
        wall_mask = (seg_mask == 10).astype(np.uint8)   # Wall class
        opening = overlap.copy()
        opening[wall_mask > 0] = 0      # remove wall pixels from overlap
        opening[a.mask > 0] = 0         # remove room-interior pixels
        opening[b.mask > 0] = 0
        opening_width = _overlap_width(opening)

        # classify edge
        if door_px >= door_min:
            edge_type = "door"
        elif opening_width >= arch_min:
            edge_type = "arch"
        else:
            edge_type = "shared-wall"

        G.add_edge(
            a.instance_id,
            b.instance_id,
            edge_type=edge_type,
            overlap_px=overlap_px,
            door_px=door_px,
            opening_width=opening_width,
        )

    return G


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def graph_summary(G: nx.Graph) -> str:
    """Return a human-readable summary of the adjacency graph."""
    lines = [f"Graph: {G.number_of_nodes()} rooms, {G.number_of_edges()} edges"]

    lines.append("\nNodes:")
    for nid, data in sorted(G.nodes(data=True)):
        area_str = f"area={data['area_px']:>6d}px"
        if "area_sqm" in data:
            area_str += f"  ({data['area_sqm']:.1f} m²)"
        lines.append(
            f"  [{nid:2d}] {data['class_name']:<12s}  "
            f"{area_str}  "
            f"centroid=({data['centroid'][0]:.0f}, {data['centroid'][1]:.0f})"
        )

    lines.append("\nEdges:")
    for u, v, data in G.edges(data=True):
        u_name = G.nodes[u]["class_name"]
        v_name = G.nodes[v]["class_name"]
        lines.append(
            f"  {u_name}[{u}] ↔ {v_name}[{v}]  "
            f"type={data['edge_type']:<12s}  "
            f"overlap={data['overlap_px']}px  "
            f"door={data['door_px']}px  "
            f"opening={data['opening_width']}px"
        )

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Smoke test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Smoke-testing build_graph_from_segmentation …\n")

    # create a synthetic 200×200 segmentation mask
    #
    #   ┌──────────┬─wall─┬──────────┐
    #   │ Bedroom  │ door │ Kitchen  │
    #   │ class=1  │  11  │ class=3  │
    #   │  (left)  │      │ (right)  │
    #   ├──────────┴──────┴──────────┤
    #   │         Living             │
    #   │         class=4            │
    #   └────────────────────────────┘

    H, W = 200, 200
    mask = np.zeros((H, W), dtype=np.uint8)

    # living room (bottom strip)
    mask[140:200, :] = 4

    # wall row
    mask[130:140, :] = 10

    # bedroom (top-left)
    mask[0:130, 0:90] = 1

    # kitchen (top-right)
    mask[0:130, 110:200] = 3

    # wall between bedroom and kitchen
    mask[0:130, 90:110] = 10

    # door between bedroom and kitchen (punch a hole in the wall)
    mask[50:70, 90:110] = 11

    # arch opening between living and bedroom (large opening, no door)
    mask[130:140, 20:60] = 0     # remove wall → open arch

    G = build_graph_from_segmentation(mask)
    print(graph_summary(G))

    # basic assertions
    assert G.number_of_nodes() == 3, f"Expected 3 rooms, got {G.number_of_nodes()}"
    assert G.number_of_edges() >= 2, f"Expected ≥2 edges, got {G.number_of_edges()}"

    # check bedroom ↔ kitchen is a door edge
    edge_types = {
        (G.nodes[u]["class_name"], G.nodes[v]["class_name"]): d["edge_type"]
        for u, v, d in G.edges(data=True)
    }
    assert ("Bedroom", "Kitchen") in edge_types or ("Kitchen", "Bedroom") in edge_types, \
        "Missing Bedroom ↔ Kitchen edge"

    bk_type = edge_types.get(("Bedroom", "Kitchen")) or edge_types.get(("Kitchen", "Bedroom"))
    assert bk_type == "door", f"Bedroom ↔ Kitchen should be 'door', got '{bk_type}'"

    print("\nAll assertions passed.")
