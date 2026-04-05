"""
build_gt_graph.py — Ground-truth room-adjacency graph from polygon data
=========================================================================
Two entry points:

1. build_gt_graph_from_polygons(plan_dict)
       Takes a plan_dict with Shapely polygons → NetworkX graph.

2. mask_to_plan_dict(seg_mask)
       Converts a (H, W) segmentation mask to plan_dict format,
       so existing ResPlan masks can feed into the polygon pipeline.

plan_dict format
----------------
{
    "rooms": [
        {"id": 0, "class_id": 2, "class_name": "Bedroom",
         "polygon": <shapely.Polygon>},
        ...
    ],
    "doors": [
        {"id": 0, "class_id": 11, "polygon": <shapely.Polygon>},
        ...
    ],
}

Class convention (same as build_graph.py / train.py):
    0  Background     5  Balcony     10  Wall
    1  LivingRoom     6  Corridor    11  Door
    2  Bedroom        7  Dining      12  Window
    3  Kitchen        8  Storage     13  Staircase
    4  Bathroom       9  Garage      14  Column
                                     15  Other
"""

from __future__ import annotations

import itertools
from typing import Any

import cv2
import networkx as nx
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from shapely.validation import make_valid

# ──────────────────────────────────────────────────────────────────────────────
# Class mapping (shared with build_graph.py)
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
# Mask → plan_dict converter
# ──────────────────────────────────────────────────────────────────────────────

def _mask_class_to_polygons(
    seg_mask: np.ndarray,
    class_id: int,
    min_area: float = 100.0,
    simplify_tol: float = 2.0,
) -> list[Polygon]:
    """
    Extract Shapely polygons from all connected components of a given class.

    Parameters
    ----------
    seg_mask     : (H, W) int array — segmentation mask.
    class_id     : Class to extract.
    min_area     : Ignore components smaller than this (in px²).
    simplify_tol : Douglas-Peucker tolerance for polygon simplification.

    Returns
    -------
    List of valid Shapely Polygons (may be empty).
    """
    binary = (seg_mask == class_id).astype(np.uint8)
    if binary.sum() == 0:
        return []

    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return []

    polygons = []
    for i, cnt in enumerate(contours):
        # only process outer contours (parent == -1 in RETR_CCOMP)
        if hierarchy[0][i][3] != -1:
            continue
        if len(cnt) < 3:
            continue

        coords = cnt.squeeze()
        if coords.ndim != 2 or coords.shape[0] < 3:
            continue

        # build exterior ring; collect holes
        exterior = [(float(x), float(y)) for x, y in coords]
        holes = []
        child = hierarchy[0][i][2]
        while child != -1:
            hcnt = contours[child].squeeze()
            if hcnt.ndim == 2 and hcnt.shape[0] >= 3:
                holes.append([(float(x), float(y)) for x, y in hcnt])
            child = hierarchy[0][child][0]

        poly = Polygon(exterior, holes)
        poly = make_valid(poly)

        # handle MultiPolygon from make_valid
        if isinstance(poly, MultiPolygon):
            for p in poly.geoms:
                if p.area >= min_area:
                    polygons.append(p.simplify(simplify_tol))
        elif poly.area >= min_area:
            polygons.append(poly.simplify(simplify_tol))

    return polygons


def mask_to_plan_dict(
    seg_mask: np.ndarray,
    room_classes: set[int] = ROOM_CLASSES,
    door_classes: set[int] = DOOR_CLASSES,
    min_room_area: float = 100.0,
    min_door_area: float = 20.0,
    simplify_tol: float = 2.0,
) -> dict[str, Any]:
    """
    Convert a segmentation mask to a plan_dict with Shapely polygons.

    Parameters
    ----------
    seg_mask       : (H, W) integer array with class ids.
    room_classes   : Set of class ids representing rooms.
    door_classes   : Set of class ids for doors (Door + FrontDoor).
    min_room_area  : Minimum polygon area for rooms (px²).
    min_door_area  : Minimum polygon area for doors (px²).
    simplify_tol   : Douglas-Peucker simplification tolerance.

    Returns
    -------
    plan_dict : dict with keys "rooms" and "doors".
    """
    rooms = []
    room_id = 0
    for cls in sorted(room_classes):
        polys = _mask_class_to_polygons(
            seg_mask, cls, min_area=min_room_area, simplify_tol=simplify_tol,
        )
        for poly in polys:
            rooms.append({
                "id":         room_id,
                "class_id":   cls,
                "class_name": CLASS_NAMES.get(cls, f"class_{cls}"),
                "polygon":    poly,
            })
            room_id += 1

    doors = []
    door_id = 0
    for dcls in sorted(door_classes):
        door_polys = _mask_class_to_polygons(
            seg_mask, dcls, min_area=min_door_area, simplify_tol=simplify_tol,
        )
        for poly in door_polys:
            doors.append({
                "id":       door_id,
                "class_id": dcls,
                "polygon":  poly,
            })
            door_id += 1

    return {"rooms": rooms, "doors": doors}


# ──────────────────────────────────────────────────────────────────────────────
# Ground-truth graph builder (Shapely-based)
# ──────────────────────────────────────────────────────────────────────────────

def build_gt_graph_from_polygons(
    plan_dict: dict[str, Any],
    *,
    adjacency_buffer: float = 15.0,
    door_buffer: float = 5.0,
    door_overlap_min: float = 15.0,
    shared_wall_min: float = 20.0,
    arch_min_length: float = 30.0,
) -> nx.Graph:
    """
    Build a ground-truth room-adjacency graph from polygon annotations.

    Parameters
    ----------
    plan_dict         : dict with "rooms" and "doors" lists.
                        Each room has: id, class_id, class_name, polygon.
                        Each door has: id, class_id, polygon.
    adjacency_buffer  : Buffer (px) applied to each room polygon before
                        testing intersection.  Two rooms are adjacent if
                        their buffered polygons intersect.
    door_buffer       : Extra buffer applied to door polygons when testing
                        overlap with the room-pair boundary zone.
    door_overlap_min  : Minimum intersection area (px²) between a door
                        polygon and the room-pair boundary to classify
                        the edge as "door".
    shared_wall_min   : Minimum shared boundary length (px) for an edge
                        to exist at all.
    arch_min_length   : Minimum shared boundary length (px) to classify
                        an edge as "arch" when no door is present.

    Returns
    -------
    G : networkx.Graph
        Nodes carry:
            id, class_id, class_name, area, centroid
        Edges carry:
            edge_type      : "door" | "arch" | "shared-wall"
            boundary_length : shared boundary length (px)
            door_ids       : list of door ids that connect the rooms

    Example
    -------
    >>> plan = mask_to_plan_dict(seg_mask)
    >>> G = build_gt_graph_from_polygons(plan)
    >>> for u, v, d in G.edges(data=True):
    ...     print(G.nodes[u]["class_name"], "↔", G.nodes[v]["class_name"],
    ...           d["edge_type"])
    """
    rooms = plan_dict["rooms"]
    doors = plan_dict.get("doors", [])

    # ── build nodes ──────────────────────────────────────────────────────────
    G = nx.Graph()
    for r in rooms:
        poly = r["polygon"]
        cx, cy = poly.centroid.coords[0]
        G.add_node(
            r["id"],
            class_id=r["class_id"],
            class_name=r["class_name"],
            area=round(poly.area, 1),
            centroid=(round(cy, 1), round(cx, 1)),  # (row, col) convention
        )

    # ── pre-buffer room polygons ─────────────────────────────────────────────
    buffered = {r["id"]: r["polygon"].buffer(adjacency_buffer) for r in rooms}

    # ── pre-buffer door polygons ─────────────────────────────────────────────
    door_polys_buf = [
        {"id": d["id"], "poly": d["polygon"].buffer(door_buffer)}
        for d in doors
    ]

    # ── pairwise adjacency ───────────────────────────────────────────────────
    for a, b in itertools.combinations(rooms, 2):
        buf_a = buffered[a["id"]]
        buf_b = buffered[b["id"]]

        # test adjacency: buffered polygons must intersect
        if not buf_a.intersects(buf_b):
            continue

        # compute shared boundary using the same buffered polygons
        shared_zone = buf_a.intersection(buf_b)
        boundary_length = shared_zone.length

        if boundary_length < shared_wall_min and shared_zone.area < shared_wall_min:
            continue    # not meaningfully adjacent

        # ── check doors in the boundary zone ─────────────────────────────────
        # expand the shared zone to catch nearby doors
        door_search_zone = shared_zone.buffer(door_buffer)
        connecting_doors = []
        for d in door_polys_buf:
            if door_search_zone.intersects(d["poly"]):
                overlap = door_search_zone.intersection(d["poly"])
                if overlap.area >= door_overlap_min:
                    connecting_doors.append(d["id"])

        # ── classify edge ────────────────────────────────────────────────────
        if connecting_doors:
            edge_type = "door"
        elif boundary_length >= arch_min_length:
            edge_type = "arch"
        else:
            edge_type = "shared-wall"

        G.add_edge(
            a["id"],
            b["id"],
            edge_type=edge_type,
            boundary_length=round(boundary_length, 1),
            door_ids=connecting_doors,
        )

    return G


# ──────────────────────────────────────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────────────────────────────────────

def gt_graph_summary(G: nx.Graph) -> str:
    """Return a human-readable summary of the ground-truth graph."""
    lines = [f"GT Graph: {G.number_of_nodes()} rooms, {G.number_of_edges()} edges"]

    lines.append("\nNodes:")
    for nid, data in sorted(G.nodes(data=True)):
        lines.append(
            f"  [{nid:2d}] {data['class_name']:<12s}  "
            f"area={data['area']:>8.0f}px²  "
            f"centroid=({data['centroid'][0]:.0f}, {data['centroid'][1]:.0f})"
        )

    lines.append("\nEdges:")
    for u, v, data in G.edges(data=True):
        u_name = G.nodes[u]["class_name"]
        v_name = G.nodes[v]["class_name"]
        door_str = f"  doors={data['door_ids']}" if data["door_ids"] else ""
        lines.append(
            f"  {u_name}[{u}] ↔ {v_name}[{v}]  "
            f"type={data['edge_type']:<12s}  "
            f"boundary={data['boundary_length']:.0f}px"
            f"{door_str}"
        )

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Smoke test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Smoke-testing build_gt_graph_from_polygons …\n")

    # ── build the same synthetic mask as build_graph.py ──────────────────────
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

    # door between bedroom and kitchen
    mask[50:70, 90:110] = 11

    # arch opening between living and bedroom (remove wall)
    mask[130:140, 20:60] = 0

    # ── step 1: mask → plan_dict ─────────────────────────────────────────────
    print("Converting mask → plan_dict …")
    plan = mask_to_plan_dict(mask)
    print(f"  rooms: {len(plan['rooms'])}  doors: {len(plan['doors'])}")
    for r in plan["rooms"]:
        print(f"    {r['class_name']}: area={r['polygon'].area:.0f}px²")
    for d in plan["doors"]:
        print(f"    Door: area={d['polygon'].area:.0f}px²")

    # ── step 2: plan_dict → graph ────────────────────────────────────────────
    print("\nBuilding ground-truth graph …")
    G = build_gt_graph_from_polygons(plan)
    print(gt_graph_summary(G))

    # ── assertions ───────────────────────────────────────────────────────────
    assert G.number_of_nodes() == 3, f"Expected 3 rooms, got {G.number_of_nodes()}"
    assert G.number_of_edges() >= 1, f"Expected ≥1 edges, got {G.number_of_edges()}"

    # check that Bedroom ↔ Kitchen has a door edge
    for u, v, data in G.edges(data=True):
        names = {G.nodes[u]["class_name"], G.nodes[v]["class_name"]}
        if names == {"Bedroom", "Kitchen"}:
            assert data["edge_type"] == "door", \
                f"Bedroom ↔ Kitchen should be 'door', got '{data['edge_type']}'"
            assert len(data["door_ids"]) >= 1, "Door ids should be populated"
            print(f"\n✓ Bedroom ↔ Kitchen: door edge with door_ids={data['door_ids']}")
            break
    else:
        raise AssertionError("Missing Bedroom ↔ Kitchen edge")

    print("\nAll assertions passed.")
