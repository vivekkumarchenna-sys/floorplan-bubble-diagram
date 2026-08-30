"""
build_gt_graph.py - Ground-truth room-adjacency graph
=========================================================================
Three entry points:

1. build_gt_graph_from_resplan(plan_record)
       READS ResPlan's own typed graph (the ``graph`` key in ResPlan.pkl)
       for a single plan record. This is the ground truth: it is not
       reconstructed from pixels at all, it is the dataset's own
       annotation. Use this one. The two functions below reconstruct a graph
       geometrically and must NOT be used as ground truth (see function 2).

2. build_gt_graph_from_polygons(plan_dict)  [legacy, geometric reconstruction]
       Takes a plan_dict with Shapely polygons -> NetworkX graph, using
       our own adjacency-buffer / boundary-length heuristics. This does
       NOT reproduce ResPlan's typed graph: it has no test for an actual
       opening width, so it labels almost
       every non-door adjacency "arch" and almost never "shared-wall",
       which is the opposite of the released dataset's own class balance.
       Retained only for the M2-on-predicted/GT-mask pipeline comparison
       (i.e. as a *second model*, not as ground truth) and for callers that
       have no plan id to look up in ResPlan.pkl (e.g. purely synthetic masks).

3. mask_to_plan_dict(seg_mask)
       Converts a (H, W) segmentation mask to plan_dict format, so masks
       can feed into build_gt_graph_from_polygons above.

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

Class convention (same as build_graph.py / train.py - this is the actual
16-class schema; there is no Corridor, Dining, Garage or Staircase class):
    0  Background     5  Balcony     10  Wall
    1  Bedroom        6  Storage     11  Door
    2  Bathroom       7  Stair       12  Window
    3  Kitchen        8  Parking     13  FrontDoor
    4  Living         9  Pool        14  Column
                                     15  Other
"""

from __future__ import annotations

import itertools
import pickle
from pathlib import Path
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
    14: "Column",
    15: "Other",
}

ROOM_CLASSES  = {1, 2, 3, 4, 5, 6, 7, 8, 9}   # all room/space types
DOOR_CLASSES  = {11, 13}                         # Door, FrontDoor


# ──────────────────────────────────────────────────────────────────────────────
# ResPlan's own typed graph - the real ground truth
# ──────────────────────────────────────────────────────────────────────────────
#
# ResPlan.pkl stores, per plan, a `graph` key: a NetworkX graph whose nodes
# are named "{room_type}_{instance}" (e.g. "bedroom_0", "kitchen_0") with
# attributes {geometry, type, area}, and whose edges carry a `type` in
# {"via_door", "adjacency", "direct", "via_window"}. This dataset-native graph,
# not a geometric reconstruction from pixels, is what Section 6 must be scored
# against.
#
# Node types observed across all 17,107 plans: bedroom, bathroom, kitchen,
# living, balcony, front_door. front_door (and any of garden/veranda/inner/
# land/pool/storage/stair/parking, present in the plan-level dict but never
# seen as graph nodes) are dropped - they are not rooms in the M2 sense.
#
# Edge-type mapping to the paper's three-way taxonomy (door / arch /
# shared-wall).
#
# *** direct does NOT mean arch. This was checked and is now corrected. ***
#
# An earlier draft mapping (via_door->door, adjacency->shared-wall,
# direct->arch, via_window->window) was a closest-string-name guess that
# needed confirmation. Two independent checks now contradict the direct->arch
# part specifically:
#
#   1. Empirical, dataset-wide: every single `direct`-typed edge in all
#      17,107 plans (16,964/16,964, zero exceptions) connects a `living` node
#      to the `front_door` pseudo-node. `front_door` nodes never have any
#      other edge type and never have degree > 1. There is no `direct` edge
#      between two interior rooms anywhere in the released data.
#   2. Documentary: the actual ResPlan paper (Abouagour & Garyfallidis,
#      arXiv:2508.14006, Aug 2025), Section 4 "Graph Construction", defines
#      its three edge categories verbatim as: "door edges connect rooms that
#      share a doorway; arch edges connect rooms separated by a large open
#      passage (without a door); and shared-wall edges indicate two rooms
#      that are adjacent with a wall between them but no direct opening."
#      This is an explicitly room-to-room definition of "arch" - it cannot be
#      the living-to-entrance relationship `direct` actually encodes. Note
#      also that the paper's own edge-type strings are literally "door" /
#      "arch" / "shared-wall", not the via_door/adjacency/direct/via_window
#      strings the released .pkl actually uses - the released schema and the
#      paper's prose do not match 1:1.
#
# Practical effect: `direct` is mapped to "arch" below purely so the
# dictionary lookup has an entry (RESPLAN_EDGE_TYPE_MAP.get() needs
# *something*), but it is functionally inert - every `direct` edge touches
# `front_door`, which is dropped as a non-room node before this map is ever
# consulted (see the edge-building loop below), so no edge is ever actually
# labelled "arch" via this path. **"arch", as ResPlan's own paper defines it,
# has no recoverable room-to-room ground truth in the data this project has
# access to.** A geometric attempt to recover it from the released wall
# polygons (splitting `adjacency` by shared-boundary wall coverage) was tried
# and abandoned: ~150 sampled real `adjacency` pairs showed a continuous,
# unimodal wall-coverage distribution (0.1-0.6), not the clean near-zero
# cluster a genuine "no wall present" arch subset would produce. Manufacturing
# an arch label from that noise would repeat the same mistake this whole
# correction pass exists to fix. Treat any "arch" row in this project's
# tables/figures as "not evaluable against ResPlan's public release", not as
# a low-but-real number.
#     via_door  -> door          (an actual doorway)
#     adjacency -> shared-wall   (rooms touch, no opening - see caveat above:
#                                 may also silently include true arches the
#                                 released data does not distinguish)
#     direct    -> arch          (inert - see above; never actually fires)
#     via_window -> "window"     (NOT one of the paper's three types. ResPlan's
#         paper doesn't discuss this edge type in its Graph Construction prose
#         at all, but the released data ships it (1.99% of all edges) and it
#         is physically a real opening in the wall, not a walkable one.
#         Collapsing it into shared-wall or arch would invent an equivalence
#         neither the paper nor the data states. Kept as an explicit fourth
#         edge_type and reported separately.

RESPLAN_TYPE_TO_CLASS_ID = {
    "bedroom":  1,
    "bathroom": 2,
    "kitchen":  3,
    "living":   4,
    "balcony":  5,
    "storage":  6,
    "stair":    7,
    "parking":  8,
    "pool":     9,
}

# Non-room node types that appear (or could appear) in ResPlan's graph but
# are not part of the M2 room-adjacency schema.
RESPLAN_NONROOM_TYPES = {"front_door", "garden", "veranda", "inner", "land"}

RESPLAN_EDGE_TYPE_MAP = {
    "via_door":  "door",
    "adjacency": "shared-wall",
    "direct":    "arch",
    "via_window": "window",
}


def load_resplan_records(pkl_path: str | Path) -> dict[int, dict[str, Any]]:
    """
    Load ResPlan.pkl once and index plan records by their integer ``id``
    (the same id used in ``data/resplan_masks/{id}_mask.png`` filenames).
    """
    pkl_path = Path(pkl_path)
    with open(pkl_path, "rb") as f:
        records = pickle.load(f)
    return {int(r["id"]): r for r in records}


def build_gt_graph_from_resplan(
    plan_record: dict[str, Any],
    *,
    edge_type_map: dict[str, str] | None = None,
    drop_types: set[str] = RESPLAN_NONROOM_TYPES,
) -> nx.Graph:
    """
    Build the ground-truth room-adjacency graph directly from ResPlan's own
    annotation, i.e. ``plan_record["graph"]``. No geometry is re-derived and
    no threshold is applied: every node and edge is exactly what ResPlan
    shipped, modulo dropping non-room nodes and renaming edge types onto the
    paper's taxonomy (see module docstring / RESPLAN_EDGE_TYPE_MAP above).

    Node ids are renumbered to small integers, ordered by ``class_id``
    ascending and then by centroid (row, then column) within a class, the
    same convention ``build_graph_from_segmentation`` uses for connected
    components, so that "ClassName[id]"-based edge matching against a
    predicted graph is at least self-consistent. This does not remove the
    instance-id fragility documented in Section 6.3/6.5 (an id-independent
    matching variant exists in eval_pipeline_variants.py for that), it only
    ensures the ground-truth side of the comparison is built the same way
    every time.

    Parameters
    ----------
    plan_record   : one entry of the list stored in ResPlan.pkl (has a
                     "graph" key holding a NetworkX graph).
    edge_type_map : override for RESPLAN_EDGE_TYPE_MAP.
    drop_types    : node "type" values to exclude as non-room.

    Returns
    -------
    G : networkx.Graph
        Nodes carry: class_id, class_name, area, centroid, resplan_name
        Edges carry: edge_type ("door" | "arch" | "shared-wall" | "window"),
                     resplan_type (the untranslated ResPlan label)
    """
    if edge_type_map is None:
        edge_type_map = RESPLAN_EDGE_TYPE_MAP

    G_raw = plan_record["graph"]

    room_items = [
        (name, data) for name, data in G_raw.nodes(data=True)
        if data.get("type") not in drop_types
    ]

    def _sort_key(item):
        name, data = item
        class_id = RESPLAN_TYPE_TO_CLASS_ID.get(data.get("type"), 999)
        centroid = data["geometry"].centroid
        # ResPlan's polygon coordinates use a y-up convention; the raster
        # masks (and every id assigned by build_graph_from_segmentation) use
        # row-major, y-down pixel order. Measured over 397 single-kitchen
        # test plans, corr(pkl_y, mask_row) = -0.71 (see
        # eval_pipeline_variants._normalized_centroids for the same finding
        # applied to Hungarian matching). Negating y here orders same-class
        # instances top-to-bottom in the *raster* sense, so canonical-id
        # numbering is at least oriented the same way as the predicted
        # graph's, even though it is not the same numeric coordinate frame.
        return (class_id, round(-centroid.y, 3), round(centroid.x, 3))

    room_items.sort(key=_sort_key)

    G = nx.Graph()
    id_map: dict[str, int] = {}
    for new_id, (name, data) in enumerate(room_items):
        class_id = RESPLAN_TYPE_TO_CLASS_ID.get(data.get("type"))
        class_name = CLASS_NAMES.get(class_id, str(data.get("type")))
        centroid = data["geometry"].centroid
        G.add_node(
            new_id,
            class_id=class_id,
            class_name=class_name,
            area=round(float(data.get("area", data["geometry"].area)), 1),
            centroid=(round(centroid.y, 1), round(centroid.x, 1)),
            resplan_name=name,
        )
        id_map[name] = new_id

    for u, v, data in G_raw.edges(data=True):
        if u not in id_map or v not in id_map:
            continue  # edge touches a dropped non-room node (e.g. front_door)
        raw_type = data.get("type")
        edge_type = edge_type_map.get(raw_type, raw_type)
        G.add_edge(
            id_map[u], id_map[v],
            edge_type=edge_type,
            resplan_type=raw_type,
        )

    return G


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
    seg_mask     : (H, W) int array - segmentation mask.
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
        door_str = f"  doors={data['door_ids']}" if data.get("door_ids") else ""
        lines.append(
            f"  {u_name}[{u}] <-> {v_name}[{v}]  "
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
    print("Converting mask -> plan_dict...")
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
    assert G.number_of_edges() >= 1, f"Expected >=1 edges, got {G.number_of_edges()}"

    # check that Bedroom <-> Kitchen has a door edge
    for u, v, data in G.edges(data=True):
        names = {G.nodes[u]["class_name"], G.nodes[v]["class_name"]}
        if names == {"Bedroom", "Kitchen"}:
            assert data["edge_type"] == "door", \
                f"Bedroom <-> Kitchen should be 'door', got '{data['edge_type']}'"
            assert len(data["door_ids"]) >= 1, "Door ids should be populated"
            print(f"\n[OK] Bedroom <-> Kitchen: door edge with door_ids={data['door_ids']}")
            break
    else:
        raise AssertionError("Missing Bedroom <-> Kitchen edge")

    print("\nAll assertions passed.")


def attach_gt_area_sqm(G, record) -> bool:
    """Attach ``area_sqm`` to a ResPlan ground-truth graph, in square metres.

    The ground-truth graph's ``area`` is a polygon area in ResPlan's own
    per-plan vector coordinate space. ``pixel_scale.json`` is NOT the right
    conversion for it: that table is built as ``net_area / interior_pixel_count``
    over the 512x512 raster mask (see build_pixel_scale.py), so it converts
    *raster pixels* to m². Applying it to vector polygon areas mixes two
    coordinate spaces and, measured over 400 plans, understated ground-truth
    room areas by a median factor of ~2.7 (range 1.1-3.4, i.e. not even a
    constant offset, because each plan's vector extent differs). Predicted
    graphs are unaffected: they measure the raster mask, which is what
    pixel_scale is calibrated for.

    Instead normalise by the plan's own ``net_area``, the same metric quantity
    build_pixel_scale.py calibrates the raster against. Both sides then sum to
    ResPlan's own net area and are directly comparable bubble-for-bubble.

    Returns True if areas were attached, False if this plan has no usable
    ``net_area`` (about 40% of ResPlan; those graphs keep pixel-area labels).
    """
    net_area = record.get("net_area")
    if not net_area or not (1.0 <= float(net_area) <= 10_000.0):
        return False
    total = sum(float(G.nodes[n].get("area", 0.0)) for n in G.nodes())
    if total <= 0:
        return False
    factor = float(net_area) / total
    for nid in G.nodes():
        G.nodes[nid]["area_sqm"] = round(G.nodes[nid].get("area", 0.0) * factor, 2)
    return True
