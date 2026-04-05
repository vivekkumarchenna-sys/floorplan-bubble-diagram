"""
compute_split_stats.py — Count room and door instances per split
=================================================================
Reads masks for each split, runs connected-component analysis,
and outputs dataset partition statistics.

Usage (Colab):
    !python compute_split_stats.py
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    _SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    _SCRIPT_DIR = Path("/content/drive/MyDrive/bubble_diagram_project")

sys.path.insert(0, str(_SCRIPT_DIR))

import cv2
import numpy as np

# class ids
ROOM_CLASSES = {1, 2, 3, 4, 5, 6, 7, 8, 9}
DOOR_CLASSES = {11, 13}

ROOT = _SCRIPT_DIR
MASK_DIR = ROOT / "data" / "resplan_masks"
SPLIT_DIR = ROOT / "data" / "splits"


def count_instances(mask: np.ndarray, class_ids: set[int], min_area: int = 100) -> int:
    """Count connected components across the given class ids."""
    total = 0
    for cls in class_ids:
        cls_mask = (mask == cls).astype(np.uint8)
        if cls_mask.sum() == 0:
            continue
        n_labels, _ = cv2.connectedComponents(cls_mask, connectivity=8)
        # n_labels includes background (label 0)
        total += n_labels - 1
    return total


def process_split(split_name: str):
    split_file = SPLIT_DIR / f"{split_name}.txt"
    if not split_file.exists():
        print(f"  {split_name}: FILE NOT FOUND")
        return None

    stems = [l.strip() for l in split_file.read_text().splitlines() if l.strip()]
    n_plans = len(stems)

    total_rooms = 0
    total_doors = 0
    total_edges = 0

    for i, stem in enumerate(stems):
        mask_path = MASK_DIR / f"{stem}_mask.png"
        if not mask_path.exists():
            continue

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        rooms = count_instances(mask, ROOM_CLASSES)
        doors = count_instances(mask, DOOR_CLASSES)

        total_rooms += rooms
        total_doors += doors

        if (i + 1) % 2000 == 0:
            print(f"    [{i+1}/{n_plans}]")

    mean_rooms = total_rooms / n_plans if n_plans > 0 else 0
    mean_doors = total_doors / n_plans if n_plans > 0 else 0

    return {
        "split": split_name,
        "plans": n_plans,
        "room_instances": total_rooms,
        "door_instances": total_doors,
        "mean_rooms": round(mean_rooms, 2),
        "mean_doors": round(mean_doors, 2),
    }


if __name__ == "__main__":
    print("Computing dataset partition statistics...\n")

    results = []
    for split in ["train", "val", "test"]:
        print(f"  Processing {split}...")
        r = process_split(split)
        if r:
            results.append(r)
            print(f"    {r['plans']} plans | {r['room_instances']} rooms | "
                  f"{r['door_instances']} doors | "
                  f"mean {r['mean_rooms']} rooms/plan | "
                  f"mean {r['mean_doors']} doors/plan")

    # totals
    total_plans = sum(r["plans"] for r in results)
    total_rooms = sum(r["room_instances"] for r in results)
    total_doors = sum(r["door_instances"] for r in results)

    print(f"\n{'='*70}")
    print(f"{'Split':>8s} | {'Plans':>7s} | {'Rooms':>8s} | {'Doors':>8s} | {'Rooms/plan':>10s} | {'Doors/plan':>10s}")
    print(f"{'-'*70}")
    for r in results:
        print(f"{r['split']:>8s} | {r['plans']:>7d} | {r['room_instances']:>8d} | {r['door_instances']:>8d} | {r['mean_rooms']:>10.2f} | {r['mean_doors']:>10.2f}")
    print(f"{'-'*70}")
    print(f"{'Total':>8s} | {total_plans:>7d} | {total_rooms:>8d} | {total_doors:>8d} |")
    print(f"{'='*70}")
