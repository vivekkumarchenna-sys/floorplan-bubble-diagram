"""
build_pixel_scale.py — Derive per-plan pixel scale from ResPlan ground truth
============================================================================
Writes ``pixel_scale.json``: a mapping ``{stem: m² per pixel}`` used by
``build_graph.py`` to attach ``area_sqm`` to room nodes.

Scale is derived as::

    scale = net_area / interior_pixel_count

where ``net_area`` (m²) comes from the ResPlan pickle and the pixel count is
taken from the ground-truth mask over the interior room classes.

Plans are dropped unless they pass every gate below, because ResPlan's
``net_area`` field is unreliable for a large fraction of the dataset
(~40% are zero or absurd, up to 7.9e10 m²). Roughly half of all plans
survive; coverage is uniform across train/val/test so no split bias is
introduced.

Gates:
    * stem resolves to a pickle entry via the ``id`` field (not list index)
    * net_area within [NET_AREA_LO, NET_AREA_HI]
    * interior pixel count >= MIN_INTERIOR_PX
    * per-class room counts in the mask match the pickle polygons exactly
    * resulting scale implies a 512 px frame between 5 m and 40 m

Usage:
    python src/build_pixel_scale.py
    python src/build_pixel_scale.py --dry-run
    python src/build_pixel_scale.py --mask-dir /content/masks
"""

from __future__ import annotations

import argparse
import datetime
import json
import pickle
import shutil
import sys
from pathlib import Path

try:
    _SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    _SCRIPT_DIR = Path("/content/drive/MyDrive/bubble_diagram_project")

sys.path.insert(0, str(_SCRIPT_DIR))


def _find_root(start: Path) -> Path:
    """Locate the project root (the dir containing data/resplan_raw/ResPlan.pkl).

    Works whether this script sits in ``src/`` or at the project root (e.g. a
    flat Colab Drive layout). Falls back to the script's parent if nothing
    matches, preserving the previous default.
    """
    for cand in (start, start.parent, *start.parents):
        if (cand / "data" / "resplan_raw" / "ResPlan.pkl").exists():
            return cand
    return start.parent

import cv2
import numpy as np
from scipy import ndimage
from tqdm import tqdm


# Room classes whose pixels net_area is taken to cover (balcony excluded —
# it yields a bathroom median closer to the architectural norm than including
# it, and matches the usual definition of net internal area).
INTERIOR_CLASSES = [1, 2, 3, 4, 6, 7]   # bedroom bath kitchen living storage stair

# Classes used as a structural fingerprint to verify stem→plan alignment.
FINGERPRINT_CLASSES = {"bedroom": 1, "bathroom": 2, "kitchen": 3,
                       "living": 4, "balcony": 5}

MIN_COMPONENT_PX = 100          # ignore specks when counting rooms
MIN_INTERIOR_PX = 5000
NET_AREA_LO, NET_AREA_HI = 15.0, 600.0          # plausible dwelling, m²
FRAME_LO_M, FRAME_HI_M = 5.0, 40.0              # implied extent of a 512 px frame
IMG_SIZE = 512


def _npoly(geom) -> int:
    """Number of polygons in a shapely geometry field (may be absent/empty)."""
    if geom is None:
        return 0
    if hasattr(geom, "is_empty") and geom.is_empty:
        return 0
    return len(geom.geoms) if hasattr(geom, "geoms") else 1


def _pkl_fingerprint(entry: dict) -> dict[str, int]:
    return {name: _npoly(entry.get(name)) for name in FINGERPRINT_CLASSES}


def _mask_fingerprint(mask: np.ndarray, counts: np.ndarray) -> dict[str, int]:
    """Count connected components per class, skipping classes with no pixels."""
    out: dict[str, int] = {}
    for name, cls in FINGERPRINT_CLASSES.items():
        if counts[cls] == 0:
            out[name] = 0
            continue
        labelled, n = ndimage.label(mask == cls)
        if n == 0:
            out[name] = 0
            continue
        sizes = np.bincount(labelled.ravel())[1:]
        out[name] = int((sizes >= MIN_COMPONENT_PX).sum())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive pixel_scale.json from ResPlan net_area")
    parser.add_argument("--root", type=str, default=str(_find_root(_SCRIPT_DIR)))
    parser.add_argument("--mask-dir", type=str, default=None,
                        help="Override mask directory (a local copy is much "
                             "faster than reading from a Drive mount)")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="Report statistics without writing")
    args = parser.parse_args()

    root = Path(args.root)
    mask_dir = Path(args.mask_dir) if args.mask_dir else root / "data" / "resplan_masks"
    pkl_path = root / "data" / "resplan_raw" / "ResPlan.pkl"
    out_path = Path(args.out) if args.out else root / "pixel_scale.json"

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    by_id = {entry["id"]: i for i, entry in enumerate(data)}
    print(f"[scale] Loaded {len(data)} plans ({len(by_id)} unique ids)")

    # Restrict to stems that appear in a split, falling back to every mask.
    stems: set[str] = set()
    for split in ("train", "val", "test"):
        split_file = root / "data" / "splits" / f"{split}.txt"
        if split_file.exists():
            stems |= {l.strip() for l in split_file.read_text().splitlines() if l.strip()}
    ordered = sorted(stems) if stems else sorted(p.name[:-9] for p in mask_dir.glob("*_mask.png"))
    print(f"[scale] Candidate stems: {len(ordered)}")

    scale_lo = (FRAME_LO_M / IMG_SIZE) ** 2
    scale_hi = (FRAME_HI_M / IMG_SIZE) ** 2

    scales: dict[str, float] = {}
    skipped = {"no_entry": 0, "bad_net_area": 0, "unreadable": 0,
               "tiny_mask": 0, "fp_mismatch": 0, "scale_range": 0}

    for stem in tqdm(ordered, desc="Deriving scale"):
        # Cheap pickle-only gates first — no disk read for plans that fail.
        idx = by_id.get(int(stem))
        if idx is None:
            skipped["no_entry"] += 1
            continue
        entry = data[idx]
        net_area = entry.get("net_area")
        if net_area is None or not np.isfinite(net_area) \
                or not (NET_AREA_LO <= net_area <= NET_AREA_HI):
            skipped["bad_net_area"] += 1
            continue

        mask = cv2.imread(str(mask_dir / f"{stem}_mask.png"), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            skipped["unreadable"] += 1
            continue

        counts = np.bincount(mask.ravel(), minlength=16)
        interior_px = int(counts[INTERIOR_CLASSES].sum())
        if interior_px < MIN_INTERIOR_PX:
            skipped["tiny_mask"] += 1
            continue

        scale = float(net_area) / interior_px
        if not (scale_lo <= scale <= scale_hi):
            skipped["scale_range"] += 1
            continue

        # Most expensive check last: does this mask really belong to this plan?
        if _pkl_fingerprint(entry) != _mask_fingerprint(mask, counts):
            skipped["fp_mismatch"] += 1
            continue

        scales[stem] = scale

    values = np.array(list(scales.values()))
    metres_per_px = np.sqrt(values)
    print(f"\n[scale] Derived {len(scales)}/{len(ordered)} "
          f"({100 * len(scales) / max(len(ordered), 1):.0f}%)")
    print(f"[scale] Skipped: {skipped}")
    print(f"[scale] {IMG_SIZE}px frame: median={np.median(metres_per_px) * IMG_SIZE:.1f} m  "
          f"p5={np.percentile(metres_per_px, 5) * IMG_SIZE:.1f}  "
          f"p95={np.percentile(metres_per_px, 95) * IMG_SIZE:.1f}")

    for split in ("train", "val", "test"):
        split_file = root / "data" / "splits" / f"{split}.txt"
        if not split_file.exists():
            continue
        split_stems = [l.strip() for l in split_file.read_text().splitlines() if l.strip()]
        hit = sum(s in scales for s in split_stems)
        print(f"[scale]   {split}: {hit}/{len(split_stems)} "
              f"({100 * hit / max(len(split_stems), 1):.0f}%)")

    if args.dry_run:
        print("\n[scale] --dry-run, nothing written")
        return

    if out_path.exists():
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = out_path.with_name(f"{out_path.stem}.backup-{stamp}.json")
        shutil.copy2(out_path, backup)
        print(f"\n[scale] Backed up existing -> {backup.name}")

    with open(out_path, "w") as f:
        json.dump(scales, f)
    print(f"[scale] Wrote {out_path} ({len(scales)} entries)")


if __name__ == "__main__":
    main()
