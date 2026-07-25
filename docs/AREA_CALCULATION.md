# Room Area Calculation (m²)

How bubble-diagram room areas are computed, and the bug that was fixed.

## Method

Each room's floor area in square metres is:

```
area_sqm = room_pixel_count × pixel_scale[stem]
```

- `room_pixel_count` — pixels of that room's connected component in the
  512×512 segmentation mask (`build_graph.py`).
- `pixel_scale[stem]` — a per-plan conversion factor (m² per pixel) stored in
  `pixel_scale.json`, keyed by the image stem.
- Applied in `build_graph.py` (`area_sqm = round(area_px × pixel_scale, 2)`) and
  labelled in `visualize.py`.

A **separate factor per plan** is required: every plan is rendered to the same
512×512 frame, so one pixel represents more real-world area in a larger dwelling.

## Deriving the per-plan scale

`src/build_pixel_scale.py` derives the factor from ResPlan ground truth:

```
pixel_scale[stem] = net_area / interior_pixel_count
```

- `net_area` — the plan's true floor area (m²) from `ResPlan.pkl`. This is the
  **carpet / net-internal area**: room interiors only, excluding walls and
  balconies. (ResPlan's other `area` field is the built-up figure, ~1.37×
  larger, and is **not** used.)
- `interior_pixel_count` — sum of mask pixels over interior room classes
  `[1,2,3,4,6,7]` = bedroom, bathroom, kitchen, living, storage, stair.
  Walls (10) and balcony (5) are excluded so the denominator matches what
  `net_area` measures.

Because the scale is defined this way, the interior room areas in a plan sum
back to `net_area` exactly. The **total is guaranteed correct**; the split
between rooms is only as good as the segmentation. (Balcony rooms still receive
a proportional m² label — balcony is excluded from the *denominator*, not
unlabelled.)

### Quality gates (a plan is dropped unless all pass)

- stem resolves to a pickle entry via the **`id` field** (not list index)
- `net_area` within [15, 600] m²
- interior pixel count ≥ 5000
- per-class room counts in the mask match the pickle polygon counts (fingerprint)
- resulting scale implies a 512px frame between 5 m and 40 m

~52% of plans pass; the rest have corrupt `net_area` (≈⅓ of ResPlan is 0 or
absurd, up to 7.9×10¹⁰) or fail the fingerprint. Dropped plans have no scale and
fall back to `px²` labels in the diagram (`visualize.py`).

## The bug that was fixed

The original `pixel_scale.json` matched each image to a ResPlan plan by its
**position in the list** instead of the plan's **`id` field**. A plan's list
index ≠ its `id` (list entry 0 has `id = 14433`; images are named by `id`), so
nearly every plan was scaled by an **unrelated plan's area**.

Effect: areas inflated ~2.5×, by a different factor per plan — bathrooms shown
at 12–15 m² instead of 3–6 m².

Confirmed by structural fingerprint (mask room-counts vs pickle polygon counts):

| Matching method | Plans matched (of 60) |
|---|---|
| by list index | 6 (10%) |
| by `id` field | 57 (95%) |

The fix: `src/build_pixel_scale.py` matches by `id` and applies the gates above.

## Verification (after regeneration)

- 512px frame median **15.3 m** (was 22–32 m on affected plans)
- bathroom area median **3.59 m²** (was 12–15 m²)
- `interior_pixel_count × scale == net_area` exactly (diff +0.00 on plans 0, 10005)
- segmentation and graph metrics **unchanged** — area never enters graph
  construction (regression-checked against `summary.csv`)

## Regenerating

```
python src/build_pixel_scale.py --dry-run   # inspect coverage + frame median
python src/build_pixel_scale.py             # backs up the old file, then writes
```

Then re-run `generate_bubble.py` / the figure scripts. Full Colab instructions:
`docs/COLAB_AREA_FIX.md`. Background narrative: `docs/AREA_SCALING_PROBLEM.md`.

## Reporting note (for the manuscript)

Displayed areas are **carpet (net-internal) areas** derived from ResPlan
`net_area`. Plans without a reliable `net_area` are shown in px² and are omitted
from any area-labelled figure set.
