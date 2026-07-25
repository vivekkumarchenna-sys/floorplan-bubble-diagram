# Handoff: Fix room-area (m²) scaling in bubble diagrams

**Generated**: 2026-07-23 23:47
**Branch**: main
**Status**: In Progress — core fix done & verified; downstream re-render + broader manuscript audit remain

## Goal

Bubble-diagram room areas were wrong (~2.5× too large; bathrooms shown at 12–15 m² instead of 3–6 m²). Regenerate the per-plan pixel→m² scale correctly, re-render all diagrams/figures, and clean up figure legends. A separate, larger manuscript-consistency audit is parked behind this.

## Completed

- [x] **Root-caused the area bug**: `pixel_scale.json` matched each image to a ResPlan plan by **list index** instead of the plan's **`id` field**. Images are named by `id`; list index ≠ id (list[0].id = 14433). Nearly every plan got an unrelated plan's area.
- [x] **Confirmed ResPlan `net_area` is CARPET area** (net internal, wall-excluded). `area` field = built-up (`area/net_area` median 1.37). So the scale denominator must be interior room pixels, no walls, no balcony.
- [x] **Regenerated `pixel_scale.json`** in Colab via `src/build_pixel_scale.py`. 8839/17107 plans covered (52%). Old file backed up to `pixel_scale.backup-20260723-170748.json`.
- [x] **Verified**: 512px frame median 15.3 m (was 22–32); bathroom median 3.59 m²; `interior_px × scale == net_area` exactly (diff +0.00 on plans 0, 10005).
- [x] `src/build_pixel_scale.py`: added `_find_root()` so it works whether placed in `src/` or at Colab project root.
- [x] `src/visualize.py`: legend moved from `upper left` (overlapping bubbles) to a horizontal strip **below** the axes; node sizing already improved earlier (absolute `SQM_TO_POINTS` when `area_sqm` present, else proportional-to-peak).
- [x] `Figures/fig_qualitative.py` + `Figures/fig_failures.py`: replaced 6 per-panel legends with **one shared bottom legend** (`show_legend=False` on graph panels + `fig.legend`).
- [x] Docs written: `docs/AREA_SCALING_PROBLEM.md`, `docs/COLAB_AREA_FIX.md`, `docs/CANONICAL_VALUES.md`.

## Not Yet Done

**Area fix (finish in Colab — needs trained checkpoint + GPU):**
- [ ] Re-render Output diagrams from **covered stems only** (avoid px² fallback): `generate_bubble.py` after uploading edited `visualize.py`.
- [ ] Re-render survey + Fig 6 + Fig 7 into `corrected_figures/` (upload edited fig scripts).
- [ ] **Regression guard**: confirm `summary.csv` (mIoU/edge_f1/ged/frobenius) is byte-identical pre/post — area never touches graph construction, so any drift = a real bug.
- [ ] Decide Fig 7 framing: covered-only (all-m²) vs absolute worst (mixed units). Covered-only cell already provided (filter `per_image.csv` → `per_image_covered.csv`, pass via `--csv`).
- [ ] Commit the 4 changed source files + docs.
- [ ] Manuscript methods note: areas are carpet/net-internal from `net_area`; ~⅓ of plans lack usable `net_area` and are shown in px².

**Broader manuscript audit (parked — see `docs/CANONICAL_VALUES.md`, nothing started):**
- [ ] Headline mIoU: per-image **0.9974** vs macro-over-active-classes **0.7585** — pick one everywhere.
- [ ] **User study has NO data** — `Downloads/user_study_workbook.xlsx` is a blank template (0/1600 cells). Collect it or remove all Table 6 / user-study claims.
- [ ] Canonical Edge F1: 0.6589 (full test n=2100) vs 0.6956 (ablation n=100) vs 0.6685 (hardcoded) — pick one.
- [ ] Backbone comparison invalid: README compares SegFormer test-per-image (0.9974) vs DeepLab val-macro (0.9684). On equal footing (val-macro) **DeepLab 0.9684 > SegFormer 0.9651**.
- [ ] Re-run pipeline on full test set (per_image.csv has 2100 of 2566).
- [ ] Rename ablation variants: "w/o post-processing" (no such stage) and "w/o corridor" (removes Storage class 6, not corridor).
- [ ] 7 code fixes: `eval_gt_upper_bound.py` (frob_normalized KeyError, bare except, hardcoded 0.6685), `build_gt_graph.py` stale schema docstring, `run_ablation.py` Corridor/Storage comment, `train.py` "weighted CE" docstring (weights are None), `dataset.py` CoarseDropout mask-fill check.

## Failed Approaches (Don't Repeat These)

- **Resolution-mismatch hypothesis**: first assumed areas were counted at one resolution but scaled at another. **Wrong** — raster, mask, and inference are all 512×512 (verified). The `--root`/scale path was never the issue; the lookup table was. Don't re-investigate resolution.
- **Index-based stem→plan matching**: this IS the original bug. Matching by list position agreed with mask room-count fingerprints on only **6/60** plans; matching by `id` field agreed on **57/60**. Always match by `id`.
- **`shutil.copytree` / serial copy of masks from Google Drive**: ~1.4 files/s, ~3h for 17k files. Google Drive FUSE throttles many-small-file reads.
- **Threaded copy looked slow at first** (1.38 it/s for the first ~250 while Drive cold) but then hit **47 it/s** once warmed — full 12k copy finished in ~4 min. Use `ThreadPoolExecutor(max_workers=32)`; don't abandon it in the first minute.
- **Running `build_pixel_scale.py` directly against the Drive mount**: ~3h (146 it/s only after copying masks to local `/content/resplan_masks`). Copy masks local first, then run with `--mask-dir /content/resplan_masks`.

## Key Decisions

| Decision | Rationale |
|----------|-----------|
| Scale = `net_area / interior_pixel_count` | Standard raster→metric recovery; makes room areas sum to net_area by construction |
| Denominator = interior rooms only `[1,2,3,4,6,7]`, exclude balcony (class 5) & walls | `net_area` is carpet area; excluding balcony gives bathroom median ~3.8 (architectural norm) |
| Match stem→plan by `id` field, not list index | 57/60 vs 6/60 fingerprint agreement |
| Drop plans with corrupt `net_area` (0, or outside 15–600 m²) | ~⅓ of ResPlan `net_area` is 0 or absurd (max 7.9e10). Dropped plans fall back to px² labels |
| Labels-only scope (no bubble-size logic change beyond what was already done) | User decision; avoids touching metrics |
| No refactor of the 4 duplicate `_load_pixel_scale` copies | User decision; minimal touch |
| Legend moved to bottom / single shared legend | User request — per-panel legends overlapped bubbles |

## Current State

**Working**: `pixel_scale.json` on Drive is correct (8839 entries, verified). Plan 0 renders Living 44.2 / bathrooms 5.0–5.8 / bedrooms 14.3–24.6 m². All edits applied to local repo files.

**Broken / expected**: ~48% of plans (dropped, corrupt net_area) render **px²** labels — this is intended graceful fallback (`visualize.py:180`), not a bug. For figures/survey, pick exemplars from covered stems.

**Uncommitted changes** (git status, excluding Output/ and Downloads/):
- Modified: `src/visualize.py`
- Untracked: `src/build_pixel_scale.py`, `src/eval_gt_upper_bound.py`, `docs/`, `evaluations/`, `Figures/*.pdf`
- `Figures/fig_qualitative.py` and `Figures/fig_failures.py` were edited (see Warnings re: case-sensitivity).

## Files to Know

| File | Why It Matters |
|------|----------------|
| `src/build_pixel_scale.py` | The regenerator. id-match + carpet denominator + corruption filter + fingerprint gate. Untracked — commit it. |
| `pixel_scale.json` | The fixed output (on Drive; local copy is still the OLD broken one — see Warnings). |
| `src/visualize.py` | `draw_bubble_diagram()` — node sizing + legend. Shared by generate_bubble/survey/figs. |
| `src/build_graph.py:255` | Where `area_sqm = area_px × pixel_scale` is applied. |
| `src/generate_bubble.py:176-180` | Consumer: `scale = map.get(stem)` → `build_graph_from_segmentation(pixel_scale=scale)`. Already correct. |
| `Figures/fig_qualitative.py`, `Figures/fig_failures.py` | Single-legend edits. |
| `docs/CANONICAL_VALUES.md` | The parked manuscript audit — every number + every contradiction. |
| `docs/COLAB_AREA_FIX.md` | Full Colab run instructions. |

## Code Context

**Scale formula (`build_pixel_scale.py`):**
```python
INTERIOR_CLASSES = [1, 2, 3, 4, 6, 7]   # bedroom bath kitchen living storage stair (NO balcony, NO wall)
by_id = {e["id"]: i for i, e in enumerate(data)}   # match by id, NOT index
scale = net_area / interior_pixel_count            # m² per pixel, per plan
# gates: net_area in [15,600]; interior_px >= 5000; mask room-counts == pkl polygon counts; frame 5–40 m
```

**Consumer (already correct, do not "fix"):**
```python
scale = pixel_scale_map.get(stem)                       # None if plan dropped
G = build_graph_from_segmentation(mask, pixel_scale=scale)
# build_graph.py:255 -> attrs["area_sqm"] = round(area_px * scale, 2)  (only if scale is not None)
# visualize.py:176 -> label "{name}\n{area_sqm:.1f} m²"  else  "{area:.0f}px²"
```

**Single-legend pattern (both fig scripts):** collect `seen_classes`/`seen_edges` in the row loop, pass `show_legend=False` to `draw_bubble_diagram`, then one `fig.legend(..., loc="lower center", ncol=len(handles))` after `tight_layout(rect=[0,0.045,1,1])`.

## Resume Instructions

All remaining area work runs in Colab. `PROJECT = "/content/drive/MyDrive/bubble_diagram_project"`. **All `.py` files live at the PROJECT ROOT in Colab, not in `src/` or `Figures/`.**

1. Upload edited files to Colab root: `visualize.py`, `fig_qualitative.py`, `fig_failures.py` (and `build_pixel_scale.py` if re-running scale).
2. Ensure masks are local: copy `data/resplan_masks` → `/content/resplan_masks` with a 32-thread `ThreadPoolExecutor` (see `docs/COLAB_AREA_FIX.md`). ~4 min.
3. Re-render qualitative (covered stems, all-m²):
   - Build `stems` = best/median/below-median edge_f1 among plans in `pixel_scale.json` (cell in chat history / COLAB doc).
   - `!python {PROJECT}/fig_qualitative.py --root {PROJECT} --stems {stems} --out {PROJECT}/corrected_figures/fig6_qualitative.pdf`
   - Expected: 3 rows, single bottom legend clear of bubbles, all m² labels.
4. Re-render failures (covered-only): filter `per_image.csv` → `per_image_covered.csv`, then
   - `!python {PROJECT}/fig_failures.py --root {PROJECT} --csv {PROJECT}/corrected_figures/per_image_covered.csv --n 4 --out {PROJECT}/corrected_figures/fig7_failures.pdf`
5. Regression guard: `pd.read_csv(f"{PROJECT}/summary.csv")` — mIoU 0.997434, edge_f1 0.658862, ged 4.2024, frob_normalized 0.4625.
   - Expected: identical to pre-fix. If changed: area logic leaked into graph construction — investigate.

## Setup Required

- Google Drive mounted; trained checkpoint at `{PROJECT}/checkpoints/segformer/best_model.pth`.
- Colab deps: torch, transformers, opencv, networkx, shapely, scipy, tqdm (shapely/networkx/tqdm needed just to unpickle `ResPlan.pkl`).
- `ResPlan.pkl` at `{PROJECT}/data/resplan_raw/ResPlan.pkl` (297 MB, 17000 plans).

## Warnings

- **Case-sensitivity trap**: repo history tracks `figures/` (lowercase); the working tree has `Figures/` (uppercase) where edits were made. `git status` shows lowercase files as **deleted** and uppercase as **untracked**. On the Linux (case-sensitive) box these are two dirs; on a case-insensitive FS they collide. Resolve the rename deliberately (`git mv`) before committing, or the fig-script edits may appear lost.
- **Local `pixel_scale.json` in the repo is still the OLD broken file** (Apr 5, 17107 entries, ~18.5 m median). The FIXED one (8839 entries) is on Google Drive only. To validate locally, pull the Drive copy first.
- `generate_bubble.py --limit N` grabs the first N stems regardless of coverage → mixed m²/px². Use covered-stems selection for clean figures.
- px² labels are correct behavior for dropped plans, not a failure.
- Do NOT "fix" the consumer scripts to force m² — they're correct; forcing it would mislabel dropped plans.
- Fig 7 currently deliberately shows worst edge_f1; many worst plans are dropped (px²). Covered-only framing excludes the absolute-worst — a defensible but stated scoping choice.
