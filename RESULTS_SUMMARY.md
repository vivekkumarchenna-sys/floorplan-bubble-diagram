# CORRECTED-GT REWORK - verified results (source of truth for manuscript/PPT)
Computed 2026-07-28 in this session; all numbers reproduced from released data + checkpoints.
Scripts in `CORRECTED_GT_METHOD/` (run copies live in scratchpad `gtwork/`).

## Segmentation (M1) - unchanged, still valid
- Per-image mIoU 0.9974 ; SegFormer val mIoU 0.9651 (checkpoint epoch 23, val_mIoU 0.965113).

## Corrected GT composition (full dataset, n = 17,107 plans)
- Per plan: **6.04 door, 0.90 open, 4.47 shared-wall, 8.16 rooms**.
- Totals: 103,258 door + 15,357 open + 76,424 shared-wall edges; 139,555 rooms.
- (STEP-1 60-plan sweep agreed: 6.28 door / 0.83 open / 4.37 shared per plan; only
  0.8% of rooms isolated, every case a genuine physical detachment - external parking
  or a fragmented raster region - not a threshold failure. Fixed-px thresholds
  generalise; no scale-aware conversion needed.)
- Thresholds (frozen): door_dil=8, door_frac=0.4, open_min=45, min_wall_contact=250, adj_dil=9.

## STEP 3 - ResPlan typed-GT is unreliable (the headline finding, full dataset n=17,107)
Corrected-GT edges matched to ResPlan's own graph by class + area-rank
(matching validated: plan 16649 reproduces the architect's 3-of-6-doors-mislabelled reading).

**Cross-tab - what ResPlan calls each room pair our geometry types as a DOOR (n=103,258):**
| ResPlan label of a real door | count | share |
|---|---|---|
| door (correct) | 49,267 | **47.7%** |
| shared-wall (mislabel) | 40,976 | **39.7%** |
| absent (no edge) | 12,971 | 12.6% |
| window | 44 | 0.0% |

- **~40% of real interior doors are typed "shared-wall" by ResPlan** (≈2/5; the handoff's
  earlier "~1/3" was the conservative aggregate estimate - the edge-matched figure is higher).
- **Door-count gap:** corrected **6.04** door-edges/plan vs ResPlan **4.00** door-edges/plan
  (≈1/3 fewer). This is NOT an artifact of our multi-room "hub" rule: across a 500-plan
  sample only **2.9%** of door edges come from a door blob opening onto ≥2 rooms
  (mean 1.030 edges per physical door blob).
- **Arch is unrecoverable:** ResPlan has **0** room-to-room `arch` edges across all 17,107
  plans (every `direct` edge points at the front door / entrance). Confirms the paper's
  existing arch audit and motivates replacing arch with **open passage**.
- Corrected `open` passages: 85.6% of them ResPlan calls shared-wall, 14.0% absent  - 
  ResPlan has no "open" concept, so open-plan connectivity is lost in its graph.

## Pipeline scoring - the contrast that exposes the GT problem
| Scoring | Edge F1 | type acc | door recall | shared-wall recall |
|---|---|---|---|---|
| OLD M2 (dilation) vs **ResPlan** GT (the paper's numbers) | 0.5096 | 0.6302 | 0.999 | 0.143 |
| OLD M2 (dilation) vs **corrected** GT, PREDICTED masks (n=2,567; see "Table 5 row 2 restated" below) | 0.9462 | 0.6943 | 0.9997 | 0.4322 |
| **NEW M2 (build_true_graph) end-to-end on PREDICTED masks vs corrected GT** | **0.9977** | **0.9997** | **0.9993** | **0.9925** |

STEP-4 full result (n=2,567 test plans, SegFormer predicted masks → build_true_graph, vs corrected GT):
Edge F1 **0.9977**, type acc **0.9997**, door recall 0.9993 (15569/15580), **open recall 0.9817** (2249/2291),
shared-wall recall 0.9925 (11449/11535), pooled type acc 0.9996 (29267/29278).
Interpretation: the corrected GT is `build_true_graph(GT mask)` and the pipeline is `build_true_graph(predicted mask)`;
because segmentation is ~0.997 mIoU the graph is reproduced almost exactly, so **perception is not the constraint**  - 
the end-to-end pipeline delivers the specified typed diagram. (This number measures segmentation robustness of the
deterministic rules, NOT independent validation of the rules - that comes from the architect check + 60-plan sweep.)

- The jump from 0.51 → 0.95 Edge F1 on the *same pipeline* is almost entirely the GT fix.
- Residual under OLD M2: shared-wall recall 0.43 (and open recall 0, since it has no open category) - a *genuine* limitation (it over-types
  shared-walls that sit next to a doorway), separate from the GT problem.
- NEW M2 on predicted masks isolates segmentation-only error (seg mIoU ~0.997) → near-perfect
  end-to-end reproduction of the corrected GT (10-plan spot check: 10/10 plans exact).

## Worked example - plan 16649
5–6 real interior doors. Corrected GT: 6 door + 4 shared-wall. ResPlan: 3 door + 3 shared-wall
 -  it mislabels the 3 Living-adjacent doors (Living–Bedroom1, Living–Bedroom2, Living–Kitchen)
as shared-wall. Architect-confirmed.

## Decisions locked (this rework)
- 3 edge types: **door / open passage / shared-wall** (arch removed).
- Two categorical outputs: bubble diagram (geographic, raster colours) + typed adjacency matrix.
- No edge weights, no weighted proximity matrix, **no Frobenius metric**.

## Lighter-backbone baseline - SegFormer-B0 (added 2026-08-31)

Trained under the identical protocol (same fixed splits, 512x512, augmentation,
CE + 0.5*Dice, AdamW + cosine, seed 42, batch 8, 27-epoch budget) on an RTX 5060
laptop GPU. 3.10 h wall-clock; best val mIoU **0.9602** at epoch 18.

Evaluated on the full 2,567-plan test split with `src/eval_backbone.py`
(M2 on predicted masks vs the geometry-derived reference, as Table 5 row 3):

| Backbone | Trainable params | Test per-image mIoU | Edge F1 | Type acc | door / open / shared |
|---|---|---|---|---|---|
| SegFormer-B3 | 47,234,768 | 0.9974 | 0.9977 | 0.9997 | 0.999 / 0.982 / 0.993 |
| SegFormer-B0 | 3,718,256  | 0.9928 | 0.9971 | 0.9996 | 0.999 / 0.984 / 0.990 |

**A 12.7x smaller encoder with ~3x the per-pixel error recovers the same typed
graph.** This is the most direct evidence that segmentation accuracy is not the
limiting factor on these plans (paper Appendix B.4, Table B.3; cited in 7.1, 7.3).

Validation of the scorer: run against the released B3 checkpoint on 120 test
plans it gives mIoU 0.9976 and Edge F1 0.9972, reproducing the full-split 0.9974
and 0.9977 within sampling noise.

Parameter counts above are trainable parameters. Counting batch-norm running
statistics as parameters (as an earlier version of the paper did) gives
47,236,304 / 59,452,560 / 3,718,768.

## Table 5 row 2 restated (added 2026-09-05)

The row-2 figures originally recorded above (0.9325 / 0.7385 / 1.0000 / 0.4514, from
`CORRECTED_GT_METHOD/rescore_result_OLD_M2_vs_corrected_GT.txt`, 2026-07-28) were computed
(a) on ground-truth masks, not predicted masks, and (b) against an earlier build of the
reference that had no open-passage category (5.91 door + 5.25 shared-wall edges/plan). Every
other table in the paper uses the current reference (6.07 door / 0.89 open / 4.49 shared-wall
edges per plan on the test split). Row 2 was therefore recomputed with
`src/rescore_table5.py` (one SegFormer-B3 pass per test plan; released defaults, including
`door_dilation_px=12`, which also reproduces `per_image.csv` plan for plan):

| Condition (predicted masks, n=2,567) | P | R | Edge F1 | type acc | door / open / shared |
|---|---|---|---|---|---|
| dilation-based M2 vs geometry-derived reference (row 2) | 0.9048 | 0.9982 | **0.9462** | **0.6943** | 0.9997 / 0.000 / 0.4322 |
| geometric M2 vs geometry-derived reference (row 3) | 0.9996 | 0.9961 | 0.9977 | 0.9997 | 0.9993 / 0.9817 / 0.9925 |

Row 3 reproduces the STEP-4 result exactly. The open recall of 0 in row 2 is structural: the
dilation-based M2 has no open-passage type, so every open edge of the reference counts as a
typing error against it. Means are per plan; pooled precision/recall are 0.8895/0.9981 (row 2)
and 0.9995/0.9956 (row 3). Per-plan values: `results/table5_rescore.csv` from the script.
