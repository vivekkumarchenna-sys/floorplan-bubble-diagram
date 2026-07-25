# MASTER CANONICAL VALUES

Extracted from codebase + latest run outputs. **Not** from manuscript/appendix.

Legend:
- ✅ verified from code or run output (file:line or file cited)
- ⬜ **NOT IN REPO** — must be measured or decided
- ⚠️ verified but contradicts README/manuscript, or needs a decision

Sources: `class_iou.csv`, `per_image.csv`, `summary.csv`, `Downloads/gt_upper_bound(1).csv`,
`Downloads/ablation_results.csv`, `history_segformer.json`, `history_deeplab.json`, `src/*.py`

---

## A. Segmentation (M1)

| Field | Value | Note |
|---|---|---|
| Per-image mIoU | **0.997434** (SD 0.008315, n=2100) | ✅ `per_image.csv` |
| Macro-averaged mIoU | **0.758513** | ✅ computed over active classes, `class_iou.csv` |
| Number of active classes | **13** | ✅ support > 0 |

⚠️ **These two mIoU numbers are not interchangeable.** 0.9974 is the mean of per-image mIoU;
0.7585 is the macro average over the 13 classes with non-zero support. The gap is caused by
Storage / Stair / Parking, which barely segment. **Decide which one the paper reports and use it
everywhere.** README currently uses 0.9974.

⚠️ **n=2100 but the test split is 2566.** Coverage is 82%. Either finish the run or state n=2100.

### Per-class metrics ✅ `class_iou.csv` (F1 computed as 2PR/(P+R))

| Class | Precision | Recall | F1 | IoU | Pixel Support |
|---|---|---|---|---|---|
| Background | 0.999931 | 0.999896 | 0.999914 | 0.999827 | 272,313,991 |
| Bedroom | 0.999461 | 0.999325 | 0.999393 | 0.999263 | 88,521,920 |
| Bathroom | 0.999785 | 0.999409 | 0.999597 | 0.999194 | 21,836,739 |
| Kitchen | 0.974570 | 0.974083 | 0.974327 | 0.973892 | 22,286,352 |
| Living | 0.999863 | 0.999814 | 0.999838 | 0.999677 | 83,537,191 |
| Balcony | 0.740832 | 0.741116 | 0.740974 | 0.740521 | 14,908,334 |
| Storage | 0.112221 | 0.112181 | 0.112201 | 0.112021 | 801,492 |
| Stair | 0.038794 | 0.036157 | 0.037429 | 0.035445 | 520,777 |
| Parking | 0.020745 | 0.020314 | 0.020527 | 0.019751 | 900,707 |
| Pool | 0.000000 | 0.000000 | 0.000000 | 0.000000 | **0** |
| Wall | 0.998116 | 0.999052 | 0.998584 | 0.997175 | 35,198,662 |
| Door | 0.997269 | 0.996893 | 0.997081 | 0.994202 | 3,480,378 |
| Window | 0.994751 | 0.994415 | 0.994583 | 0.993457 | 5,473,929 |
| FrontDoor | 0.998494 | 0.997260 | 0.997877 | 0.996245 | 721,928 |
| **Macro (13 active)** | **0.759603** | **0.759224** | **0.759410** | **0.758513** | — |

Class 14 (Column) and 15 (Other) also have zero support — excluded from macro, as is Pool.

### Backbone comparison

| Metric | SegFormer-B3 | DeepLabV3+ (R101) |
|---|---|---|
| Params (M) | ⬜ **NOT IN REPO** (README claims 47.3) | ⬜ **NOT IN REPO** (README claims 59.3) |
| FLOPs (G) | ⬜ **NOT IN REPO** | ⬜ **NOT IN REPO** |
| Val macro mIoU | **0.965113** ✅ | **0.968412** ✅ |
| Test per-image mIoU | **0.997434** ✅ | ⬜ never evaluated on test |
| GPU-hours | **2.72** ✅ (9808 s) | **5.40** ✅ (19450 s) |
| Inference (ms/img) | ⬜ **NOT IN REPO** | ⬜ **NOT IN REPO** |
| Training epochs | **27** ✅ | **50** ✅ |

⚠️ **The README backbone comparison is invalid.** It puts SegFormer's *test per-image* mIoU
(0.9974) against DeepLab's *val macro* mIoU (0.9684). On identical footing (val macro),
**DeepLab 0.9684 > SegFormer 0.9651**. Either evaluate DeepLab on test with the same metric, or
drop the claim that SegFormer wins on accuracy. GPU-hours (2.72 vs 5.40) remain a valid
SegFormer advantage.

⚠️ `Figures/fig_tables.py:166` renders a **FLOPs (G)** column with no computed source.

---

## B. Training Configuration

Source: `src/train.py` (CFG block, lines 60–98), `src/train_deeplab.py`

| Field | Value |
|---|---|
| Class weighting | **Uniform** ✅ `CLASS_WEIGHTS = None` (`train.py:90`) |
| λ_CE | **1.0** (implicit) ✅ |
| λ_Dice | **0.5** ✅ `DICE_WEIGHT` |
| Optimizer | **AdamW** ✅ |
| Learning rate | **6e-5** ✅ |
| Weight decay | **0.01** ✅ |
| Batch size | **8** (SegFormer) / **16** (DeepLab) ✅ |
| Max epochs | **100** (SegFormer) / **50** (DeepLab) ✅ |
| LR schedule | **CosineAnnealingLR** ✅ |
| T_max | **100** ✅ |
| η_min | **1e-7** ✅ |
| Early stopping patience | **15** ✅ |
| Monitor metric | **val mIoU** ✅ |
| Best validation mIoU | **0.965113** @ epoch **23** ✅ |
| Best validation loss | **0.068911** @ epoch **25** ✅ (mIoU there 0.961481) |
| Training stopped at epoch | **27** ⚠️ |
| Random seed | **42** ✅ |
| AMP | **Enabled, fp16** ✅ |
| Gradient clipping | **1.0** (max norm) ✅ |
| Ignore index | **255** ✅ |
| Dice epsilon | **1e-6** ✅ |
| Number of classes | **16** ✅ |
| Post-processing | ⚠️ **None — no post-processing stage exists** |
| Closing kernel | ⚠️ **N/A — not implemented** |
| Small-region threshold | ⚠️ **N/A — not implemented** |

⚠️ **Training did not early-stop.** Best epoch 23, patience 15 → would have run to epoch 38.
Log ends at 27 with `EPOCHS=100`. **Interrupted, not converged.** Do not describe as early-stopped
or converged.

⚠️ **Cosine schedule never completed.** `T_max=100` but training stopped at 27, so LR traversed
only ~27% of the curve and never approached η_min.

⚠️ **Docstring says "weighted CrossEntropy"** (`train.py:7`) but weights are `None`. Docstring is
wrong; code is uniform.

⚠️ **No post-processing module exists anywhere.** The ablation variant named
"w/o post-processing" actually ablates graph-construction thresholds — see section I.

---

## C. Data Augmentation

Source: `src/dataset.py:_build_train_transforms` — applied in this order

| Operation | Probability | Parameters |
|---|---|---|
| HorizontalFlip | 0.5 | — |
| VerticalFlip | 0.5 | — |
| RandomRotate90 | 0.5 | — |
| RandomResizedCrop | 1.0 | 512×512, scale=(0.64, 1.44), ratio=(0.75, 1.333), INTER_LINEAR |
| GaussNoise | 0.5 | var_limit=(10.0, 50.0) |
| CoarseDropout | 0.3 | max_holes=12, min_holes=4, height 16–64, width 16–64, fill_value=0 |
| Normalize | 1.0 | mean=(0.485,0.456,0.406) std=(0.229,0.224,0.225) |
| ToTensorV2 | 1.0 | — |

Val/test pipeline: `Resize(512,512, INTER_LINEAR)` → `Normalize` → `ToTensorV2`.

⚠️ **CoarseDropout may be corrupting labels.** `fill_value=0` with no `mask_fill_value` set. The
inline comment claims it "keeps original mask values under the hole", but Albumentations' default
also fills the mask. Verify against the installed version — if masks are being zeroed, 30% of
training samples carried corrupted labels.

---

## D. Dataset

| Field | Value |
|---|---|
| Total plans | **17,104** (split files) ⚠️ |
| Stratification method | ⚠️ **None — no splitting code in repo**; `splits/*.txt` are pre-existing flat lists |
| Rasterization resolution | **512 × 512** ✅ |
| Approximate door size | ⬜ **derived estimate ~130 px area/instance** — not measured |
| Approximate window size | ⬜ **NOT IN REPO** |

| Split | Plans | Room Instances | Door Instances | Mean Rooms | Mean Doors |
|---|---|---|---|---|---|
| Train | **11,973** ✅ | ⬜ | ⬜ | ⬜ | ⬜ |
| Validation | **2,565** ✅ | ⬜ | ⬜ | ⬜ | ⬜ |
| Test | **2,566** ✅ | ⬜ | ⬜ | ⬜ | ⬜ |
| Total | **17,104** ✅ | ⬜ | ⬜ | ⬜ | ⬜ |

**To fill the ⬜ cells:** run `python src/compute_split_stats.py`. The script exists and computes
exactly these values, but its output is not saved anywhere in the repo.

Available proxy from `per_image.csv` (test, n=2100): mean rooms/plan **8.185** (GT) vs **8.192**
(pred); mean edges/plan **12.819** (GT) vs **12.905** (pred). These are graph nodes/edges, not
raw room/door instance counts — do not substitute them for the table above.

⚠️ **Three conflicting dataset totals:** 17,107 mask files · 17,104 split entries · 17,000
ResPlan.pkl entries. Reconcile before quoting a dataset size.

---

## E. Graph Construction (M2)

Source: `src/build_graph.py:178` `build_graph_from_segmentation`

| Field | Value |
|---|---|
| Minimum room area | **100 px** ✅ |
| Dilation radius | **15 px** ✅ |
| Kernel size | **31 × 31** (`MORPH_RECT`, `2*15+1`) ✅ |
| Iterations | **1** ✅ |
| Door overlap threshold | **15 px** (`door_min`) ✅ |
| Shared-wall threshold | **20 px** overlap (`wall_min`) ✅ |
| Arch threshold | **30 px** opening width (`arch_min`) ✅ |
| Pruning threshold | ⚠️ **N/A — no pruning implemented** |
| Pruning applied? | **N** ✅ |

Edge weights (`src/proximity.py:39`):
- Door = **1.0** ✅
- Arch = **0.8** ✅
- Shared-wall = **0.3** ✅

**Corridor edges present? — N** ✅

> **"No corridor-mediated edges are implemented in the final pipeline."**

Only first-order pairwise adjacency between dilated room masks exists. No second-order or
transitive edges.

⚠️ **There is no Corridor class in the 16-class schema.** `run_ablation.py:80` targets
`class_id 6`, commented `# Corridor/Storage class` — class 6 is **Storage**. The variant labelled
"w/o corridor adj." removes Storage-room edges. See section I.

Ground-truth graph parameters (`src/build_gt_graph.py:202`):
- Adjacency buffer = **15.0** ✅
- Door buffer = **5.0** ✅
- Door overlap minimum = **15.0** px² ✅
- Shared-wall minimum = **20.0** px ✅
- Arch minimum length = **30.0** px ✅

---

## F. Proximity Matrix (M3)

| Field | Value |
|---|---|
| Matrix construction | Symmetric N×N, zero diagonal, nodes sorted by id. `matrix[i,j] = weight[edge_type]`, else 0. Not weighted by distance or area ✅ |
| Hierarchical variant? | **N** ✅ — not implemented |
| Frobenius formula | `‖P − G‖_F`, smaller matrix zero-padded to `n = max(n_pred, n_gt)` ✅ |
| Normalization method | **`frobenius / n`** ✅ (`evaluate.py:292`) |
| Edge-type accuracy — matched only? | **Y** ✅ — `correct_type / tp`, 0.0 when tp=0 |
| GED method | `nx.optimize_graph_edit_distance`, class-aware: same `class_name` → 0 else 1.0; same `edge_type` → 0 else 1.0. Anytime algorithm, returns best upper bound at cutoff ✅ |
| Timeout | **2.0 s** (`inference.py`) / 5.0 s (`run_ablation.py` default) ✅ |
| Edge matching rule | Canonical sorted pair of `"ClassName[node_id]"` labels ✅ |

⚠️ **GED must be described as an approximate upper bound with a 2 s cutoff**, not converged GED.

⚠️ **The edge matching rule is almost certainly capping Edge F1.** Keys embed the integer instance
id, which is assigned by iteration order over classes then connected components. One extra or
missing room shifts every later id and invalidates all of its edges. Evidence: the GT-mask upper
bound (section H) is only 0.0019 above the full pipeline — perfect segmentation changes nothing,
which is what you would expect if the ceiling is imposed by the matching rule rather than by
perception. This also explains why "w/o room types" collapses to exactly 0.0: relabelling every
node `"Room"` destroys every key.

**Recommended:** report an id-independent matching variant (e.g. bipartite matching on class +
centroid) alongside the current number.

---

## G. Bubble Diagram (M4)

| Field | Value |
|---|---|
| Layout algorithm | `nx.spring_layout` (Fruchterman–Reingold), `seed=42`, `k = 2.0/√N` ✅ |
| Alternative evaluated | ⚠️ **None** |
| Node radius rule | ⚠️ **Per-image min-max normalised area** → `node_size` 300–3000 pt² (`visualize.py:129-133`) |
| Node colour rule | `ROOM_COLORS[class_name]` lookup ✅ |
| Edge style mapping | solid = door · dashed = arch · dotted = shared-wall ✅ |

⚠️ **Do not claim bubble area is proportional to room area.** Per-image min-max means the largest
room always renders at 3000 pt² and the smallest at 300 pt², whatever their true sizes. Bubbles
encode within-plan rank, not area, and are not comparable across diagrams.

⚠️ **The m² labels on the diagrams are wrong.** `pixel_scale.json` was built with an incorrect
stem→plan mapping; areas are inflated ~2.5x (bathrooms render at 12–15 m² instead of 3–6 m²).
Regeneration verified: correct mapping is **by `id` field, not list index** (57/60 fingerprint
match), scale = `net_area / interior_px`. Affects `Output/*.png`, survey stimuli, Fig 6, Fig 7.

⚠️ **Survey Q3 asks participants to judge whether room sizes are proportional** — against diagrams
where sizes are neither proportional nor correctly scaled. This invalidates that dimension
regardless of what data is eventually collected.

---

## H. Full Test-Set Results

### Proposed Method ✅ `per_image.csv` (n=2100)

| Metric | Value |
|---|---|
| Edge Precision | **0.6574** (SD 0.1580) |
| Edge Recall | **0.6606** (SD 0.1571) |
| Edge F1 | **0.6589** (SD 0.1574) |
| GED Mean | **4.2024** |
| GED SD | **5.4352** |
| Edge-Type Accuracy | **0.7853** (SD 0.1954) |
| Normalized Frobenius | **0.4625** (SD 0.1126) |
| Raw Frobenius | 3.7904 (SD 1.2760) |

### GT-Mask Input (M2–M4 only) ✅ `Downloads/gt_upper_bound(1).csv` (n=2567)

| Metric | Value |
|---|---|
| Edge Precision | **0.6594** (SD 0.1547) |
| Edge Recall | **0.6624** (SD 0.1541) |
| Edge F1 | **0.6608** (SD 0.1543) |
| GED Mean | **4.1617** |
| GED SD | **5.3822** |
| Edge-Type Accuracy | **0.7870** (SD 0.1938) |
| Normalized Frobenius | ⚠️ **ALL NaN — bug, see below** |

### ⚠️ Key finding

**Upper bound Edge F1 (0.6608) exceeds the pipeline (0.6589) by only 0.0019.** Perfect
segmentation yields essentially no improvement. **Effectively all graph error originates in
M2–M4, not M1.** This is a genuine and reportable result, but it inverts the usual framing —
the segmentation is not the bottleneck; graph extraction and/or edge matching is.

⚠️ **Sample sizes differ:** pipeline n=2100, upper bound n=2567. Re-run the pipeline on the full
test set before presenting these side by side.

⚠️ **Frobenius bug:** `eval_gt_upper_bound.py:113` reads column `frob_normalized`; `evaluate.py:296`
returns `normalized` → `KeyError` → swallowed by the bare `except` at line 114 → NaN for every row.

⚠️ **Hardcoded value:** `eval_gt_upper_bound.py:148` sets `pipeline_ef1 = 0.6685`, matching no
actual run. Its printed "error from M1 / M2–M4" attribution is therefore wrong.

⚠️ **Edge F1 exists in three conflicting values across the project:**
`0.6589` (full test, n=2100) · `0.6956` (ablation, n=100) · `0.6685` (hardcoded).
**Pick one canonical value and propagate.**

---

## I. Ablation

**Number of images: 100** ✅ (`n_images` column; script default is 300, so it ran with `--limit 100`)

| Variant | mIoU | Edge F1 | GED | Normalized Frobenius | time (s) |
|---|---|---|---|---|---|
| Full | 0.9978 | 0.6956 | 3.54 | 0.4415 | 46.8 |
| w/o Door | 0.9978 | 0.6956 | 12.72 | 0.4957 | 37.7 |
| w/o Room Types | 0.9978 | 0.0000 | 11.94 | 0.4415 | 172.3 |
| w/o Corridor | 0.9978 | 0.6888 | 3.66 | 0.4443 | 57.0 |
| w/o Post-processing | 0.9978 | 0.0000 | 12.46 | 0.6112 | 9.4 |
| w/o Edge Typing | 0.9978 | 0.6956 | 12.74 | 0.6120 | 138.7 |

**"Without post-processing" changes:** `dilation_px=0, door_min=1, wall_min=1, arch_min=1,
min_room_area=10` (`run_ablation.py:89`).

⚠️ **Misnomer — rename.** There is no post-processing stage in the pipeline. This variant ablates
*graph-construction thresholding*, primarily the 15 px dilation. A reviewer will ask to see the
post-processing code and find none.

⚠️ **"w/o Corridor" does not test corridors.** It removes edges touching class 6 = **Storage**,
whose IoU is 0.112 (barely segments), which is why the effect is a negligible 0.0068.

⚠️ **Three variants share Edge F1 = 0.6956** (full / w/o door / w/o edge typing). Expected —
`edge_metrics` scores topology only, and those variants alter `edge_type`, which affects type
accuracy and GED, not F1. But Fig 4 plots Edge F1, so three bars are identical by construction and
carry no signal.

⚠️ **Ablation mIoU (0.9978) ≠ full-test mIoU (0.9974)** — different subset (n=100 vs n=2100).

---

## J. User Study

# ⛔ NO DATA EXISTS — SECTION CANNOT BE COMPLETED

`Downloads/user_study_workbook.xlsx` is an **unfilled template**:
- `Raw_Ratings`: **0 of 1600** rating cells populated
- `Participant_Info`: placeholder strings only (`"P01, P02, …"`, `"Architect / Student"`, `"Integer"`)
- `Table6_Output`: Excel formulas, all resolving to `"—"`
- `Summary_Stats`: annotated *"U Statistic and p-value must be computed externally"* — never done
- No `scipy.stats`, `pingouin`, ICC, or Kendall's W code anywhere in the repo

| Field | Value |
|---|---|
| Participants | ⛔ **NONE COLLECTED** (workbook provisions for 20) |
| Participant type | ⛔ no data (template says "Architect / Student") |
| Experience level | ⛔ no data |
| Mean familiarity | ⛔ no data |
| Pairs shown | **20 stimuli** designed ✅ `survey_stimuli.csv` |
| Pipeline pairs | **10** ✅ |
| Ground-truth pairs | **10** ✅ |
| Statistical test | **Mann-Whitney U** (planned) ✅ template |
| Alpha | **0.05** (planned) ✅ template |
| Effect size | **rank-biserial r** (planned) ✅ template |

| Dimension | Pipeline Mean±SD | GT Mean±SD | U | p | Effect Size |
|---|---|---|---|---|---|
| Plausibility | ⛔ | ⛔ | ⛔ | ⛔ | ⛔ |
| Adj_Correctness | ⛔ | ⛔ | ⛔ | ⛔ | ⛔ |
| RoomType_Accuracy | ⛔ | ⛔ | ⛔ | ⛔ | ⛔ |
| Readability | ⛔ | ⛔ | ⛔ | ⛔ | ⛔ |
| *(5th dimension)* | ⚠️ see note | | | | |

**Reliability:** Overall ICC = ⛔ · Overall Kendall W = ⛔ · Per-dimension = ⛔

The workbook provisions for **Cohen's Kappa**, not ICC or Kendall's W. If the manuscript reports
ICC or Kendall's W, neither the instrument nor any code supports them.

⚠️ **Dimension count mismatch:** the workbook defines **4** dimensions (Plausibility,
Adj_Correctness, RoomType_Accuracy, Readability) but the questionnaire has **5** questions. Q5
(overall usefulness) has no corresponding row in `Raw_Ratings` or `Summary_Stats`. Resolve before
collecting data.

### Survey questions ✅ `src/generate_survey_docx.py:36`

| # | Text | Maps to dimension |
|---|---|---|
| Q1 | "The bubble diagram accurately represents the rooms in the floor plan." | Plausibility |
| Q2 | "The room adjacencies (connections) are correctly captured." | Adj_Correctness |
| Q3 | "The room sizes in the diagram are proportional to the actual plan." | RoomType_Accuracy *(mismatch — see below)* |
| Q4 | "The diagram is easy to read and interpret." | Readability |
| Q5 | "Overall, I would rate this bubble diagram as useful for understanding the layout." | ⚠️ **no dimension row** |

Scale: 1 (Strongly Disagree) → 5 (Strongly Agree).

⚠️ **Q3 asks about size proportionality but maps to a dimension named RoomType_Accuracy.** The
question and the dimension measure different things. Q3 is also unanswerable as designed —
bubble sizes are rank-normalised, not proportional (section G).

**Decision required:** collect the study, or remove all user-study claims from the manuscript.
This is a paper-structure decision, not a code fix.

---

## K. Schema

Source: `src/build_graph.py:50` (authoritative — matches `train.py` and README)

| Class | ID | Role |
|---|---|---|
| Background | 0 | ignored |
| Bedroom | 1 | **room node** |
| Bathroom | 2 | **room node** |
| Kitchen | 3 | **room node** |
| Living | 4 | **room node** |
| Balcony | 5 | **room node** |
| Storage | 6 | **room node** |
| Stair | 7 | **room node** |
| Parking | 8 | **room node** |
| Pool | 9 | **room node** (zero support in data) |
| Wall | 10 | boundary — subtracted when measuring opening width |
| Door | 11 | **door trigger** |
| Window | 12 | ignored |
| FrontDoor | 13 | **door trigger** |
| Column | 14 | ignored (zero support) |
| Other | 15 | ignored (zero support) |

- **Room-node IDs: {1, 2, 3, 4, 5, 6, 7, 8, 9}** ✅ `ROOM_CLASSES`
- **Door-trigger IDs: {11, 13}** ✅ `DOOR_CLASSES`

⚠️ **`build_gt_graph.py:27-33` docstring declares a completely different schema** — "1 LivingRoom,
2 Bedroom, 4 Bathroom, 6 Corridor, 7 Dining, 9 Garage, 13 Staircase". The executable dict at
line 51 is correct and matches the table above. **The docstring is stale and must be deleted** —
it is the likeliest artifact to mislead a reviewer reading the source, and it is the origin of the
phantom "Corridor" class that appears in the ablation naming.

---

## L. References

| Field | Value |
|---|---|
| RPLAN citation number | ⬜ **NOT IN CODEBASE** — check manuscript |
| Duplicate references | ⬜ **NOT IN CODEBASE** — check manuscript |

No bibliography exists in the repo. `CITATION.cff` cites only this software.

⚠️ `README.md:54` links the dataset to `https://github.com/ResPlanProject`, which does not
resolve to a dataset paper. Verify this points to the correct ResPlan/RPLAN citation — note that
**ResPlan and RPLAN are different datasets**; confirm which one is actually used before citing.

---

## M. Environment

⚠️ `requirements.txt` gives **minimum floors, not installed versions**. None of these are
reproducible as stated. **Run `pip freeze` in the Colab that produced the results.**

| Package | Declared floor | Actual |
|---|---|---|
| Python | >=3.10 (README) | ⬜ |
| PyTorch | >=2.0 | ⬜ |
| Transformers | >=4.30 | ⬜ |
| OpenCV | >=4.7 | ⬜ |
| NetworkX | >=3.1 | ⬜ |
| Shapely | >=2.0 | ⬜ |
| Albumentations | >=1.3 | ⬜ **critical — see C** |
| NumPy | >=1.24 | ⬜ |
| SciPy | not declared | ⬜ |
| pandas | >=2.0 | ⬜ |
| matplotlib | >=3.7 | ⬜ |
| scikit-image | >=0.21 | ⬜ |
| Pillow | >=9.0 | ⬜ |
| python-docx | >=0.8 | ⬜ |
| tqdm | >=4.60 | ⬜ |

Albumentations matters most: `CoarseDropout`'s mask-fill behaviour changed across 1.x and
determines whether training labels were corrupted (section C).

---

## Outstanding decisions before the consistency sweep

| # | Decision | Blocks |
|---|---|---|
| 1 | Per-image (0.9974) or macro (0.7585) mIoU as the headline? | Abstract, Table 2, Table 3, Fig 3 |
| 2 | User study: collect it, or remove all claims? | Table 6, §User Study, reviewer response |
| 3 | Canonical Edge F1: 0.6589 / 0.6956 / 0.6685 | Abstract, Table 4, Table 5, Fig 4 |
| 4 | Re-run pipeline on full test set (2100 → 2566)? | Table 4, upper-bound comparison |
| 5 | Fix or drop the backbone comparison | Table 3 |
| 6 | Rename "w/o post-processing" and "w/o corridor" | Table 5, Fig 4, §Ablation |
| 7 | Regenerate `pixel_scale.json` + re-render diagrams | Fig 6, Fig 7, survey stimuli |
| 8 | Report an id-independent edge-matching variant? | Table 4, §Limitations |

## Code fixes (independent of the above)

1. `eval_gt_upper_bound.py:113` — `frob_normalized` → `normalized`
2. `eval_gt_upper_bound.py:114` — replace bare `except` so failures surface
3. `eval_gt_upper_bound.py:148` — remove hardcoded `pipeline_ef1 = 0.6685`
4. `build_gt_graph.py:27-33` — delete stale schema docstring
5. `run_ablation.py:80` — fix `# Corridor/Storage` comment and variant name
6. `train.py:7` — docstring says "weighted CrossEntropy"; weights are `None`
7. `dataset.py:68-77` — verify `CoarseDropout` mask behaviour
