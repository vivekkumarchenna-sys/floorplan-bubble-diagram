# Floor Plan Bubble Diagram Generator

Automatically convert architectural floor plan images into **bubble diagrams**, abstract graph representations showing rooms, their connections, and spatial relationships.

```
Floor Plan Image  →  Semantic Segmentation  →  Room Graph  →  Bubble Diagram
```

## Pipeline

| Step | Script | Description |
|------|--------|-------------|
| 1 | `src/dataset.py` | PyTorch dataset with albumentations augmentation |
| 2 | `src/train.py` | SegFormer-B3 training (16-class semantic segmentation) |
| 3 | `src/train_deeplab.py` | DeepLabV3+ alternative training script |
| 4 | `src/truegraph_builder.py` | **Module M2 (current): geometry-based typed connectivity graph - door / open passage / shared-wall - built directly from the mask; the same `build_true_graph` procedure produces both the geometry-derived reference and the pipeline graph** |
| 5 | `src/render_bubble.py` | **M2/M3/M4 as a reusable module: the typed graph as a NetworkX graph, the categorical typed adjacency matrix, and the geographic bubble diagram of Section 5.4 (nodes at room centroids, marker area proportional to floor area, node colour sampled from the raster)** |
| 6 | `src/build_graph.py` | Legacy dilation-based graph extraction (kept for the ablation and comparison) |
| 6 | `src/build_gt_graph.py` | Reads ResPlan's own released graph; used only for the dataset-reliability audit, not as ground truth |
| 7 | `figures/typed_proximity.py`, `figures/adj_matrix.py` | Categorical adjacency views (no weights). Note the two file names are historical and read the wrong way round: `typed_proximity.py` draws the square adjacency matrix and `adj_matrix.py` draws the diamond-lattice proximity chart. `figures/fig5_tables.py` draws both as one figure. `src/proximity.py` is the legacy weighted version |
| 8 | `src/visualize.py` | Bubble diagram visualisation; geographic bubble diagrams (rooms at their plan positions) are rendered by the `figures/` scripts |
| 9 | `src/evaluate.py` | Evaluation metrics (mIoU, edge F1, edge-type accuracy, GED) |
| 10 | `src/build_corrected_gt.py`, `src/step3_resplan_mismatch.py`, `src/step4_rescore_newM2.py` | Reproduce the geometry-derived reference, the ResPlan reliability audit, and the re-scored results |
| 9 | `src/inference.py` | Batch evaluation on test set with GT comparison |
| 10 | `src/generate_bubble.py` | End-to-end: image → typed bubble diagram. Uses `render_bubble` (the pipeline the paper reports) by default; `--legacy` selects the pre-rework dilation-based M2, weighted proximity matrix and force-directed rendering |
| 11 | `src/run_ablation.py` | Automated 6-variant ablation study |
| 12 | `src/generate_survey.py` | Generate paired stimuli for user survey |
| 12b | `src/user_study_stats.py` | Reproduce the user-study table (paper Table E.4). The raters scored both conditions, so the test is a **Wilcoxon signed-rank** on participant-level means, not Mann-Whitney over individual rating cells |
| 13 | `src/generate_survey_docx.py` | Create survey as Word document |
| 14 | `src/compute_split_stats.py` | Dataset partition statistics |
| 15 | `src/probe_learned_edge_typing.py` | Feature extraction for an exploratory learned-edge-typing probe (not reported in the paper) |
| 16 | `src/train_probe_classifier.py` | Trains and selects the probe's gradient-boosted classifier (exploratory) |
| 17 | `src/benchmark_runtime.py` | Per-module runtime/GPU-memory benchmark (paper Appendix B.3) |
| 18 | `src/recompute_per_class_presence.py` | Per-class presence/precision/recall/IoU recomputation from a saved evaluation run |

## Segmentation Classes

| ID | Class | ID | Class |
|----|-------|----|-------|
| 0 | Background | 8 | Parking |
| 1 | Bedroom | 9 | Pool |
| 2 | Bathroom | 10 | Wall |
| 3 | Kitchen | 11 | Door |
| 4 | Living | 12 | Window |
| 5 | Balcony | 13 | FrontDoor |
| 6 | Storage | 14 | Column |
| 7 | Stair | 15 | Other |

## Setup

### Requirements

- Python 3.10+
- CUDA-compatible GPU (for training / fast inference)

```bash
pip install -r requirements.txt
```

`albumentations==1.3.1` is pinned exactly and must stay pinned. `dataset.py` uses
`RandomResizedCrop` with an area scale of `(0.64, 1.44)`; values above 1.0 are accepted at 1.3.1
but rejected from 1.4 onward, so a newer release raises a validation error instead of reproducing
the paper's augmentation. Everything else in `requirements.txt` is a minimum version.

### Data

This project uses the [ResPlan dataset](https://github.com/ResPlanProject), rasterised residential floor plans with per-pixel semantic labels. See [`data/README.md`](data/README.md) for setup instructions.

### Model Checkpoint

Download the trained checkpoints and place them in `checkpoints/`. See [`checkpoints/README.md`](checkpoints/README.md) for download links.

## Usage

### Generate Bubble Diagrams

The default path is the pipeline the paper reports: geometry-based M2 (`truegraph_builder.build_true_graph`), the categorical typed adjacency matrix, and the geographic bubble diagram of Section 5.4. Pass `--legacy` to run the pre-rework dilation-based graph with the weighted proximity matrix and the force-directed rendering instead — that is the version the user-study stimuli of Section 7.3 were drawn with, and it is not the pipeline the reported numbers come from.

```bash
# single image
python src/generate_bubble.py --image path/to/floorplan.png

# folder of images
python src/generate_bubble.py --image path/to/images/ --limit 20

# custom checkpoint and output directory
python src/generate_bubble.py --image img.png --ckpt path/to/model.pth --out output/
```

### Use in Python / Jupyter

```python
from src.generate_bubble import BubbleGenerator

gen = BubbleGenerator("checkpoints/segformer/best_model.pth")

# display inline
gen.show("path/to/floorplan.png")

# or get the graph and matrix
fig, G, matrix, labels = gen.generate("path/to/floorplan.png")
```

### Google Colab

```python
from google.colab import drive
drive.mount('/content/drive')

import sys
sys.path.insert(0, "/content/drive/MyDrive/bubble_diagram_project")

from generate_bubble import BubbleGenerator

gen = BubbleGenerator("checkpoints/segformer/best_model.pth")
gen.show("data/resplan_raster/42.png")
```

### Train a Model

```bash
# SegFormer-B3
python src/train.py

# DeepLabV3+ (alternative)
python src/train_deeplab.py
```

### Evaluate on Test Set

```bash
# full test set (GED timeout defaults to 3.0 s/image)
python src/inference.py

# faster, but fewer graphs converge (lower GED timeout)
python src/inference.py --ged-timeout 2
```

Inference supports batch checkpointing (saves every 300 images) and resume on crash.

GED is expensive and is computed with a per-image timeout (default **3.0 s** in `src/inference.py` and `src/recompute_ged_parallel.py`; **5.0 s** in `src/run_ablation.py`). A **lower** timeout is faster but lets **fewer** graphs converge; a higher one is slower but converges more. The reported GED is a mean over only the subset of plans (n/2567 on the full test set) whose edit distance converged within the cutoff, so the value is both timeout- and machine-dependent and is not directly comparable across different timeouts or hardware.

### Ablation Study

```bash
# run all 6 variants (default: 300 images per variant, GED timeout 5.0 s)
python src/run_ablation.py

# fewer images, lower GED timeout, for speed
python src/run_ablation.py --limit 100 --ged-timeout 3
```

### User Survey

```bash
# generate paired stimuli (pipeline vs GT)
python src/generate_survey.py

# create Word document survey
python src/generate_survey_docx.py
```

## Evaluation Metrics

| Metric | What it measures |
|--------|-----------------|
| **mIoU** | Pixel-level segmentation accuracy (per-class IoU) |
| **Edge F1** | Graph edge detection (precision/recall of room connections) |
| **Type Accuracy** | Correct edge classification (door / open passage / shared-wall) |
| **GED** | Graph edit distance (structural similarity) |

## Graph Edge Types

The typed graph uses three categorical edge types. There are no edge weights: every room pair is exactly one of these types, or unconnected.

| Type | Meaning |
|------|---------|
| `door` | Two rooms joined by a doorway (interior door, class 11) |
| `open passage` | Two rooms meeting through a gap in the wall, with no door |
| `shared-wall` | Two rooms separated by a continuous wall, with no opening |

These are read from raster geometry by `src/truegraph_builder.py` (`build_true_graph`). "Open passage" replaces the earlier `arch` category: ResPlan's own graph has no interior-room instance of its `arch`/`direct` relation (all 16,964 `direct` edges in the release connect a room to the building's front-door node, never two interior rooms), so `arch` has no evaluable ground truth in this dataset. The geometry-derived reference is built by the same `build_true_graph` procedure applied to the ground-truth mask; `src/build_gt_graph.py` (which reads ResPlan's own released graph) is retained only for the dataset-reliability audit of the paper, not as the ground truth.

## Results

### Backbone Comparison

Both models are compared on the same metric - **validation macro mIoU** (mean over classes). DeepLabV3+ is marginally higher (0.9684 vs 0.9651).

| Method | Params (M) | Val mIoU (macro) | GPU-hours |
|--------|-----------|------------------|-----------|
| SegFormer-B3 | 47.2 | 0.9651 | ~3 (A100) |
| DeepLabV3+ (R101) | 59.3 | 0.9684 | ~5.4 (A100) |
| SegFormer-B0 | 3.7 | 0.9602 | ~3.1 (RTX 5060 laptop) |

Parameter counts are trainable parameters, measured from the released checkpoints: 47,234,768 (SegFormer-B3) and 59,343,024 (DeepLabV3+ R101). Batch-normalisation running statistics are buffers, not parameters, and are excluded; counting them instead gives 47,236,304 and 59,452,560. An earlier version of this table quoted SegFormer's *test per-image* mIoU (0.9974) against DeepLab's *validation macro* mIoU - not the same metric - and listed FLOPs / inference-time figures that are not measured anywhere in this repository.

### Lighter backbone

SegFormer-B0 is 12.7x smaller than B3 and segments measurably worse, but recovers the same graph
(`src/eval_backbone.py`, full 2,567-plan test split, M2 vs the geometry-derived reference):

| Backbone | Trainable params | Test per-image mIoU | Edge F1 | Edge-type acc |
|---|---|---|---|---|
| SegFormer-B3 | 47,234,768 | 0.9974 | 0.9977 | 0.9997 |
| SegFormer-B0 | 3,718,256 | 0.9928 | 0.9971 | 0.9996 |

Segmentation accuracy is not the limiting factor on these plans.

### Ablation Results

Ablation of the three geometry-based M2 rules, each disabled in turn, scored against the geometry-derived reference over an 800-plan sample (`figures/fig4_ablation.py`, matching paper Table 6 / Fig 4):

| Variant (disables one rule) | Effect vs geometry-derived reference |
|---|---|
| Full pipeline | Edge F1 1.00, edge-type accuracy 1.00 |
| w/o door precedence | edge-type accuracy drops to 0.61, door recall to 0.26 |
| w/o open-passage detection | open-passage recall drops to 0.00 |
| w/o multi-room hub | door recall drops to 0.97 |

Door precedence is the dominant rule: without it, a door adjacent to several rooms is no longer resolved to the correct pair, and both typing accuracy and door recall collapse. The full per-variant numbers are in paper Table 6.

### Paper Figures

All figures are generated by the Python scripts below and saved as vector PDFs with embedded TrueType fonts (`pdf.fonttype = 42`), so re-running a script reproduces the submitted artwork byte for byte:

| Figure | Script | Description |
|--------|--------|-------------|
| Fig 1 | `figures/fig1_pipeline.py` | Pipeline overview: raster input, predicted label map, typed adjacency matrix, typed bubble diagram |
| Fig 2 | `figures/fig_training.py` | Training vs validation loss curve (from `history_segformer.json`) |
| Fig 3 | `figures/fig_training.py` | Training vs validation mIoU curve (from `history_segformer.json`) |
| Fig 4 | `figures/fig4_ablation.py` | Ablation bar chart (new-rules M2, scored vs geometry-derived reference) |
| Fig 5 | `figures/fig5_tables.py` | The two tabular views of one typed graph: proximity chart (a) and typed adjacency matrix (b), plan 13388 |
| Fig 6 | `figures/fig6_combined.py` | Qualitative: raster + typed bubble diagram for sample plans |
| Fig 7 | `figures/fig7_mismatch.py` | ResPlan reliability: geometry-derived reference vs ResPlan's released graph |

```bash
# generate figures (run from the repo root; Fig 2/3 need no GPU)
python figures/fig1_pipeline.py
python figures/fig_training.py
python figures/fig4_ablation.py
python figures/fig5_tables.py Fig5.pdf 13388
python figures/fig6_combined.py
python figures/fig7_mismatch.py
```

## Project Structure

```
floorplan-bubble-diagram/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
│
├── src/                        # pipeline scripts
│   ├── dataset.py
│   ├── train.py
│   ├── train_deeplab.py
│   ├── truegraph_builder.py    # Module M2 (current): geometry-based typed graph
│   ├── render_bubble.py        # M2/M3/M4 as a module: typed graph, categorical matrix, geographic diagram
│   ├── build_graph.py          # legacy dilation-based graph (ablation/comparison)
│   ├── build_gt_graph.py       # reads ResPlan's own graph (reliability audit only)
│   ├── proximity.py            # legacy weighted matrix
│   ├── visualize.py
│   ├── evaluate.py
│   ├── inference.py
│   ├── generate_bubble.py
│   ├── build_corrected_gt.py       # build the geometry-derived reference
│   ├── step3_resplan_mismatch.py   # ResPlan reliability audit
│   ├── step4_rescore_newM2.py      # re-score the pipeline vs geometry-based GT
│   ├── step4_confusion.py
│   ├── ablation_newrules.py
│   ├── run_ablation.py
│   ├── compute_split_stats.py
│   ├── generate_survey.py
│   ├── user_study_stats.py      # paired Wilcoxon analysis of the user study
│   ├── generate_survey_docx.py
│   ├── generate_survey_pdf.py
│   ├── benchmark_runtime.py
│   └── recompute_per_class_presence.py
│
├── figures/                    # figure generation scripts + PDFs
│   ├── fig1_pipeline.py
│   ├── fig_training.py
│   ├── fig4_ablation.py
│   ├── fig5_tables.py
│   ├── typed_proximity.py
│   ├── fig6_combined.py
│   ├── fig7_mismatch.py
│   ├── adj_matrix.py
│   ├── bubble_only.py
│   └── plan_to_bubble.py
│
├── RESULTS_SUMMARY.md          # every reported number, with provenance
│
├── output/                     # generated bubble diagrams (created at run time)
│
├── checkpoints/                # model weights (not in repo)
│   └── README.md
│
└── data/                       # dataset (not in repo)
    └── README.md
```

## The ground-truth reliability finding

Scoring the pipeline exposed a defect in the dataset. ResPlan distributes a precomputed typed room-adjacency
graph, but auditing it against the rasters over all 17,107 plans shows it types about 40% of the interior
doors in its own vector geometry as shared-wall, and provides no interior-room instance of its `arch` class.
A pipeline scored against that annotation is charged for the annotation's errors (Edge F1 0.51). The paper
therefore builds a geometry-derived reference directly from the raster with `src/truegraph_builder.py`
(`build_true_graph`), validated against an architect's reading. Against it the same pipeline recovers doors
(recall 1.0) and topology (Edge F1 0.93), and the deterministic geometric construction reproduces the graph
end-to-end at Edge F1 0.998 - a robustness check, since the rules define the reference. The reliability
audit and the re-scoring are reproduced by `src/build_corrected_gt.py`, `src/step3_resplan_mismatch.py`
and `src/step4_rescore_newM2.py`; `RESULTS_SUMMARY.md` documents every reported number.

## Limitations

The M2 rules are deterministic and hand-set for the 512 px raster convention used here; they are chosen so
the graph-construction stage is fully attributable and its failures diagnosable, and learned relational
reasoning is expected to supersede them. Room correspondence between predicted and ground-truth graphs is by
class and centroid, which is robust when segmentation is near-perfect (as here). The residual error is a
tendency of proximity typing to over-report shared walls beside doorways, which the geometry-based rules
largely, but not entirely, remove.

## Trained weights and full archive

The trained SegFormer-B3 and DeepLabV3+ checkpoints are too large for this repository (541 MB and 678 MB on disk, above GitHub's 100 MB per-file limit). A complete archive of the project, including both checkpoints, the rasterised inputs, and the user-study materials, is available at:

https://doi.org/10.5281/zenodo.21600699

The ResPlan files in that archive are redistributed for reproducibility only. ResPlan is released by its authors under a permissive open-source licence and their release remains the canonical source.

## License

The code in this repository is released under the MIT License; see [LICENSE](LICENSE). This covers the code only. The ResPlan dataset and any files derived from it remain subject to the licence granted by the ResPlan authors.

## Citation

If you use this code, please cite:

```bibtex
@software{floorplan_bubble_diagram,
  title={Typed Bubble Diagrams from Floor Plan Images: A Reproducible Baseline and Evaluation Protocol},
  author={Chenna, Vivek Kumar and P, Bimal},
  url={https://github.com/vivekkumarchenna-sys/floorplan-bubble-diagram},
  license={MIT}
}
```
