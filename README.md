# Floor Plan Bubble Diagram Generator

Automatically convert architectural floor plan images into **bubble diagrams** — abstract graph representations showing rooms, their connections, and spatial relationships.

```
Floor Plan Image  →  Semantic Segmentation  →  Room Graph  →  Bubble Diagram
```

## Pipeline

| Step | Script | Description |
|------|--------|-------------|
| 1 | `src/dataset.py` | PyTorch dataset with albumentations augmentation |
| 2 | `src/train.py` | SegFormer-B3 training (16-class semantic segmentation) |
| 3 | `src/train_deeplab.py` | DeepLabV3+ alternative training script |
| 4 | `src/build_graph.py` | Extract room-adjacency graph from segmentation mask |
| 5 | `src/build_gt_graph.py` | Ground-truth graph from polygon data (Shapely) |
| 6 | `src/proximity.py` | Weighted adjacency matrix from room graph |
| 7 | `src/visualize.py` | Bubble diagram visualisation (Fruchterman-Reingold layout) |
| 8 | `src/evaluate.py` | Evaluation metrics (mIoU, edge F1, GED, Frobenius) |
| 9 | `src/inference.py` | Batch evaluation on test set with GT comparison |
| 10 | `src/generate_bubble.py` | End-to-end: image → bubble diagram |
| 11 | `src/run_ablation.py` | Automated 6-variant ablation study |
| 12 | `src/generate_survey.py` | Generate paired stimuli for user survey |
| 13 | `src/generate_survey_docx.py` | Create survey as Word document |
| 14 | `src/compute_split_stats.py` | Dataset partition statistics |

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

### Data

This project uses the [ResPlan dataset](https://github.com/ResPlanProject) — rasterised residential floor plans with per-pixel semantic labels. See [`data/README.md`](data/README.md) for setup instructions.

### Model Checkpoint

Download the trained checkpoints and place them in `checkpoints/`. See [`checkpoints/README.md`](checkpoints/README.md) for download links.

## Usage

### Generate Bubble Diagrams

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
python src/inference.py

# faster (reduce GED timeout)
python src/inference.py --ged-timeout 5
```

Inference supports batch checkpointing (saves every 300 images) and resume on crash.

### Ablation Study

```bash
# run all 6 variants (default: 300 images per variant)
python src/run_ablation.py

# fewer images for speed
python src/run_ablation.py --limit 100 --ged-timeout 2
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
| **Type Accuracy** | Correct edge classification (door / arch / shared-wall) |
| **GED** | Graph edit distance (structural similarity) |
| **Frobenius Norm** | Distance between proximity matrices |

## Graph Edge Types

| Type | Weight | Meaning |
|------|--------|---------|
| `door` | 1.0 | Direct passage through a door |
| `arch` | 0.8 | Open passage (no physical door) |
| `shared-wall` | 0.3 | Adjacent rooms with no passage |

## Results

### Backbone Comparison

| Method | Params (M) | mIoU | GPU-hours |
|--------|-----------|------|-----------|
| SegFormer-B3 | 47.3 | 0.9974 | ~3 (A100) |
| DeepLabV3+ (R101) | 59.3 | 0.9684 | ~5.4 (A100) |

### Ablation Results

| Variant | mIoU | Edge F1 | GED | Frobenius |
|---------|------|---------|-----|-----------|
| Full pipeline | 0.9978 | 0.6956 | 3.54 | 0.4415 |
| w/o door edges | 0.9978 | 0.6956 | 12.72 | 0.4957 |
| w/o room types | 0.9978 | 0.0000 | 11.94 | 0.4415 |
| w/o corridor adj. | 0.9978 | 0.6888 | 3.66 | 0.4443 |
| w/o post-processing | 0.9978 | 0.0000 | 12.46 | 0.6112 |
| w/o edge typing | 0.9978 | 0.6956 | 12.74 | 0.6120 |

Key findings: room type classification and dilation-based post-processing are critical components — removing either drops Edge F1 to zero.

### Paper Figures

All figures are generated via Python scripts and saved as 300 DPI PDFs:

| Figure | Script | Description |
|--------|--------|-------------|
| Fig 1 | `figures/fig_pipeline.py` | Pipeline overview diagram |
| Fig 2 | `figures/fig_training.py` | Training vs validation loss curve |
| Fig 3 | `figures/fig_training.py` | Training vs validation mIoU curve |
| Fig 4 | `figures/fig_ablation.py` | Ablation bar chart (Edge F1 + mIoU) |
| Fig 5 | `figures/fig_heatmap.py` | Proximity matrix heatmaps (GT vs Pred) |
| Fig 6 | `figures/fig_qualitative.py` | Qualitative pipeline panel (3 rows x 7 cols) |
| Fig 7 | `figures/fig_failures.py` | Failure cases with error annotation |

```bash
# generate figures (Fig 1-3 need no GPU)
python figures/fig_pipeline.py
python figures/fig_training.py
python figures/fig_heatmap.py
python figures/fig_qualitative.py
python figures/fig_failures.py
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
│   ├── build_graph.py
│   ├── build_gt_graph.py
│   ├── proximity.py
│   ├── visualize.py
│   ├── evaluate.py
│   ├── inference.py
│   ├── generate_bubble.py
│   ├── run_ablation.py
│   ├── compute_split_stats.py
│   ├── generate_survey.py
│   ├── generate_survey_docx.py
│   └── generate_survey_pdf.py
│
├── figures/                    # figure generation scripts + PDFs
│   ├── fig_pipeline.py
│   ├── fig_training.py
│   ├── fig_heatmap.py
│   ├── fig_qualitative.py
│   ├── fig_failures.py
│   ├── fig_ablation.py
│   └── fig_tables.py
│
├── checkpoints/                # model weights (not in repo)
│   └── README.md
│
└── data/                       # dataset (not in repo)
    └── README.md
```

## License

MIT License. See [LICENSE](LICENSE).

## Citation

If you use this code, please cite:

```bibtex
@software{floorplan_bubble_diagram,
  title={Automated Bubble Diagram Extraction from Architectural Floor Plans},
  author={Chenna, Vivek Kumar and Dev, Adhithya},
  url={https://github.com/vivekkumarchenna-sys/floorplan-bubble-diagram},
  license={MIT}
}
```
