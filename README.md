# Floor Plan Bubble Diagram Generator

Automatically convert architectural floor plan images into **bubble diagrams** — abstract graph representations showing rooms, their connections, and spatial relationships.

```
Floor Plan Image  →  Semantic Segmentation  →  Room Graph  →  Bubble Diagram
```

## Pipeline

| Step | Script | Description |
|------|--------|-------------|
| 1 | `dataset.py` | PyTorch dataset with albumentations augmentation |
| 2 | `train.py` | SegFormer-B3 training (16-class semantic segmentation) |
| 3 | `train_deeplab.py` | DeepLabV3+ alternative training script |
| 4 | `build_graph.py` | Extract room-adjacency graph from segmentation mask |
| 5 | `build_gt_graph.py` | Ground-truth graph from polygon data (Shapely) |
| 6 | `proximity.py` | Weighted adjacency matrix from room graph |
| 7 | `visualize.py` | Bubble diagram visualisation (Fruchterman-Reingold layout) |
| 8 | `evaluate.py` | Evaluation metrics (mIoU, edge F1, GED, Frobenius) |
| 9 | `inference.py` | Batch evaluation on test set with GT comparison |
| 10 | `generate_bubble.py` | End-to-end: image → bubble diagram |
| 11 | `run_ablation.py` | Automated 6-variant ablation study |
| 12 | `fig_ablation.py` | Publication-quality ablation bar chart |

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

This project uses the [ResPlan dataset](https://github.com/ResPlanProject) — rasterised residential floor plans with per-pixel semantic labels.

Expected directory structure:

```
data/
  resplan_raster/     # RGB images: {id}.png
  resplan_masks/      # Grayscale masks: {id}_mask.png
  splits/
    train.txt         # one image stem per line
    val.txt
    test.txt
```

> **Note:** Data is not included in this repository due to size. Place the dataset in `data/` or update paths in the config.

### Model Checkpoint

Download the trained checkpoint and place it at:

```
checkpoints/segformer/best_model.pth
```

## Usage

### Generate Bubble Diagrams

```bash
# single image
python generate_bubble.py --image path/to/floorplan.png

# folder of images
python generate_bubble.py --image path/to/images/ --limit 20

# custom checkpoint and output directory
python generate_bubble.py --image img.png --ckpt path/to/model.pth --out output/
```

### Use in Python / Jupyter

```python
from generate_bubble import BubbleGenerator

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

gen = BubbleGenerator("/content/drive/MyDrive/bubble_diagram_project/checkpoints/segformer/best_model.pth")
gen.show("/content/drive/MyDrive/bubble_diagram_project/data/resplan_raster/42.png")
```

### Train a Model

```bash
# SegFormer-B3
python train.py

# DeepLabV3+ (alternative)
python train_deeplab.py
```

Training saves checkpoints to `checkpoints/segformer/best_model.pth` and logs to `results/history_segformer.json`.

### Evaluate on Test Set

```bash
# full evaluation (pixel metrics + graph metrics)
python inference.py

# with visual samples
python inference.py --save-vis 20

# faster (reduce GED timeout)
python inference.py --ged-timeout 5
```

Inference supports batch checkpointing (saves every 300 images) and resume on crash. Outputs per-image CSV, summary stats, and per-class IoU to `results/eval_<timestamp>/`.

### Ablation Study

```bash
# run all 6 variants (default: 300 images per variant)
python run_ablation.py

# fewer images for speed
python run_ablation.py --limit 100 --ged-timeout 2

# all test images
python run_ablation.py --limit 0
```

Outputs `results/ablation/ablation_results.csv` and `fig4_ablation.pdf`.

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

Trained SegFormer-B3 on ResPlan dataset (17,107 floor plans):

| Metric | Value |
|--------|-------|
| val mIoU | 0.965 |
| test mIoU | 0.998 |
| Edge F1 | 0.696 |
| Converged at | Epoch 23 |

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

![Ablation Bar Chart](fig4_ablation.pdf)

## Project Structure

```
floorplan-bubble-diagram/
  dataset.py            # PyTorch dataset + augmentation
  train.py              # SegFormer-B3 training
  train_deeplab.py      # DeepLabV3+ training
  build_graph.py        # Segmentation mask → room graph
  build_gt_graph.py     # GT polygons → room graph
  proximity.py          # Room graph → adjacency matrix
  visualize.py          # Bubble diagram drawing
  evaluate.py           # Evaluation metrics
  inference.py          # Batch test-set evaluation
  generate_bubble.py    # End-to-end bubble diagram generator
  run_ablation.py       # Automated ablation study (6 variants)
  fig_ablation.py       # Ablation bar chart generator
  fig4_ablation.pdf     # Ablation results chart
  requirements.txt
  LICENSE
  README.md
```

## License

MIT License. See [LICENSE](LICENSE).
