# Floor Plan Bubble Diagram Project

## Project Overview
Automated pipeline that converts architectural floor plan images into bubble diagrams via semantic segmentation and graph extraction. Uses the ResPlan dataset (17,107 residential floor plans).

## Pipeline (in order)
1. `dataset.py` — PyTorch `ResPlanSegDataset`, albumentations augmentation, splits from txt files
2. `train.py` — SegFormer-B3 (nvidia/segformer-b3-finetuned-ade-512-512), 16-class, 512x512, AdamW lr=6e-5, cosine annealing, early stopping patience=15
3. `train_deeplab.py` — DeepLabV3+ ResNet-101 alternative (not yet trained)
4. `build_graph.py` — Segmentation mask → room-adjacency graph via connected components + dilation. Key params: `dilation_px=15, door_min=15, wall_min=20, arch_min=30`
5. `build_gt_graph.py` — GT graph from Shapely polygons (buffer-based adjacency)
6. `proximity.py` — Weighted adjacency matrix (door=1.0, arch=0.8, shared-wall=0.3)
7. `visualize.py` — Bubble diagram (Fruchterman-Reingold layout, room-colored nodes)
8. `evaluate.py` — 4 metrics: `compute_miou`, `edge_metrics`, `graph_edit_distance`, `frobenius_norm`
9. `inference.py` — Batch evaluation on test set. Saves every 300 images (--batch-save). Has --limit and --ged-timeout flags. Resume support via existing per_image.csv.
10. `generate_bubble.py` — End-to-end: image → segmentation → graph → bubble diagram. `BubbleGenerator` class for notebook use.
11. `run_ablation.py` — 6-variant ablation study (automated)
12. `fig_ablation.py` — Ablation bar chart (Fig 4)
13. `fig_training.py` — Loss + mIoU curves (Fig 2 & 3)
14. `fig_heatmap.py` — Proximity matrix heatmaps GT vs Pred (Fig 5)
15. `fig_qualitative.py` — 3-row × 7-col qualitative panel (Fig 6)
16. `fig_failures.py` — Worst edge_f1 failure cases (Fig 7)
17. `fig_pipeline.py` — Pipeline overview diagram (Fig 1)
18. `compute_split_stats.py` — Dataset partition statistics (room/door instance counts per split)

## 16 Segmentation Classes
0=Background, 1=Bedroom, 2=Bathroom, 3=Kitchen, 4=Living, 5=Balcony, 6=Storage, 7=Stair, 8=Parking, 9=Pool, 10=Wall, 11=Door, 12=Window, 13=FrontDoor, 14=Column, 15=Other

## Class mapping in build_graph.py
- `ROOM_CLASSES = {1,2,3,4,5,6,7,8,9}` — all room/space types
- `DOOR_CLASSES = {11, 13}` — Door, FrontDoor

## Data Layout (on Google Colab Drive)
```
/content/drive/MyDrive/bubble_diagram_project/
  data/
    resplan_raster/     # RGB images: {id}.png (17,107 images)
    resplan_masks/      # Grayscale masks: {id}_mask.png
    resplan_raw/        # ResPlan.pkl (original Shapely polygons)
    splits/
      train.txt         # 11,973 stems
      val.txt           # 2,565 stems
      test.txt          # 2,567 stems
  checkpoints/segformer/best_model.pth
  results/
    history_segformer.json
    eval_20260403_155913/   # inference results
      per_image.csv
      summary.csv
      class_iou.csv
    ablation/
      ablation_results.csv
      fig4_ablation.pdf
```

## Current Model Results
- **SegFormer-B3**: Converged at epoch 23, val_mIoU=0.965
- **Test set (2,100 images)**:
  - mIoU: 0.9974
  - Edge F1: 0.6589
  - Edge Type Acc: 0.7853
  - GED: 4.20 ± 5.44
  - Frobenius (norm): 0.4625

## Per-Class IoU (test)
- Excellent (>0.99): Background, Bedroom, Bathroom, Living, Wall, Door, Window, FrontDoor
- Good: Kitchen (0.974)
- Weak: Balcony (0.741)
- Failing: Storage (0.112), Stair (0.035), Parking (0.020)
- Dead (no samples): Pool, class_14, class_15

## Ablation Results (100 images per variant)
- Full pipeline: edge_f1=0.6956
- w/o door edges: edge_f1=0.6956 (no change)
- w/o room types: edge_f1=0.0000 (critical)
- w/o corridor adj: edge_f1=0.6888 (minor)
- w/o post-processing: edge_f1=0.0000 (critical)
- w/o edge typing: edge_f1=0.6956 (no change)

## Key Finding
Edge F1 bottleneck is graph extraction thresholds (dilation_px, door_min, arch_min) and alignment gap between build_graph.py (dilation-based) and build_gt_graph.py (Shapely buffer-based), NOT the segmentation model.

## GitHub
https://github.com/Adhithya0109DEV/floorplan-bubble-diagram

## Colab Paths
- Project root: `/content/drive/MyDrive/bubble_diagram_project`
- Checkpoint: `checkpoints/segformer/best_model.pth`
- History JSON: `results/history_segformer.json`
- Per-image CSV: `results/eval_20260403_155913/per_image.csv`

## Conventions
- All figure scripts output 300 DPI PDFs with Times New Roman font
- Graph nodes have: class_id, class_name, area_px, centroid
- Graph edges have: edge_type (door/arch/shared-wall), overlap_px, door_px, opening_width
- Colab compatibility: all scripts handle `__file__` not defined via try/except fallback
- Git: user=Adhithya0109DEV, email=adhithyadevt7@gmail.com

## In Progress
- Area conversion: adding sq meter labels to bubble diagrams (scale from ResPlan.pkl polygon areas)
- compute_split_stats.py: running on Colab to get room/door instance counts per split for Table 1
- DeepLabV3+ training: not started, optional for backbone comparison table
