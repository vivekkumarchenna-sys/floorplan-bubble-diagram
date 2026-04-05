# Data Setup

This project uses the **ResPlan** dataset (17,107 residential floor plans).

## Directory structure

```
data/
  resplan_raster/     # RGB floor plan images: {id}.png
  resplan_masks/      # Grayscale segmentation masks: {id}_mask.png (values 0-15)
  resplan_raw/        # ResPlan.pkl (original Shapely polygon annotations)
  splits/
    train.txt         # 11,973 image stems
    val.txt           # 2,565 image stems
    test.txt          # 2,567 image stems
```

## 16 Segmentation Classes

| ID | Class      | ID | Class     |
|----|------------|----|-----------|
| 0  | Background | 8  | Parking   |
| 1  | Bedroom    | 9  | Pool      |
| 2  | Bathroom   | 10 | Wall      |
| 3  | Kitchen    | 11 | Door      |
| 4  | Living     | 12 | Window    |
| 5  | Balcony    | 13 | FrontDoor |
| 6  | Storage    | 14 | Column    |
| 7  | Stair      | 15 | Other     |

## Obtaining the dataset

**Download:** [Google Drive](https://drive.google.com/drive/folders/1YJJPfIVkQEnt1sYkobAKnT9wNWpieGFx?usp=sharing)

Download and extract the contents into this `data/` directory. The split files (`train.txt`, `val.txt`, `test.txt`) contain one image stem per line (e.g., `0`, `1`, `42`).
