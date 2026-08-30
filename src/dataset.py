"""
ResPlanSegDataset
-----------------
PyTorch Dataset for residential / architectural plan segmentation.

Directory layout:
    resplan_raster/   {id}.png          – flat folder of RGB images
    resplan_masks/    {id}_mask.png     – flat folder of grayscale masks
    splits/
        train.txt   – one image stem per line (e.g. "1042")
        val.txt
        test.txt

Usage:
    train_ds = ResPlanSegDataset(img_dir, mask_dir, splits_file="splits/train.txt", split="train")
    val_ds   = ResPlanSegDataset(img_dir, mask_dir, splits_file="splits/val.txt",   split="val")
    loader   = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=4)
"""

import os
from pathlib import Path

import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Augmentation pipelines
# ---------------------------------------------------------------------------

def _make_coarse_dropout(p: float = 0.3):
    """CoarseDropout occlusion, portable across albumentations versions.

    Newer albumentations (>= 1.4 / 2.x) renamed the CoarseDropout kwargs, so
    try the current API first and fall back to the legacy names, letting the
    same code run on either. ``fill`` (legacy ``fill_value``) sets what the
    *image* holes are filled with (0 here). The *mask* is deliberately left
    untouched: ``fill_mask`` (legacy ``mask_fill_value``) is None, i.e. not
    applied, so the segmentation labels under a hole are preserved.
    """
    try:
        return A.CoarseDropout(
            num_holes_range=(4, 12),
            hole_height_range=(16, 64),
            hole_width_range=(16, 64),
            fill=0,
            fill_mask=None,
            p=p,
        )
    except TypeError:
        # albumentations < 1.4 used the older kwarg names
        return A.CoarseDropout(
            max_holes=12,
            min_holes=4,
            max_height=64,
            min_height=16,
            max_width=64,
            min_width=16,
            fill_value=0,
            mask_fill_value=None,
            p=p,
        )


def _build_train_transforms(img_size: int = 512) -> A.Compose:
    """
    Training augmentation pipeline:
      - Geometric  : HorizontalFlip, VerticalFlip, RandomRotate90
      - Scale jitter: ±20 %  (RandomResizedCrop covers this cleanly)
      - Noise      : GaussNoise
      - Dropout    : CoarseDropout
      - Colour norm: Normalize + ToTensorV2
    """
    return A.Compose([
        # ----- geometric -----
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),

        # ----- scale jitter ±20 % -----
        # scale=(0.64, 1.44) → side lengths from 0.8× to 1.2× of img_size
        A.RandomResizedCrop(
            height=img_size,
            width=img_size,
            scale=(0.64, 1.44),   # area fraction → sqrt gives ±20 % on each side
            ratio=(0.75, 1.333),
            interpolation=cv2.INTER_LINEAR,
            p=1.0,
        ),

        # ----- noise -----
        A.GaussNoise(
            var_limit=(10.0, 50.0),   # std as fraction of [0,1] pixel range
            p=0.5,
        ),

        # ----- occlusion / regularisation -----
        _make_coarse_dropout(p=0.3),

        # ----- normalisation -----
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
        ToTensorV2(),
    ])


def _build_val_transforms(img_size: int = 512) -> A.Compose:
    """Validation pipeline: resize to a fixed square, then normalize."""
    return A.Compose([
        A.Resize(img_size, img_size, interpolation=cv2.INTER_LINEAR),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
        ToTensorV2(),
    ])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

_IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


class ResPlanSegDataset(Dataset):
    """
    Parameters
    ----------
    img_dir     : str | Path  – flat folder of RGB images ({stem}.png).
    mask_dir    : str | Path  – flat folder of masks ({stem}_mask.png).
    splits_file : str | Path  – txt file with one image stem per line.
    split       : "train" | "val"  – selects the augmentation pipeline.
    img_size    : int  – spatial size (H = W) fed to the model (default 512).
    transform   : A.Compose | None – overrides the built-in pipeline.
    """

    def __init__(
        self,
        img_dir: str | Path,
        mask_dir: str | Path,
        splits_file: str | Path,
        split: str = "train",
        img_size: int = 512,
        transform: A.Compose | None = None,
    ) -> None:
        super().__init__()
        self.img_dir  = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.split    = split.lower()

        if transform is not None:
            self.transform = transform
        elif self.split == "train":
            self.transform = _build_train_transforms(img_size)
        else:
            self.transform = _build_val_transforms(img_size)

        # read stems from the split txt file
        splits_file = Path(splits_file)
        if not splits_file.exists():
            raise FileNotFoundError(f"Splits file not found: {splits_file}")
        stems = [l.strip() for l in splits_file.read_text().splitlines() if l.strip()]
        if not stems:
            raise ValueError(f"Splits file is empty: {splits_file}")

        # build and validate paths
        self.img_paths: list[Path] = []
        missing_imgs, missing_masks = [], []
        for stem in stems:
            img_p  = self.img_dir  / f"{stem}.png"
            mask_p = self.mask_dir / f"{stem}_mask.png"
            if not img_p.exists():
                missing_imgs.append(stem)
            if not mask_p.exists():
                missing_masks.append(stem)
            self.img_paths.append(img_p)

        if missing_imgs:
            raise FileNotFoundError(
                f"{len(missing_imgs)} image(s) not found in {self.img_dir}. "
                f"First few: {missing_imgs[:5]}"
            )
        if missing_masks:
            raise FileNotFoundError(
                f"{len(missing_masks)} mask(s) not found in {self.mask_dir}. "
                f"First few: {missing_masks[:5]}"
            )

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.img_paths)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        img_path  = self.img_paths[idx]
        mask_path = self.mask_dir / f"{img_path.stem}_mask.png"

        # --- load image (BGR → RGB) ---
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            raise IOError(f"Could not read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # H×W×3, uint8

        # --- load mask (grayscale) ---
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)  # H×W, uint8
        if mask is None:
            raise IOError(f"Could not read mask: {mask_path}")

        # --- augment ---
        augmented = self.transform(image=image, mask=mask)
        image_t   = augmented["image"]          # float32 tensor C×H×W
        mask_t    = augmented["mask"].long()    # int64  tensor H×W

        return {"image": image_t, "mask": mask_t, "stem": img_path.stem}


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile, random
    from PIL import Image

    print("Smoke-testing ResPlanSegDataset …")

    # ---- create a tiny fake dataset in a temp directory ----
    with tempfile.TemporaryDirectory() as tmp:
        img_dir   = Path(tmp) / "images"
        mask_dir  = Path(tmp) / "masks"
        split_dir = Path(tmp) / "splits"
        img_dir.mkdir(); mask_dir.mkdir(); split_dir.mkdir()

        rng   = np.random.default_rng(0)
        stems = [f"sample_{i:03d}" for i in range(6)]
        for stem in stems:
            img_arr = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
            Image.fromarray(img_arr).save(img_dir / f"{stem}.png")
            msk_arr = rng.integers(0, 4, (480, 640), dtype=np.uint8)
            Image.fromarray(msk_arr).save(mask_dir / f"{stem}_mask.png")

        # write simple split files (4 train, 2 val)
        (split_dir / "train.txt").write_text("\n".join(stems[:4]))
        (split_dir / "val.txt").write_text("\n".join(stems[4:]))

        for split in ("train", "val"):
            ds = ResPlanSegDataset(
                img_dir, mask_dir,
                splits_file=split_dir / f"{split}.txt",
                split=split, img_size=256,
            )
            loader = DataLoader(ds, batch_size=2, shuffle=(split == "train"))
            batch = next(iter(loader))
            img, mask = batch["image"], batch["mask"]
            print(
                f"  [{split:5s}]  image {tuple(img.shape)}  "
                f"dtype={img.dtype}  "
                f"mask {tuple(mask.shape)}  dtype={mask.dtype}  "
                f"classes={mask.unique().tolist()}"
            )

    print("Done – all OK.")
