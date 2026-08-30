"""
generate_bubble.py - Image → SegFormer → Bubble Diagram
========================================================
Standalone script: give it a floor-plan image (or folder of images),
it runs the trained model and outputs bubble diagram(s).

Colab usage:
    # single image
    !python generate_bubble.py --image data/resplan_raster/42.png

    # folder of images
    !python generate_bubble.py --image data/resplan_raster/ --limit 20

    # custom checkpoint + output dir
    !python generate_bubble.py --image img.png --ckpt checkpoints/segformer/best_model.pth --out output/

    # show inline in Colab notebook (instead of saving)
    from generate_bubble import BubbleGenerator
    gen = BubbleGenerator("checkpoints/segformer/best_model.pth")
    gen.show("data/resplan_raster/42.png")
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    _SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    _SCRIPT_DIR = Path("/content/drive/MyDrive/bubble_diagram_project")

sys.path.insert(0, str(_SCRIPT_DIR))

import json

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import SegformerForSemanticSegmentation

from build_graph import (
    CLASS_NAMES, build_graph_from_segmentation, graph_summary,
)
from proximity import compute_proximity_matrix, print_proximity_matrix
from visualize import draw_bubble_diagram
import render_bubble as rb

# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

PRETRAINED    = "nvidia/segformer-b3-finetuned-ade-512-512"
NUM_CLASSES   = 16
IMG_SIZE      = 512
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

# Derived from inference.py's canonical palette rather than restated here.
# Every past copy of this table drifted from it independently (a BGR/RGB
# swap, then a four-class cyclic shift, then three near-identical greens);
# deriving it removes the possibility by construction.
from inference import _PALETTE as _BGR_PALETTE
_PALETTE = _BGR_PALETTE[:, ::-1].copy()   # canonical palette is BGR, figures draw RGB


# ══════════════════════════════════════════════════════════════════════════════
# Generator class (also usable as a library in notebooks)
# ══════════════════════════════════════════════════════════════════════════════

class BubbleGenerator:
    """Load model once, generate bubble diagrams for any number of images."""

    def __init__(self, ckpt_path: str | Path, device: str | None = None,
                 pixel_scale_path: str | Path | None = None,
                 legacy: bool = False):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = self._load_model(ckpt_path)
        self.pixel_scale_map = self._load_pixel_scale(pixel_scale_path)
        # legacy=True selects the pre-rework dilation-based M2, the weighted
        # proximity matrix and the force-directed rendering. The default is the
        # geometry-based pipeline the paper reports (Sections 5.2-5.4).
        self.legacy = legacy

    def _load_model(self, ckpt_path: str | Path) -> SegformerForSemanticSegmentation:
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        id2label = {i: str(i) for i in range(NUM_CLASSES)}
        label2id = {v: k for k, v in id2label.items()}

        model = SegformerForSemanticSegmentation.from_pretrained(
            PRETRAINED,
            num_labels=NUM_CLASSES,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        model.to(self.device).eval()

        epoch = ckpt.get("epoch", "?")
        miou  = ckpt.get("val_mIoU", "?")
        print(f"[model] Loaded: epoch={epoch}  val_mIoU={miou}  device={self.device}")
        return model

    @staticmethod
    def _load_pixel_scale(path: str | Path | None) -> dict[str, float] | None:
        if path is None:
            # try default location
            default = _SCRIPT_DIR / "pixel_scale.json"
            if default.exists():
                path = default
            else:
                return None
        path = Path(path)
        if not path.exists():
            print(f"[warn] pixel_scale.json not found at {path}, areas will be in pixels")
            return None
        with open(path) as f:
            data = json.load(f)
        print(f"[scale] Loaded pixel_scale.json ({len(data)} entries)")
        return data

    @torch.no_grad()
    def segment(self, image_bgr: np.ndarray) -> np.ndarray:
        """Run segmentation, return (H, W) int mask at original resolution."""
        h, w = image_bgr.shape[:2]
        img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(self.device)

        logits = self.model(pixel_values=tensor).logits
        logits = F.interpolate(logits, size=(IMG_SIZE, IMG_SIZE),
                               mode="bilinear", align_corners=False)
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()

        if (h, w) != (IMG_SIZE, IMG_SIZE):
            pred = cv2.resize(pred.astype(np.uint8), (w, h),
                              interpolation=cv2.INTER_NEAREST)
        return pred.astype(np.int64)

    def generate(self, image_path: str | Path, save_path: str | Path | None = None):
        """
        Full pipeline: image → segmentation → graph → bubble diagram.

        Returns (fig, G, matrix, labels).
        If save_path is given, saves the figure and closes it.
        """
        image_path = Path(image_path)
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise IOError(f"Could not read: {image_path}")

        # segment
        mask = self.segment(image_bgr)

        # build graph (with area in sq meters if scale is available)
        stem = image_path.stem
        scale = None
        if self.pixel_scale_map is not None:
            scale = self.pixel_scale_map.get(stem)
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        if self.legacy:
            G = build_graph_from_segmentation(mask, pixel_scale=scale)
            if G.number_of_nodes() > 0:
                matrix, labels = compute_proximity_matrix(G)
            else:
                matrix, labels = np.zeros((0, 0)), []
        else:
            # M2 (Section 5.2): deterministic geometric typed graph.
            # M3 (Section 5.3): the categorical typed adjacency matrix.
            G = rb.build_typed_graph(mask, raster_rgb=img_rgb, pixel_scale=scale)
            matrix, labels = rb.typed_adjacency(G)

        # draw
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))

        # panel 1: original image
        axes[0].imshow(img_rgb)
        axes[0].set_title("Input Floor Plan", fontsize=13)
        axes[0].axis("off")

        # panel 2: segmentation overlay
        overlay_rgb = self._make_overlay(img_rgb, mask)
        axes[1].imshow(overlay_rgb)
        axes[1].set_title("Segmentation", fontsize=13)
        axes[1].axis("off")

        # panel 3: typed bubble diagram (M4, Section 5.4)
        if not self.legacy:
            rb.draw_typed_bubble(G, axes[2], title="Typed bubble diagram")
        elif G.number_of_nodes() > 0:
            draw_bubble_diagram(G, ax=axes[2], title="Bubble Diagram (legacy)")
        else:
            axes[2].text(0.5, 0.5, "No rooms detected", ha="center", va="center",
                         transform=axes[2].transAxes, fontsize=14)
            axes[2].set_title("Bubble Diagram", fontsize=13)
            axes[2].axis("off")

        fig.suptitle(image_path.stem, fontsize=15, fontweight="bold")
        fig.tight_layout()

        if save_path is not None:
            fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
            plt.close(fig)
            fig = None

        return fig, G, matrix, labels

    def show(self, image_path: str | Path):
        """Generate and display inline (for Colab/Jupyter)."""
        matplotlib.use("module://matplotlib_inline.backend_inline")
        fig, G, matrix, labels = self.generate(image_path)
        if not self.legacy:
            print(f"{G.number_of_nodes()} rooms, {G.number_of_edges()} typed edges")
            print("")
            print("Typed adjacency matrix (D door, OP open passage, SW shared wall):")
            print(rb.format_adjacency(matrix, labels))
            plt.show()
            return
        print(graph_summary(G))
        if labels:
            print("\nProximity Matrix:")
            print(print_proximity_matrix(matrix, labels))
        plt.show()

    @staticmethod
    def _make_overlay(img_rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        colour = _PALETTE[mask.clip(0, NUM_CLASSES - 1)]
        overlay = (img_rgb * (1 - alpha) + colour * alpha).astype(np.uint8)
        overlay[mask == 0] = img_rgb[mask == 0]
        return overlay


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Generate bubble diagrams from floor plan images")
    parser.add_argument(
        "--image", type=str, required=True,
        help="Path to a single image or a folder of images",
    )
    parser.add_argument(
        "--ckpt", type=str,
        default=str(_SCRIPT_DIR / "checkpoints" / "segformer" / "best_model.pth"),
        help="Path to .pth checkpoint",
    )
    parser.add_argument(
        "--out", type=str,
        default=str(_SCRIPT_DIR / "output" / "bubbles"),
        help="Output directory for saved figures",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Max images to process from a folder (0 = all)",
    )
    parser.add_argument(
        "--pixel-scale", type=str, default=None,
        help="Path to pixel_scale.json for area in sq meters (auto-detected if in project root)",
    )
    parser.add_argument(
        "--legacy", action="store_true",
        help="Use the legacy dilation-based M2, the weighted proximity matrix and the "
             "force-directed rendering instead of the geometry-based pipeline of the paper",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    gen = BubbleGenerator(args.ckpt, pixel_scale_path=args.pixel_scale, legacy=args.legacy)

    # single image or folder
    image_path = Path(args.image)
    if image_path.is_file():
        paths = [image_path]
    elif image_path.is_dir():
        paths = sorted([p for p in image_path.iterdir() if p.suffix.lower() in _IMG_EXTS])
        if args.limit > 0:
            paths = paths[:args.limit]
    else:
        raise FileNotFoundError(f"Not found: {image_path}")

    print(f"[run] Processing {len(paths)} image(s) -> {out_dir}\n")

    for i, p in enumerate(paths):
        save_to = out_dir / f"{p.stem}_bubble.png"
        _, G, matrix, labels = gen.generate(p, save_path=save_to)

        n = G.number_of_nodes()
        e = G.number_of_edges()
        print(f"  [{i+1:>4}/{len(paths)}]  {p.stem:>8s}  ->  {n} rooms, {e} edges  ->  {save_to.name}")

    print(f"\nDone. Output: {out_dir}")


if __name__ == "__main__":
    main()
