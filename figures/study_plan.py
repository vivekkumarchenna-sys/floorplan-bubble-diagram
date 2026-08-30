# -*- coding: utf-8 -*-
"""Labelled floor plan for the user study: the rasterised plan with each room's
name printed at its centroid, in the same colours the bubble diagram uses.
Usage (from repo root):  python study_plan.py <out.png> <stem>
"""
import sys
sys.path.insert(0, "src")
import cv2, numpy as np, re
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import truegraph_builder as tg

stem = sys.argv[2] if len(sys.argv) > 2 else "16649"
out = sys.argv[1]
mask = cv2.imread(f"data/resplan_masks/{stem}_mask.png", 0).astype(np.int64)
raster = cv2.cvtColor(cv2.imread(f"data/resplan_raster/{stem}.png"), cv2.COLOR_BGR2RGB)
R = tg.rooms_of(mask)
nm = tg.name_map(R, mask)
def disp(s): return re.sub(r"([A-Za-z])(\d)$", r"\1 \2", s)

fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(raster)
for i, (c, cm) in R.items():
    ys, xs = np.where(cm)
    ax.text(xs.mean(), ys.mean(), disp(nm[i]), ha="center", va="center",
            fontsize=10, fontweight="bold", color="#111",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#666", lw=0.7, alpha=0.9))
ax.set_title(f"Floor plan - plan {stem}", fontsize=14, fontweight="bold")
ax.axis("off")
fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); print("saved", out)
