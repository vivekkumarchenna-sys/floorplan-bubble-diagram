# -*- coding: utf-8 -*-
"""Typed (categorical) proximity chart: rooms x rooms grid, each cell coloured by
connection type (door / open passage / shared wall / none). Categorical types only."""
import sys; sys.path.insert(0, "src")
import cv2, numpy as np, re
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
# Embed TrueType (not Type 3) fonts so the released PDFs match the submitted artwork.
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
from matplotlib.patches import Rectangle, Patch
import truegraph_builder as tg

stem = sys.argv[2] if len(sys.argv) > 2 else "16649"
mask = cv2.imread(f"data/resplan_masks/{stem}_mask.png", 0).astype(np.int64)
R, edges = tg.build_true_graph(mask); nm = tg.name_map(R, mask)
def disp(s): return re.sub(r"([A-Za-z])(\d)$", r"\1 \2", s)
order = ["Living", "Kitchen", "Balcony", "Bathroom", "Bedroom"]
ids = sorted(R, key=lambda i: (next((k for k, o in enumerate(order) if nm[i].startswith(o)), 9), nm[i]))
labels = [disp(nm[i]) for i in ids]
n = len(ids)
et = {frozenset({a, b}): t for (a, b), t in edges.items()}
COL = {"door": "#2E7D32", "open": "#8E24AA", "shared-wall": "#F9A825", None: "#ECECEC"}
SYM = {"door": "D", "open": "OP", "shared-wall": "SW", None: ""}

fig, ax = plt.subplots(figsize=(8.6, 7.4))
for r in range(n):
    for c in range(n):
        if r == c:
            ax.add_patch(Rectangle((c, n-1-r), 1, 1, facecolor="#455A64", edgecolor="white", lw=1.5))
            continue
        t = et.get(frozenset({ids[r], ids[c]}))
        ax.add_patch(Rectangle((c, n-1-r), 1, 1, facecolor=COL.get(t, COL[None]), edgecolor="white", lw=1.5))
        if t:
            ax.text(c+0.5, n-1-r+0.5, SYM[t], ha="center", va="center",
                    fontsize=13, fontweight="bold", color="white")
ax.set_xlim(0, n); ax.set_ylim(0, n); ax.set_aspect("equal")
ax.set_xticks([i+0.5 for i in range(n)]); ax.set_xticklabels(labels, rotation=45, ha="left", fontsize=11)
ax.xaxis.tick_top()
ax.set_yticks([i+0.5 for i in range(n)]); ax.set_yticklabels(labels[::-1], fontsize=11)
ax.tick_params(length=0)
for sp in ax.spines.values(): sp.set_visible(False)
leg = [Patch(fc=COL["door"], label="door (D)"), Patch(fc=COL["open"], label="open passage (OP)"),
       Patch(fc=COL["shared-wall"], label="shared wall (SW)"), Patch(fc=COL[None], label="not connected")]
ax.legend(handles=leg, loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=4, fontsize=10, frameon=False)
ax.set_title(f"Typed adjacency matrix - plan {stem}", fontsize=13, fontweight="bold", pad=14)
fig.tight_layout()
fig.savefig(sys.argv[1], dpi=160, bbox_inches="tight"); print("saved", sys.argv[1])
