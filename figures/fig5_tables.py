# -*- coding: utf-8 -*-
"""Fig 5: the two tabular views of the same typed graph, side by side.
(a) proximity chart - the diamond lattice used in architectural programming;
(b) typed adjacency matrix. Both are rendered from the graph that module M2
builds from the plan's segmentation mask, so they carry identical information.

usage:  python figures/fig5_tables.py out/Fig5.pdf [stem]
"""
import sys; sys.path.insert(0, "src")
import re
import cv2, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
# Embed TrueType (not Type 3) fonts so the released PDFs match the submitted artwork.
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
import matplotlib.lines as mlines
from matplotlib.patches import Rectangle, Patch
import truegraph_builder as tg

out  = sys.argv[1]
stem = sys.argv[2] if len(sys.argv) > 2 else "13388"

mask = cv2.imread(f"data/resplan_masks/{stem}_mask.png", 0).astype(np.int64)
R, edges = tg.build_true_graph(mask); nm = tg.name_map(R, mask)

def disp(s): return re.sub(r"([A-Za-z])(\d)$", r"\1 \2", s)
order = ["Living", "Kitchen", "Balcony", "Bathroom", "Bedroom"]
ids = sorted(R, key=lambda i: (next((k for k, o in enumerate(order) if nm[i].startswith(o)), 99), nm[i]))
labels = [disp(nm[i]) for i in ids]
n = len(ids)
et = {frozenset({a, b}): t for (a, b), t in edges.items()}

COL_D = {"door": "#2E7D32", "open": "#8E24AA", "shared-wall": "#F9A825", None: "#EDEDED"}
COL_M = {"door": "#2E7D32", "open": "#8E24AA", "shared-wall": "#F9A825", None: "#ECECEC"}
SYM   = {"door": "D", "open": "OP", "shared-wall": "SW", None: ""}

fig = plt.figure(figsize=(13.2, 6.9))
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.05], wspace=0.06)

# ---- (a) proximity chart: the diamond lattice -------------------------------
axa = fig.add_subplot(gs[0])
def center(i, j): return (j - i, -(i + j))
for i in range(n):
    for j in range(i + 1, n):
        cx, cy = center(i, j)
        dia = [(cx + 1, cy), (cx, cy + 1), (cx - 1, cy), (cx, cy - 1), (cx + 1, cy)]
        axa.plot([p[0] for p in dia], [p[1] for p in dia], color="#c9c9c9", lw=0.9, zorder=1)
        t = et.get(frozenset({ids[i], ids[j]}))
        axa.scatter([cx], [cy], s=430, c=COL_D.get(t, COL_D[None]),
                    edgecolors="#555", linewidths=1.1, zorder=3)
for i in range(n):
    axa.text(-1.6, -2 * i, labels[i], ha="right", va="center", fontsize=12, fontweight="bold")
    axa.plot([-1.4, 1], [-2 * i, -2 * i], color="#888", lw=1.0, zorder=2)
axa.set_aspect("equal"); axa.axis("off")
axa.set_xlim(-5.5, n + 0.5); axa.set_ylim(-2 * (n - 1) - 1.5, 2.2)
axa.set_title(f"Plan {stem} \u2013 typed proximity chart", fontsize=14, fontweight="bold", pad=6)
h = [mlines.Line2D([], [], marker="o", ls="", ms=13, mec="#555", mfc=COL_D["door"], label="door"),
     mlines.Line2D([], [], marker="o", ls="", ms=13, mec="#555", mfc=COL_D["open"], label="open passage"),
     mlines.Line2D([], [], marker="o", ls="", ms=13, mec="#555", mfc=COL_D["shared-wall"], label="shared-wall"),
     mlines.Line2D([], [], marker="o", ls="", ms=13, mec="#555", mfc=COL_D[None], label="not connected")]
axa.legend(handles=h, loc="upper right", fontsize=11, frameon=True,
           title="Adjacency", title_fontsize=12)

# ---- (b) typed adjacency matrix --------------------------------------------
axb = fig.add_subplot(gs[1])
for r in range(n):
    for c in range(n):
        if r == c:
            axb.add_patch(Rectangle((c, n - 1 - r), 1, 1, facecolor="#455A64",
                                    edgecolor="white", lw=1.5))
            continue
        t = et.get(frozenset({ids[r], ids[c]}))
        axb.add_patch(Rectangle((c, n - 1 - r), 1, 1, facecolor=COL_M.get(t, COL_M[None]),
                                edgecolor="white", lw=1.5))
        if t:
            axb.text(c + 0.5, n - 1 - r + 0.5, SYM[t], ha="center", va="center",
                     fontsize=13, fontweight="bold", color="white")
axb.set_xlim(0, n); axb.set_ylim(0, n); axb.set_aspect("equal")
axb.set_xticks([i + 0.5 for i in range(n)])
axb.set_xticklabels(labels, rotation=45, ha="left", fontsize=11)
axb.xaxis.tick_top()
axb.set_yticks([i + 0.5 for i in range(n)])
axb.set_yticklabels(labels[::-1], fontsize=11)
axb.tick_params(length=0)
for sp in axb.spines.values(): sp.set_visible(False)
leg = [Patch(fc=COL_M["door"], label="door (D)"),
       Patch(fc=COL_M["open"], label="open passage (OP)"),
       Patch(fc=COL_M["shared-wall"], label="shared wall (SW)"),
       Patch(fc=COL_M[None], label="not connected")]
axb.legend(handles=leg, loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=4,
           fontsize=10, frameon=False)
axb.set_title(f"Typed adjacency matrix \u2013 plan {stem}", fontsize=14, fontweight="bold", pad=14)

fig.text(0.26, 0.015, "(a) Proximity chart", ha="center", fontsize=15, fontweight="bold")
fig.text(0.755, 0.015, "(b) Adjacency matrix", ha="center", fontsize=15, fontweight="bold")
fig.savefig(out, dpi=170, bbox_inches="tight"); print("saved", out)
