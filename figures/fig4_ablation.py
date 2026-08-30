# -*- coding: utf-8 -*-
"""Fig 4: ablation of the M2 rules - edge-type accuracy and per-type recall per variant."""
import sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
# Embed TrueType (not Type 3) fonts so the released PDFs match the submitted artwork.
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
import numpy as np

variants = ["Full rules", "− door\nprecedence", "− open-passage\ndetection", "− multi-room\nhub rule"]
metrics = {
    "Edge-type acc.": [1.000, 0.607, 0.933, 0.993],
    "Door recall":    [1.000, 0.263, 1.000, 0.974],
    "Open recall":    [1.000, 1.000, 0.000, 1.000],
    "Shared-wall recall": [1.000, 1.000, 1.000, 1.000],
}
colors = ["#37474F", "#2E7D32", "#8E24AA", "#F9A825"]
x = np.arange(len(variants)); w = 0.2
fig, ax = plt.subplots(figsize=(10, 5.2))
for k, (name, vals) in enumerate(metrics.items()):
    ax.bar(x + (k - 1.5) * w, vals, w, label=name, color=colors[k], edgecolor="#222", linewidth=0.5)
ax.set_xticks(x); ax.set_xticklabels(variants, fontsize=10)
ax.set_ylim(0, 1.08); ax.set_ylabel("score", fontsize=11)
ax.set_title("Ablation of the M2 rules (800-plan sample, GT masks, vs geometry-based ground truth)",
             fontsize=12, fontweight="bold")
ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.20), fontsize=10, frameon=True)
ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)
for spine in ("top", "right"): ax.spines[spine].set_visible(False)
fig.tight_layout()
fig.savefig(sys.argv[1], dpi=150, bbox_inches="tight"); print("saved", sys.argv[1])
