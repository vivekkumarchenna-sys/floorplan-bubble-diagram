import sys; sys.path.insert(0, "src")
import cv2, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
# Embed TrueType (not Type 3) fonts so the released PDFs match the submitted artwork.
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
import matplotlib.lines as mlines
import truegraph_builder as tg

stem = sys.argv[2] if len(sys.argv) > 2 else "16649"
mask = cv2.imread(f"data/resplan_masks/{stem}_mask.png", 0).astype(np.int64)
R, edges = tg.build_true_graph(mask)
nm = tg.name_map(R, mask)

def disp(s):
    import re; return re.sub(r"([A-Za-z])(\d)$", r"\1 \2", s)

order_pref = ["Living", "Kitchen", "Balcony", "Bathroom", "Bedroom"]
ids = sorted(R, key=lambda i: (next((k for k, p in enumerate(order_pref) if nm[i].startswith(p)), 99), nm[i]))
labels = [disp(nm[i]) for i in ids]
n = len(ids)
etype = {frozenset({a, b}): t for (a, b), t in edges.items()}

COL = {"door": "#2E7D32", "open": "#8E24AA", "shared-wall": "#F9A825", None: "#EDEDED"}

def center(i, j):            # cell for pair i<j, rotated-diamond coords
    return (j - i, -(i + j))

fig, ax = plt.subplots(figsize=(10, 9))
# diamond cells + coloured dots
for i in range(n):
    for j in range(i + 1, n):
        cx, cy = center(i, j)
        dia = [(cx + 1, cy), (cx, cy + 1), (cx - 1, cy), (cx, cy - 1), (cx + 1, cy)]
        ax.plot([p[0] for p in dia], [p[1] for p in dia], color="#c9c9c9", lw=0.9, zorder=1)
        t = etype.get(frozenset({ids[i], ids[j]}))
        ax.scatter([cx], [cy], s=430, c=COL.get(t, COL[None]), edgecolors="#555", linewidths=1.1, zorder=3)

# room labels down the left; the line runs right until it touches the first diamond vertex (1, -2i)
for i in range(n):
    ax.text(-1.6, -2 * i, labels[i], ha="right", va="center", fontsize=12, fontweight="bold")
    ax.plot([-1.4, 1], [-2 * i, -2 * i], color="#888", lw=1.0, zorder=2)

ax.set_aspect("equal"); ax.axis("off")
ax.set_xlim(-5.5, n + 0.5); ax.set_ylim(-2 * (n - 1) - 1.5, 2.2)
ax.set_title(f"Plan {stem} - typed proximity chart", fontsize=16, fontweight="bold", pad=6)
h = [mlines.Line2D([], [], marker="o", ls="", ms=13, mec="#555", mfc=COL["door"], label="door"),
     mlines.Line2D([], [], marker="o", ls="", ms=13, mec="#555", mfc=COL["open"], label="open passage"),
     mlines.Line2D([], [], marker="o", ls="", ms=13, mec="#555", mfc=COL["shared-wall"], label="shared-wall"),
     mlines.Line2D([], [], marker="o", ls="", ms=13, mec="#555", mfc=COL[None], label="not connected")]
ax.legend(handles=h, loc="upper right", fontsize=11, frameon=True, title="Adjacency", title_fontsize=12)
fig.tight_layout(); fig.savefig(sys.argv[1], dpi=150, bbox_inches="tight"); print("saved", sys.argv[1])
