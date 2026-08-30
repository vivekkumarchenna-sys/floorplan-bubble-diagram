import sys; sys.path.insert(0, "src")
import cv2, numpy as np, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import truegraph_builder as tg
from build_graph import CLASS_NAMES

stem = sys.argv[2] if len(sys.argv) > 2 else "16649"
mask = cv2.imread(f"data/resplan_masks/{stem}_mask.png", 0).astype(np.int64)
raster = cv2.cvtColor(cv2.imread(f"data/resplan_raster/{stem}.png"), cv2.COLOR_BGR2RGB)
R, edges = tg.build_true_graph(mask)
nm = tg.name_map(R, mask)
scale = json.load(open("pixel_scale.json")).get(str(stem))

def disp(name):  # "Bedroom1" -> "Bedroom 1"
    import re
    return re.sub(r"([A-Za-z])(\d)$", r"\1 \2", name)

# raster color per room (its actual fill colour in the plan) -> same colour for the bubble
def room_color(cm):
    px = raster[cm]
    vals, cnts = np.unique(px.reshape(-1, 3), axis=0, return_counts=True)
    r, g, b = vals[cnts.argmax()]
    return (r/255, g/255, b/255)
rcol = {i: room_color(cm) for i, (c, cm) in R.items()}

vis = raster.copy()
vis[mask == 11] = [255, 0, 255]   # door magenta
vis[mask == 12] = [255, 165, 0]   # window orange

fig, (axL, axR) = plt.subplots(1, 2, figsize=(20, 8), gridspec_kw={"width_ratios": [1, 1.2]})
axL.imshow(vis)
for i, (c, cm) in R.items():
    ys, xs = np.where(cm)
    axL.text(xs.mean(), ys.mean(), disp(nm[i]), ha="center", va="center", fontsize=10, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
axL.set_title("Rasterised floor plan  (magenta = door, orange = window)", fontsize=14, fontweight="bold")
axL.axis("off")

pos = {}; area = {}
for i, (c, cm) in R.items():
    ys, xs = np.where(cm); pos[i] = (xs.mean(), -ys.mean()); area[i] = cm.sum() * scale if scale else cm.sum()
amax = max(area.values())
for e, t in edges.items():
    a, b = tuple(e); x = [pos[a][0], pos[b][0]]; y = [pos[a][1], pos[b][1]]
    if t == "door":
        axR.plot(x, y, color="#111111", lw=2.8, solid_capstyle="round", zorder=1)
    elif t == "open":
        axR.plot(x, y, color="#8E24AA", lw=5.2, solid_capstyle="round", zorder=1)  # thick purple = open passage
    else:
        axR.plot(x, y, color="#9E9E9E", lw=2.4, ls=(0, (1, 3)), dash_capstyle="round", zorder=1)
cx0 = np.mean([p[0] for p in pos.values()]); cy0 = np.mean([p[1] for p in pos.values()])
for i in R:
    size_i = 380 + 2600 * (area[i] / amax)
    axR.scatter([pos[i][0]], [pos[i][1]], s=size_i, c=[rcol[i]], edgecolors="#333", linewidths=1.8, zorder=3)
    lbl = f"{disp(nm[i])}\n{area[i]:.1f} m²" if scale else disp(nm[i])
    dx = pos[i][0] - cx0; dy = pos[i][1] - cy0; n = (dx * dx + dy * dy) ** 0.5
    if n < 1e-6:            # node at the graph centre (the hub): drop its label below
        ux, uy = 0.0, -1.0
    else:
        ux, uy = dx / n, dy / n
    off = (size_i / 3.14159) ** 0.5 + 9      # just clear the bubble edge (small gap)
    ha = "left" if ux > 0.35 else ("right" if ux < -0.35 else "center")
    va = "bottom" if uy > 0.35 else ("top" if uy < -0.35 else "center")
    axR.annotate(lbl, pos[i], xytext=(off * ux, off * uy), textcoords="offset points",
                 ha=ha, va=va, fontsize=10, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#bbb", lw=0.6, alpha=0.92), zorder=5)
axR.set_title("Typed bubble diagram  (door = black,  open passage = purple,  shared-wall = dotted)", fontsize=14, fontweight="bold")
axR.axis("off"); axR.margins(0.22)
h = [mlines.Line2D([], [], color="#111", lw=2.8, label="door"),
     mlines.Line2D([], [], color="#8E24AA", lw=5.0, label="open passage"),
     mlines.Line2D([], [], color="#9E9E9E", lw=2.4, ls=(0, (1, 3)), label="shared-wall")]
axR.legend(handles=h, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.05), frameon=True, fontsize=11)
fig.suptitle(f"Plan {stem}: floor plan  →  typed bubble diagram (bubble colours = raster colours)", fontsize=15, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(sys.argv[1], dpi=140, bbox_inches="tight"); print("saved", sys.argv[1])
