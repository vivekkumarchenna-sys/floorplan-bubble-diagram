# -*- coding: utf-8 -*-
"""Clean single-panel geographic typed bubble diagram (3 types, raster colours)."""
import sys; sys.path.insert(0, "src")
import cv2, numpy as np, json, re
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import truegraph_builder as tg

stem = sys.argv[2] if len(sys.argv) > 2 else "15389"
mask = cv2.imread(f"data/resplan_masks/{stem}_mask.png", 0).astype(np.int64)
raster = cv2.cvtColor(cv2.imread(f"data/resplan_raster/{stem}.png"), cv2.COLOR_BGR2RGB)
R, edges = tg.build_true_graph(mask); nm = tg.name_map(R, mask)
scale = json.load(open("pixel_scale.json")).get(str(stem))
def disp(s): return re.sub(r"([A-Za-z])(\d)$", r"\1 \2", s)
def rcol(cm):
    px = raster[cm]; v, c = np.unique(px.reshape(-1, 3), axis=0, return_counts=True)
    r, g, b = v[c.argmax()]; return (r/255, g/255, b/255)
col = {i: rcol(cm) for i, (c, cm) in R.items()}
pos = {}; area = {}
for i, (c, cm) in R.items():
    ys, xs = np.where(cm); pos[i] = (xs.mean(), -ys.mean()); area[i] = cm.sum()*scale if scale else cm.sum()
amax = max(area.values())
fig, ax = plt.subplots(figsize=(8.5, 7))
for e, t in edges.items():
    a, b = tuple(e); x = [pos[a][0], pos[b][0]]; y = [pos[a][1], pos[b][1]]
    if t == "door": ax.plot(x, y, color="#111", lw=2.8, zorder=1)
    elif t == "open": ax.plot(x, y, color="#8E24AA", lw=5.2, zorder=1)
    else: ax.plot(x, y, color="#9E9E9E", lw=2.4, ls=(0, (1, 3)), zorder=1)
cx0 = np.mean([p[0] for p in pos.values()]); cy0 = np.mean([p[1] for p in pos.values()])
for i in R:
    s = 360 + 2400*(area[i]/amax)
    ax.scatter([pos[i][0]], [pos[i][1]], s=s, c=[col[i]], edgecolors="#333", linewidths=1.7, zorder=3)
    lbl = f"{disp(nm[i])}\n{area[i]:.1f} m²" if scale else disp(nm[i])
    dx = pos[i][0]-cx0; dy = pos[i][1]-cy0; n = (dx*dx+dy*dy)**0.5 or 1
    ux, uy = dx/n, dy/n; off = (s/3.14159)**0.5 + 8
    ha = "left" if ux > 0.35 else ("right" if ux < -0.35 else "center")
    va = "bottom" if uy > 0.35 else ("top" if uy < -0.35 else "center")
    ax.annotate(lbl, pos[i], xytext=(off*ux, off*uy), textcoords="offset points", ha=ha, va=va,
                fontsize=9.5, fontweight="bold", bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#bbb", lw=0.6, alpha=0.92), zorder=5)
ax.set_title(f"Typed bubble diagram - plan {stem}", fontsize=14, fontweight="bold")
ax.axis("off"); ax.margins(0.2)
h = [mlines.Line2D([], [], color="#111", lw=2.8, label="door"),
     mlines.Line2D([], [], color="#8E24AA", lw=5.0, label="open passage"),
     mlines.Line2D([], [], color="#9E9E9E", lw=2.4, ls=(0, (1, 3)), label="shared wall")]
ax.legend(handles=h, loc="lower center", ncol=3, fontsize=10, frameon=True, bbox_to_anchor=(0.5, -0.06))
fig.tight_layout(); fig.savefig(sys.argv[1], dpi=155, bbox_inches="tight"); print("saved", sys.argv[1])
