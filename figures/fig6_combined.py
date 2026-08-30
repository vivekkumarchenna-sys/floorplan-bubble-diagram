# -*- coding: utf-8 -*-
"""Fig 6: three example plans, each raster (left) beside typed bubble diagram (right)."""
import sys; sys.path.insert(0, "src")
import cv2, numpy as np, json, re
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
# Embed TrueType (not Type 3) fonts so the released PDFs match the submitted artwork.
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
import matplotlib.lines as mlines
import truegraph_builder as tg

STEMS = ["15389", "16649", "13388"]
scale_all = json.load(open("pixel_scale.json"))

def disp(s): return re.sub(r"([A-Za-z])(\d)$", r"\1 \2", s)

fig, axes = plt.subplots(len(STEMS), 2, figsize=(13.2, 4.3 * len(STEMS)),
                         gridspec_kw={"width_ratios": [1.2, 1.2]})
for row, stem in enumerate(STEMS):
    mask = cv2.imread(f"data/resplan_masks/{stem}_mask.png", 0).astype(np.int64)
    raster = cv2.cvtColor(cv2.imread(f"data/resplan_raster/{stem}.png"), cv2.COLOR_BGR2RGB)
    R, edges = tg.build_true_graph(mask); nm = tg.name_map(R, mask)
    scale = scale_all.get(str(stem))
    def rcol(cm):
        px = raster[cm]; v, c = np.unique(px.reshape(-1, 3), axis=0, return_counts=True)
        r, g, b = v[c.argmax()]; return (r/255, g/255, b/255)
    col = {i: rcol(cm) for i, (c, cm) in R.items()}
    axL, axR = axes[row]
    vis = raster.copy(); vis[mask == 11] = [255, 0, 255]; vis[mask == 12] = [255, 165, 0]
    axL.imshow(vis)
    for i, (c, cm) in R.items():
        ys, xs = np.where(cm)
        axL.text(xs.mean(), ys.mean(), disp(nm[i]), ha="center", va="center", fontsize=7, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.04", fc="white", ec="none", alpha=0.85))
    axL.set_title(f"Plan {stem} - rasterised floor plan", fontsize=12, fontweight="bold"); axL.axis("off")
    pos = {}; area = {}
    for i, (c, cm) in R.items():
        ys, xs = np.where(cm); pos[i] = (xs.mean(), -ys.mean()); area[i] = cm.sum()*scale if scale else cm.sum()
    amax = max(area.values())
    for e, t in edges.items():
        a, b = tuple(e); x = [pos[a][0], pos[b][0]]; y = [pos[a][1], pos[b][1]]
        if t == "door": axR.plot(x, y, color="#111", lw=2.6, zorder=1)
        elif t == "open": axR.plot(x, y, color="#8E24AA", lw=5.0, zorder=1)
        else: axR.plot(x, y, color="#9E9E9E", lw=2.2, ls=(0, (1, 3)), zorder=1)
    cx0 = np.mean([p[0] for p in pos.values()]); cy0 = np.mean([p[1] for p in pos.values()])
    for i in R:
        s = 300 + 2200*(area[i]/amax)
        axR.scatter([pos[i][0]], [pos[i][1]], s=s, c=[col[i]], edgecolors="#333", linewidths=1.5, zorder=3)
        lbl = f"{disp(nm[i])}\n{area[i]:.1f} m²" if scale else disp(nm[i])
        dx = pos[i][0]-cx0; dy = pos[i][1]-cy0; n = (dx*dx+dy*dy)**0.5 or 1
        ux, uy = dx/n, dy/n; off = (s/3.14159)**0.5 + 8
        ha = "left" if ux > 0.35 else ("right" if ux < -0.35 else "center")
        va = "bottom" if uy > 0.35 else ("top" if uy < -0.35 else "center")
        axR.annotate(lbl, pos[i], xytext=(off*ux, off*uy), textcoords="offset points", ha=ha, va=va,
                     fontsize=8, fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#bbb", lw=0.5, alpha=0.92), zorder=5)
    axR.set_title(f"Plan {stem} - typed bubble diagram", fontsize=12, fontweight="bold", pad=14)
    axR.axis("off"); axR.margins(0.30)
h = [mlines.Line2D([], [], color="#111", lw=2.6, label="door"),
     mlines.Line2D([], [], color="#8E24AA", lw=5.0, label="open passage"),
     mlines.Line2D([], [], color="#9E9E9E", lw=2.2, ls=(0, (1, 3)), label="shared wall")]
fig.legend(handles=h, loc="lower center", ncol=3, fontsize=12, frameon=True, bbox_to_anchor=(0.5, 0.005))
fig.tight_layout(rect=[0, 0.03, 1, 1])
fig.savefig(sys.argv[1], dpi=150, bbox_inches="tight"); print("saved", sys.argv[1])
