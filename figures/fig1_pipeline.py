# -*- coding: utf-8 -*-
"""Fig 1: the four-module pipeline on plan 16649.
(a) raster  (b) M1 predicted label map  (c) M3 adjacency matrix  (d) M4 bubble diagram."""
import sys; sys.path.insert(0, "src")
import cv2, numpy as np, json, re, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
# Embed TrueType (not Type 3) fonts so the released PDFs match the submitted artwork.
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
import matplotlib.lines as mlines
from matplotlib.patches import Rectangle, Patch
import truegraph_builder as tg
from inference import load_model, predict_mask, _PALETTE

stem = "16649"
raster = cv2.imread(f"data/resplan_raster/{stem}.png")
raster_rgb = cv2.cvtColor(raster, cv2.COLOR_BGR2RGB)
gtmask = cv2.imread(f"data/resplan_masks/{stem}_mask.png", 0).astype(np.int64)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model("checkpoints/segformer/best_model.pth", DEV)
pred = predict_mask(model, raster, DEV)
# colour the predicted label map with each class's dominant RASTER colour so that
# panel (b) matches the raster (a) and the bubble diagram (d) exactly. The stored
# _PALETTE is a separate Material-Design palette that does NOT match the ResPlan
# raster hues (which is why the old BGR<->RGB flip mis-tinted Living/Balcony), so
# we sample the actual raster colour of each predicted class instead.
lab_rgb = np.full((*pred.shape, 3), 255, np.uint8)
for c in range(1, 16):
    m = (pred == c)
    if not m.any():
        continue
    px = raster_rgb[m]; v, cnt = np.unique(px.reshape(-1, 3), axis=0, return_counts=True)
    lab_rgb[m] = v[cnt.argmax()]
lab_rgb[pred == 0] = (255, 255, 255)

R, edges = tg.build_true_graph(gtmask); nm = tg.name_map(R, gtmask)
scale = json.load(open("pixel_scale.json")).get(str(stem))
def disp(s): return re.sub(r"([A-Za-z])(\d)$", r"\1 \2", s)
def rcol(cm):
    px = raster_rgb[cm]; v, cnt = np.unique(px.reshape(-1, 3), axis=0, return_counts=True)
    r, g, b = v[cnt.argmax()]; return (r/255, g/255, b/255)
col = {i: rcol(cm) for i, (c, cm) in R.items()}

fig = plt.figure(figsize=(11.0, 8.6))
gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.1], height_ratios=[0.82, 1.2],
                      wspace=0.10, hspace=0.30)

# (a) raster
axa = fig.add_subplot(gs[0, 0]); axa.imshow(raster_rgb); axa.axis("off")
axa.set_title("(a) Rasterised floor plan", fontsize=13, fontweight="bold")

# (b) predicted label map
axb = fig.add_subplot(gs[0, 1]); axb.imshow(lab_rgb); axb.axis("off")
axb.set_title("(b) M1 predicted label map", fontsize=13, fontweight="bold")

# (c) typed proximity chart (square grid)
axc = fig.add_subplot(gs[1, 0])
order = ["Living", "Kitchen", "Balcony", "Bathroom", "Bedroom"]
ids = sorted(R, key=lambda i: (next((k for k, o in enumerate(order) if nm[i].startswith(o)), 9), nm[i]))
labels = [disp(nm[i]) for i in ids]; n = len(ids)
et = {frozenset({a, b}): t for (a, b), t in edges.items()}
Cc = {"door": "#2E7D32", "open": "#8E24AA", "shared-wall": "#F9A825", None: "#ECECEC"}
SYM = {"door": "D", "open": "OP", "shared-wall": "SW", None: ""}
for r in range(n):
    for c in range(n):
        if r == c:
            axc.add_patch(Rectangle((c, n-1-r), 1, 1, facecolor="#455A64", edgecolor="white", lw=1.2)); continue
        t = et.get(frozenset({ids[r], ids[c]}))
        axc.add_patch(Rectangle((c, n-1-r), 1, 1, facecolor=Cc.get(t, Cc[None]), edgecolor="white", lw=1.2))
        if t: axc.text(c+0.5, n-1-r+0.5, SYM[t], ha="center", va="center", fontsize=11, fontweight="bold", color="white")
axc.set_xlim(0, n); axc.set_ylim(0, n); axc.set_aspect("equal")
axc.set_xticks([i+0.5 for i in range(n)]); axc.set_xticklabels(labels, rotation=45, ha="left", fontsize=10); axc.xaxis.tick_top()
axc.set_yticks([i+0.5 for i in range(n)]); axc.set_yticklabels(labels[::-1], fontsize=10); axc.tick_params(length=0)
for sp in axc.spines.values(): sp.set_visible(False)
axc.set_title("(c) M3 adjacency matrix", fontsize=13, fontweight="bold", y=1.32)

# (d) bubble diagram (geographic)
axd = fig.add_subplot(gs[1, 1])
pos = {}; area = {}
for i, (c, cm) in R.items():
    ys, xs = np.where(cm); pos[i] = (xs.mean(), -ys.mean()); area[i] = cm.sum()*scale if scale else cm.sum()
amax = max(area.values())
for e, t in edges.items():
    a, b = tuple(e); x = [pos[a][0], pos[b][0]]; y = [pos[a][1], pos[b][1]]
    if t == "door": axd.plot(x, y, color="#111", lw=2.4, zorder=1)
    elif t == "open": axd.plot(x, y, color="#8E24AA", lw=4.5, zorder=1)
    else: axd.plot(x, y, color="#9E9E9E", lw=2.0, ls=(0, (1, 3)), zorder=1)
cx0 = np.mean([p[0] for p in pos.values()]); cy0 = np.mean([p[1] for p in pos.values()])
for i in R:
    s = 220 + 1500*(area[i]/amax)
    axd.scatter([pos[i][0]], [pos[i][1]], s=s, c=[col[i]], edgecolors="#333", linewidths=1.3, zorder=3)
    dx = pos[i][0]-cx0; dy = pos[i][1]-cy0; nn = (dx*dx+dy*dy)**0.5 or 1
    ux, uy = dx/nn, dy/nn; off = (s/3.14159)**0.5 + 6
    ha = "left" if ux > 0.35 else ("right" if ux < -0.35 else "center")
    va = "bottom" if uy > 0.35 else ("top" if uy < -0.35 else "center")
    axd.annotate(disp(nm[i]), pos[i], xytext=(off*ux, off*uy), textcoords="offset points", ha=ha, va=va,
                 fontsize=9.5, fontweight="bold", bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="#ccc", lw=0.4, alpha=0.9), zorder=5)
axd.axis("off"); axd.margins(0.28)
axd.set_title("(d) M4 typed bubble diagram", fontsize=13, fontweight="bold")
h = [mlines.Line2D([], [], color="#111", lw=2.4, label="door"),
     mlines.Line2D([], [], color="#8E24AA", lw=4.5, label="open passage"),
     mlines.Line2D([], [], color="#9E9E9E", lw=2.0, ls=(0, (1, 3)), label="shared wall")]
axd.legend(handles=h, loc="lower center", ncol=3, fontsize=11, frameon=False, bbox_to_anchor=(0.5, -0.13))
fig.savefig(sys.argv[1], dpi=160, bbox_inches="tight"); print("saved", sys.argv[1])
