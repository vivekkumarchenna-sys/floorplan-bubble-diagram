# -*- coding: utf-8 -*-
"""Fig 7: annotation mismatch on plan 16649. Corrected GT | ResPlan | Pipeline.
Each panel draws edges in the STYLE of the type that graph assigns (door=solid,
open=thick purple, shared-wall=dotted, absent=no line); an edge is RED when its
type disagrees with the corrected GT. Labels sit outside the bubbles."""
import sys; sys.path.insert(0, "src")
import cv2, numpy as np, json, torch, re
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
# Embed TrueType (not Type 3) fonts so the released PDFs match the submitted artwork.
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
import matplotlib.lines as mlines
import truegraph_builder as tg
from build_gt_graph import build_gt_graph_from_resplan, load_resplan_records
from inference import load_model, predict_mask

stem = "16649"
mask = cv2.imread(f"data/resplan_masks/{stem}_mask.png", 0).astype(np.int64)
raster = cv2.imread(f"data/resplan_raster/{stem}.png")
R, gt_edges = tg.build_true_graph(mask); nm = tg.name_map(R, mask)
pos = {}; area = {}
for i, (c, cm) in R.items():
    ys, xs = np.where(cm); pos[i] = (xs.mean(), -ys.mean()); area[i] = int(cm.sum())
cents = {i: (pos[i][0], -pos[i][1]) for i in R}
raster_rgb = cv2.cvtColor(raster, cv2.COLOR_BGR2RGB)
def rcol(cm):
    px = raster_rgb[cm]; v, c = np.unique(px.reshape(-1, 3), axis=0, return_counts=True)
    r, g, b = v[c.argmax()]; return (r / 255, g / 255, b / 255)
col = {i: rcol(cm) for i, (c, cm) in R.items()}
G1 = {frozenset({a, b}): t for (a, b), t in gt_edges.items()}

# ResPlan mapped by class+area rank
recs = load_resplan_records("data/resplan_raw/ResPlan.pkl"); Gr = build_gt_graph_from_resplan(recs[int(stem)])
mbc = {}
for i, (c, cm) in R.items(): mbc.setdefault(c, []).append((i, int(cm.sum())))
for c in mbc: mbc[c].sort(key=lambda t: t[1])
rbc = {}
for u in Gr.nodes(): rbc.setdefault(Gr.nodes[u]["class_id"], []).append((u, Gr.nodes[u].get("area", 0)))
for c in rbc: rbc[c].sort(key=lambda t: t[1])
rp2m = {}
for c in rbc:
    for k, (u, _) in enumerate(rbc[c]):
        if c in mbc and k < len(mbc[c]): rp2m[u] = mbc[c][k][0]
G2 = {}
for u, v, dd in Gr.edges(data=True):
    if u in rp2m and v in rp2m and rp2m[u] != rp2m[v]: G2[frozenset({rp2m[u], rp2m[v]})] = dd.get("edge_type")

# Pipeline = geometric M2 on predicted mask
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model("checkpoints/segformer/best_model.pth", DEV)
pred = predict_mask(model, raster, DEV)
Rp, pe = tg.build_true_graph(pred)
cp = {i: (np.where(cm)[1].mean(), np.where(cm)[0].mean()) for i, (c, cm) in Rp.items()}
mp = {}
for i, (ci, cmi) in Rp.items():
    best, bd = None, 900
    for j, (cj, cmj) in R.items():
        if cj != ci: continue
        dd = (cp[i][0] - cents[j][0]) ** 2 + (cp[i][1] - cents[j][1]) ** 2
        if dd < bd: bd, best = dd, j
    if best is not None: mp[i] = best
G3 = {}
for (a, b), t in pe.items():
    if a in mp and b in mp and mp[a] != mp[b]: G3[frozenset({mp[a], mp[b]})] = t

def disp(s): return re.sub(r"([A-Za-z])(\d)$", r"\1 \2", s)

def zigzag(ax, p1, p2, color="#E57373", lw=2.0, amp=11.0, pitch=26.0):
    """Break-line (zigzag) between p1 and p2: a broken / missing connection."""
    x1, y1 = p1; x2, y2 = p2
    L = np.hypot(x2 - x1, y2 - y1)
    px, py = -(y2 - y1) / L, (x2 - x1) / L
    k = 2 * max(3, int(L / pitch))
    xs, ys = [], []
    for i in range(k + 1):
        t = i / k
        cx, cy = x1 + t * (x2 - x1), y1 + t * (y2 - y1)
        o = 0.0 if i in (0, k) else (amp if i % 2 else -amp)
        xs.append(cx + o * px); ys.append(cy + o * py)
    ax.plot(xs, ys, color=color, lw=lw, zorder=1, solid_capstyle="round")
STYLE = {"door": ("#111111", "-", 2.8), "open": ("#8E24AA", "-", 5.0),
         "shared-wall": ("#9E9E9E", (0, (1, 3)), 2.4), "window": ("#00897B", "-", 1.8)}
amax = max(area.values())
scale = json.load(open("pixel_scale.json")).get(str(stem))
cx0 = np.mean([p[0] for p in pos.values()]); cy0 = np.mean([p[1] for p in pos.values()])

def draw(ax, edges, title, ref=None, marks=False):
    # missing edges: present in the reference (GT) but absent here -> pale "ghost" line + hollow ✗
    if marks and ref is not None:
        for e, t in ref.items():
            if e in edges:
                continue
            a, b = tuple(e)
            zigzag(ax, pos[a], pos[b])          # broken / missing connection
            mx, my = (pos[a][0] + pos[b][0]) / 2, (pos[a][1] + pos[b][1]) / 2
            ax.text(mx, my, "✗", color="#D32F2F", fontsize=11, fontweight="bold",
                    ha="center", va="center", zorder=6,
                    bbox=dict(boxstyle="circle,pad=0.12", fc="#FDECEA", ec="#D32F2F", lw=1.0))
    for e, t in edges.items():
        a, b = tuple(e); x = [pos[a][0], pos[b][0]]; y = [pos[a][1], pos[b][1]]
        c, ls, lw = STYLE.get(t, ("#9E9E9E", "-", 2.0))
        wrong = ref is not None and (e not in ref or ref[e] != t)
        if wrong: c = "#D32F2F"                     # keep the type's linestyle, colour red
        ax.plot(x, y, color=c, ls=ls, lw=lw, solid_capstyle="round", zorder=2 if wrong else 1)
        if marks:                                   # ✓ correct / ✗ wrong at the edge midpoint
            mx, my = (x[0] + x[1]) / 2, (y[0] + y[1]) / 2
            ax.text(mx, my, "✗" if wrong else "✓",
                    color="#D32F2F" if wrong else "#2E7D32",
                    fontsize=13, fontweight="bold", ha="center", va="center", zorder=6,
                    bbox=dict(boxstyle="circle,pad=0.12", fc="white",
                              ec="#D32F2F" if wrong else "#2E7D32", lw=1.2))
    for i in R:
        s = 260 + 1700 * (area[i] / amax)
        ax.scatter([pos[i][0]], [pos[i][1]], s=s, c=[col[i]], edgecolors="#333", linewidths=1.5, zorder=3)
        lbl = f"{disp(nm[i])}\n{area[i]*scale:.1f} m²" if scale else disp(nm[i])
        dx = pos[i][0] - cx0; dy = pos[i][1] - cy0; n = (dx * dx + dy * dy) ** 0.5 or 1
        ux, uy = dx / n, dy / n; off = (s / 3.14159) ** 0.5 + 8
        ha = "left" if ux > 0.35 else ("right" if ux < -0.35 else "center")
        va = "bottom" if uy > 0.35 else ("top" if uy < -0.35 else "center")
        ax.annotate(lbl, pos[i], xytext=(off * ux, off * uy), textcoords="offset points",
                    ha=ha, va=va, fontsize=8.5, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#ccc", lw=0.5, alpha=0.92), zorder=5)
    ax.set_title(title, fontsize=12.5, fontweight="bold", pad=12); ax.axis("off"); ax.margins(0.32)

# pipeline reproduces the corrected GT exactly on this plan (Edge F1 0.998 dataset-wide),
# so a separate pipeline panel would duplicate the corrected-GT panel. Two panels only.
assert G3 == G1, "pipeline != corrected GT on 16649 - restore 3rd panel"
fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.9))
draw(axes[0], G1, "Geometry-based ground truth")
draw(axes[1], G2, "ResPlan released graph", ref=G1, marks=True)
h = [mlines.Line2D([], [], color="#111", lw=2.8, label="door"),
     mlines.Line2D([], [], color="#8E24AA", lw=5.0, label="open passage"),
     mlines.Line2D([], [], color="#9E9E9E", lw=2.4, ls=(0, (1, 3)), label="shared wall"),
     mlines.Line2D([], [], color="#E57373", lw=2.0, label="missing connection (absent in ResPlan)"),
     mlines.Line2D([], [], marker="X", color="w", markerfacecolor="#D32F2F", markersize=12,
                   label="wrong / missing vs ground truth")]
fig.legend(handles=h, loc="lower center", ncol=4, fontsize=11, frameon=True, bbox_to_anchor=(0.5, 0.02))
fig.suptitle("Plan 16649 - ResPlan types three real Living-room doors as shared walls (✗); the pipeline recovers them",
             fontsize=13.5, fontweight="bold")
fig.tight_layout(rect=[0, 0.15, 1, 0.94])
fig.savefig(sys.argv[1], dpi=150, bbox_inches="tight"); print("saved", sys.argv[1])
