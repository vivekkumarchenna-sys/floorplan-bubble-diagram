# -*- coding: utf-8 -*-
"""Batch-render the three categorical views (proximity matrix, proximity chart,
bubble diagram) for every plan in a split. Typed palette throughout; the matrix
and chart carry no areas. Parallel across CPU cores.

Usage (run from the repo root, which holds src/ and data/):
    python CORRECTED_GT_METHOD/batch_render_test.py <out_dir> [split_file] [limit]
"""
import sys, os, json, re
sys.path.insert(0, "src")
import cv2, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.patches import Rectangle, Patch
from concurrent.futures import ProcessPoolExecutor, as_completed

OUT = sys.argv[1] if len(sys.argv) > 1 else "test_views"
SPLIT = sys.argv[2] if len(sys.argv) > 2 else "data/splits/test.txt"
LIMIT = int(sys.argv[3]) if len(sys.argv) > 3 else 0

COL = {"door": "#2E7D32", "open": "#8E24AA", "shared-wall": "#F9A825", None: "#ECECEC"}
SYM = {"door": "D", "open": "OP", "shared-wall": "SW", None: ""}
ORDER = ["Living", "Kitchen", "Balcony", "Bathroom", "Bedroom"]

def _disp(s): return re.sub(r"([A-Za-z])(\d)$", r"\1 \2", s)

def _load(stem):
    import truegraph_builder as tg
    mask = cv2.imread(f"data/resplan_masks/{stem}_mask.png", 0).astype(np.int64)
    R, edges = tg.build_true_graph(mask)
    nm = tg.name_map(R, mask)
    ids = sorted(R, key=lambda i: (next((k for k, o in enumerate(ORDER) if nm[i].startswith(o)), 9), nm[i]))
    return mask, R, edges, nm, ids

def render_chart(stem, out):
    mask, R, edges, nm, ids = _load(stem)
    n = len(ids)
    labels = [_disp(nm[i]) for i in ids]
    et = {frozenset({a, b}): t for (a, b), t in edges.items()}
    fig, ax = plt.subplots(figsize=(8.6, 7.4))
    for r in range(n):
        for c in range(n):
            if r == c:
                ax.add_patch(Rectangle((c, n-1-r), 1, 1, facecolor="#455A64", edgecolor="white", lw=1.5)); continue
            t = et.get(frozenset({ids[r], ids[c]}))
            ax.add_patch(Rectangle((c, n-1-r), 1, 1, facecolor=COL.get(t, COL[None]), edgecolor="white", lw=1.5))
            if t:
                ax.text(c+0.5, n-1-r+0.5, SYM[t], ha="center", va="center", fontsize=13, fontweight="bold", color="white")
    ax.set_xlim(0, n); ax.set_ylim(0, n); ax.set_aspect("equal")
    ax.set_xticks([i+0.5 for i in range(n)]); ax.set_xticklabels(labels, rotation=45, ha="left", fontsize=11); ax.xaxis.tick_top()
    ax.set_yticks([i+0.5 for i in range(n)]); ax.set_yticklabels(labels[::-1], fontsize=11); ax.tick_params(length=0)
    for sp in ax.spines.values(): sp.set_visible(False)
    leg = [Patch(fc=COL["door"], label="door (D)"), Patch(fc=COL["open"], label="open passage (OP)"),
           Patch(fc=COL["shared-wall"], label="shared wall (SW)"), Patch(fc=COL[None], label="not connected")]
    ax.legend(handles=leg, loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=4, fontsize=10, frameon=False)
    ax.set_title(f"Typed adjacency matrix - plan {stem}", fontsize=13, fontweight="bold", pad=14)
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)

def render_matrix(stem, out):
    mask, R, edges, nm, ids = _load(stem)
    n = len(ids)
    labels = [_disp(nm[i]) for i in ids]
    et = {frozenset({a, b}): t for (a, b), t in edges.items()}
    def center(i, j): return (j - i, -(i + j))
    fig, ax = plt.subplots(figsize=(10, 9))
    for i in range(n):
        for j in range(i + 1, n):
            cx, cy = center(i, j)
            dia = [(cx+1, cy), (cx, cy+1), (cx-1, cy), (cx, cy-1), (cx+1, cy)]
            ax.plot([p[0] for p in dia], [p[1] for p in dia], color="#c9c9c9", lw=0.9, zorder=1)
            t = et.get(frozenset({ids[i], ids[j]}))
            ax.scatter([cx], [cy], s=430, c=COL.get(t, COL[None]), edgecolors="#555", linewidths=1.1, zorder=3)
    for i in range(n):
        ax.text(-1.6, -2*i, labels[i], ha="right", va="center", fontsize=12, fontweight="bold")
        ax.plot([-1.4, 1], [-2*i, -2*i], color="#888", lw=1.0, zorder=2)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(-5.5, n + 0.5); ax.set_ylim(-2*(n-1) - 1.5, 2.2)
    ax.set_title(f"Typed proximity chart - plan {stem}", fontsize=16, fontweight="bold", pad=6)
    h = [mlines.Line2D([], [], marker="o", ls="", ms=13, mec="#555", mfc=COL["door"], label="door"),
         mlines.Line2D([], [], marker="o", ls="", ms=13, mec="#555", mfc=COL["open"], label="open passage"),
         mlines.Line2D([], [], marker="o", ls="", ms=13, mec="#555", mfc=COL["shared-wall"], label="shared-wall"),
         mlines.Line2D([], [], marker="o", ls="", ms=13, mec="#555", mfc=COL[None], label="not connected")]
    ax.legend(handles=h, loc="upper right", fontsize=11, frameon=True, title="Adjacency", title_fontsize=12)
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)

_SCALE = None
def render_bubble(stem, out):
    global _SCALE
    if _SCALE is None:
        try: _SCALE = json.load(open("pixel_scale.json"))
        except Exception: _SCALE = {}
    mask, R, edges, nm, ids = _load(stem)
    raster = cv2.cvtColor(cv2.imread(f"data/resplan_raster/{stem}.png"), cv2.COLOR_BGR2RGB)
    scale = _SCALE.get(str(stem))
    def rcol(cm):
        px = raster[cm]; v, c = np.unique(px.reshape(-1, 3), axis=0, return_counts=True)
        r, g, b = v[c.argmax()]; return (r/255, g/255, b/255)
    col = {i: rcol(cm) for i, (c, cm) in R.items()}
    pos = {}; area = {}
    for i, (c, cm) in R.items():
        ys, xs = np.where(cm); pos[i] = (xs.mean(), -ys.mean()); area[i] = cm.sum()*scale if scale else cm.sum()
    amax = max(area.values()) if area else 1
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
        lbl = f"{_disp(nm[i])}\n{area[i]:.1f} m2" if scale else _disp(nm[i])
        dx = pos[i][0]-cx0; dy = pos[i][1]-cy0; nrm = (dx*dx+dy*dy)**0.5 or 1
        ux, uy = dx/nrm, dy/nrm; off = (s/3.14159)**0.5 + 8
        ha = "left" if ux > 0.35 else ("right" if ux < -0.35 else "center")
        va = "bottom" if uy > 0.35 else ("top" if uy < -0.35 else "center")
        ax.annotate(lbl, pos[i], xytext=(off*ux, off*uy), textcoords="offset points", ha=ha, va=va,
                    fontsize=9.5, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#bbb", lw=0.6, alpha=0.92), zorder=5)
    ax.set_title(f"Typed bubble diagram - plan {stem}", fontsize=14, fontweight="bold")
    ax.axis("off"); ax.margins(0.2)
    h = [mlines.Line2D([], [], color="#111", lw=2.8, label="door"),
         mlines.Line2D([], [], color="#8E24AA", lw=5.0, label="open passage"),
         mlines.Line2D([], [], color="#9E9E9E", lw=2.4, ls=(0, (1, 3)), label="shared wall")]
    ax.legend(handles=h, loc="lower center", ncol=3, fontsize=10, frameon=True, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)

def work(stem):
    try:
        _, R, _, _, ids = _load(stem)
        if len(ids) < 2: return (stem, "skip<2rooms")
        render_matrix(stem, os.path.join(OUT, f"{stem}_chart.png"))   # diamond = proximity chart
        render_chart(stem, os.path.join(OUT, f"{stem}_matrix.png"))   # square = adjacency matrix
        render_bubble(stem, os.path.join(OUT, f"{stem}_bubble.png"))
        return (stem, "ok")
    except Exception as e:
        return (stem, "ERR:" + str(e)[:80])

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    stems = [l.strip() for l in open(SPLIT) if l.strip()]
    if LIMIT: stems = stems[:LIMIT]
    done = 0; errs = []
    with ProcessPoolExecutor(max_workers=min(20, os.cpu_count() or 4)) as ex:
        futs = {ex.submit(work, s): s for s in stems}
        for f in as_completed(futs):
            s, status = f.result(); done += 1
            if status != "ok": errs.append((s, status))
            if done % 200 == 0: print(f"  {done}/{len(stems)} done, {len(errs)} skipped/err", flush=True)
    print(f"DONE: {done} plans, {len(errs)} skipped/err")
    for s, st in errs[:20]: print("   ", s, st)
    json.dump({"done": done, "errors": errs}, open(os.path.join(OUT, "_render_log.json"), "w"))
