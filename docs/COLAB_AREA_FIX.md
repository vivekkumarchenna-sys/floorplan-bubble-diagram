# Colab: Fix Room-Area Scaling — Run Instructions

Regenerates `pixel_scale.json` with correct `id`-based matching so bubble-diagram
m² labels are right. **No metrics change** — area never enters graph construction.

Root cause and background: `docs/AREA_SCALING_PROBLEM.md`.
Full plan: room-area scaling fix (labels only, balcony excluded, no refactor).

---

## 0. Upload

Upload the edited **`src/build_pixel_scale.py`** to your Colab project. It now
auto-detects the project root, so it works whether you place it in `src/` or at
the project root.

Set your project path once:

```python
from google.colab import drive
drive.mount('/content/drive')

PROJECT = "/content/drive/MyDrive/bubble_diagram_project"   # adjust if different
```

**Speed tip:** reading 17k masks off a Drive mount is slow. Copy masks to local
disk first (one-time, ~1–2 min):

```python
import shutil, os
LOCAL_MASKS = "/content/resplan_masks"
if not os.path.exists(LOCAL_MASKS):
    shutil.copytree(f"{PROJECT}/data/resplan_masks", LOCAL_MASKS)
print("masks ready:", len(os.listdir(LOCAL_MASKS)))
```

---

## 1. Dry run — verify gates BEFORE writing anything

```python
!python {PROJECT}/src/build_pixel_scale.py --dry-run \
    --root {PROJECT} --mask-dir /content/resplan_masks
```

**Acceptance gates — all three must hold, otherwise stop and report:**

| Check | Expected |
|---|---|
| Coverage | ~60–65% of split stems derived |
| `512px frame: median` | **10–20 m** (broken file is 22–32 m on affected plans) |
| Skip breakdown | dominated by `bad_net_area` (~⅓) and `fp_mismatch` — **not** `no_entry` |

If `no_entry` is large, the stem→plan mapping is off — do not proceed, send the
output back.

---

## 2. Regenerate `pixel_scale.json`

Only after step 1 passes:

```python
!python {PROJECT}/src/build_pixel_scale.py \
    --root {PROJECT} --mask-dir /content/resplan_masks
```

- Auto-backs up the old file to `pixel_scale.backup-<timestamp>.json` (reversible).
- Writes ~10–11k entries (down from 17,107). Dropped stems are intentionally absent.

---

## 3. Spot-check the new scale (no writes)

```python
import json, numpy as np, cv2
from pathlib import Path

root = Path(PROJECT)
scale = json.load(open(root/"pixel_scale.json"))
print("entries:", len(scale))

# frame sizes on known plans — expect ~15–17 m (were 23 / 32 m)
for s in ["0", "10005"]:
    v = scale.get(s)
    if v: print(f"  plan {s}: {(v**0.5)*512:.1f} m frame")

# bathroom median across covered plans — expect ~3.5–4.5 m² per bathroom
from scipy import ndimage
bath = []
for s in list(scale)[:600]:
    m = cv2.imread(str(root/"data/resplan_masks"/f"{s}_mask.png"), 0)
    if m is None: continue
    lab, n = ndimage.label(m == 2)                 # class 2 = Bathroom
    if n == 0: continue
    sizes = np.bincount(lab.ravel())[1:]
    bath += [sz*scale[s] for sz in sizes if sz >= 100]
b = np.array(bath)
print(f"bathroom m²: median={np.median(b):.2f}  p25={np.percentile(b,25):.2f}  p75={np.percentile(b,75):.2f}")

# carpet consistency: interior room areas should sum ~= net_area (exact by construction)
import pickle
data = pickle.load(open(root/"data/resplan_raw/ResPlan.pkl","rb"))
by_id = {e["id"]: i for i,e in enumerate(data)}
INT = [1,2,3,4,6,7]
s = "0"
m = cv2.imread(str(root/"data/resplan_masks"/f"{s}_mask.png"), 0)
interior_px = int(np.isin(m, INT).sum())
print(f"plan {s}: interior_px*scale = {interior_px*scale[s]:.1f} m²  vs net_area = {data[by_id[int(s)]]['net_area']:.1f} m²")
```

Pass criteria: frames ~15–17 m, bathroom median ~3.5–4.5, interior-sum ≈ net_area.

---

## 4. Re-render downstream artifacts (needs the trained checkpoint)

The consumers already read the scale correctly — just re-run them.

```python
# Sample bubble diagrams → Output/*.png
!python {PROJECT}/src/generate_bubble.py --image {PROJECT}/data/resplan_raster/ --limit 30

# Survey stimuli (pipeline + GT bubbles)
!python {PROJECT}/src/generate_survey.py

# Fig 6 (qualitative) and Fig 7 (failures)
!python {PROJECT}/Figures/fig_qualitative.py
!python {PROJECT}/Figures/fig_failures.py
```

**Before finalizing Fig 6 / Fig 7 / survey exemplars**, confirm each chosen stem
has a scale (else that diagram shows px² and mixes units):

```python
picks = ["10431", "11045"]   # replace with your actual figure/survey stems
missing = [s for s in picks if s not in scale]
print("stems WITHOUT scale (would show px²):", missing or "none — all covered")
```

Notes:
- Dropped plans fall back to `px²` labels (`visualize.py:148`) — no crash.
- Balcony rooms still get a proportional m² label; balcony is only excluded from
  the carpet *denominator*, not unlabeled.

---

## 5. Regression guard — metrics must be unchanged

Area never touches graph construction, so this must match your existing numbers.
If it drifts, something else changed:

```python
import pandas as pd
print(pd.read_csv(root/"summary.csv").to_string(index=False))
# mIoU / edge_f1 / ged / frobenius identical to before the fix
```

---

## Rollback

Restore the backup:

```python
import shutil, glob
bak = sorted(glob.glob(f"{PROJECT}/pixel_scale.backup-*.json"))[-1]
shutil.copy2(bak, f"{PROJECT}/pixel_scale.json")
print("restored from", bak)
```
