# The Room-Area Scaling Problem in the Bubble Diagrams

*Plain-language explanation for the project team and supervisor.*

## What the diagrams are supposed to show

Each bubble diagram labels every room with its floor area in square metres
(e.g. "Bedroom 24 m²"). To produce that number, the pipeline counts how many
pixels a room occupies in the 512×512 image and multiplies by a per-plan
conversion factor stored in `pixel_scale.json`:

```
area_in_m² = (number of pixels in the room) × (square metres per pixel for that plan)
```

Every plan needs its own conversion factor, because a large apartment and a
small one are both drawn to fill the same 512×512 frame — so one pixel
represents more real-world area in the larger dwelling.

## The symptom

The printed areas are roughly **2.5× too large**. The clearest evidence is
bathrooms: in the generated diagrams they come out around **12–15 m²**, whereas
real residential bathrooms are almost always **3–6 m²**. Bedrooms and living
rooms were similarly inflated (a "bedroom" of 62 m² is really the size of a
small studio apartment). The inflation was present in every diagram, which
pointed to a systematic error rather than random noise.

## What we ruled out

We first suspected a **resolution mismatch** — that the areas were counted at
one image size but the conversion factor was calibrated at another. We checked,
and this was **not** the cause: the source images, the segmentation masks, and
the model all operate at 512×512. So the pixel counts themselves are correct.

## The actual cause

The dataset (ResPlan) ships each floor plan with its true floor area already
recorded, in square metres, in a field called `net_area`. The correct
conversion factor is simply:

```
square metres per pixel = net_area (from dataset) ÷ (number of interior pixels in that plan)
```

The problem is in **how each plan's true area was matched to its image**. The
dataset is a list of 17,000 plans, and each plan has an `id` field. Critically,
**a plan's position in the list is not the same as its `id`** — for example, the
first entry in the list has `id = 14433`, not `0`. The image files, however, are
named by `id` (`0.png`, `10000.png`, and so on).

The existing `pixel_scale.json` was built by matching each image to the plan at
the **same position in the list**, instead of matching by the `id` field. So
nearly every image was paired with the true area of a **different, unrelated
plan**. Because those areas are effectively random relative to the correct one,
every diagram was mislabelled — and by a different, unpredictable factor each
time.

### Confirmation

For each mask we counted how many rooms of each type it contains
(e.g. "3 bedrooms, 3 bathrooms, 1 kitchen") and compared that fingerprint
against both matching methods on a 60-plan sample:

| Matching method | Plans whose room-counts matched |
| --------------- | ------------------------------- |
| By list position | **6 / 60 (10%)** |
| By `id` field | **57 / 60 (95%)** |

When we recomputed the conversion factor using the correct `id`-based matching,
the bathroom areas fell to a median of about **3.8 m²** — exactly the
architecturally realistic range — and the implied building width dropped from an
implausible 22–32 m to a sensible **15–17 m**.

## A second, separate issue found along the way

The dataset is also partly **corrupt**: about a third of the plans
(**5,458 of 17,000**) have a recorded area of zero, and a few have absurd values
(one lists an area of 79 billion m²). These bad entries have to be filtered out.
After correct matching **and** filtering, roughly **60–65%** of plans yield a
trustworthy area; the rest should simply be shown without an m² label rather than
with a wrong one.

## A related caveat about the bubble sizes themselves

Independent of the numerical labels, the **visual size** of each bubble is not
proportional to room area either. The drawing code rescales bubbles within each
diagram so that the largest room is always drawn at the maximum size and the
smallest at the minimum, regardless of their true areas. So the bubbles
communicate only the *ranking* of rooms within one plan, not their real sizes,
and cannot be compared across diagrams. This matters because one of the
user-study questions explicitly asks reviewers to judge whether "room sizes are
proportional to the actual plan."

## What this affects, and what does not

**Affected (must be regenerated):**
- The m² labels on all generated diagrams
- The survey stimuli
- Any figures showing areas (Figures 6 and 7)

**Not affected (results unchanged):**
- Segmentation accuracy (mIoU)
- Graph structure
- Edge F1 / GED / Frobenius metrics

Area is never used in graph construction, so **none of the core quantitative
results change**.

## The fix

1. Regenerate `pixel_scale.json` using **`id`-based matching** and the
   corruption filter (a script for this has been prepared).
2. Re-render the affected diagrams.
3. *(Optional)* Correct the bubble-drawing code so bubble size is genuinely
   proportional to area, **or** state in the caption that size encodes rank
   rather than absolute area.

## One-sentence summary

The pipeline's geometry is sound, but each plan's true floor area was attached
to the wrong image because of an index-versus-`id` mismatch, compounded by
corrupt entries in the source dataset — both are correctable, and neither
undermines the segmentation or graph-extraction results.
