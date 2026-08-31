"""
user_study_stats.py - reproduce the user-study table (paper Table E.4).
=======================================================================
The twenty raters each scored both conditions, so pipeline and ground-truth
ratings are paired within rater. The test is therefore a Wilcoxon signed-rank
test on the twenty participant-level condition means, not a Mann-Whitney U test
over the two thousand individual rating cells: those cells are repeated
measures from the same twenty people and are not independent observations.

An earlier version of this analysis used Mann-Whitney over the cells. It
reported plausibility at p = 0.002 and layout readability at p = 0.006, both
surviving Holm correction. Under the paired test the same data give p = 0.020
and p = 0.017, and neither survives Holm correction across the five dimensions
(both reach 0.083). The paper reports the paired result.

Reads the workbook deposited with the Zenodo archive:
    User_Studies/original_pilot_study/user_study_workbook_latest.xlsx

usage:
    python src/user_study_stats.py --workbook path/to/user_study_workbook_latest.xlsx
"""
from __future__ import annotations

import argparse

import numpy as np
from scipy import stats

try:
    import openpyxl
except ImportError as exc:  # pragma: no cover
    raise SystemExit("openpyxl is required: pip install openpyxl") from exc

# sheet labels -> the names used in the paper
DIMENSIONS = [
    ("Plausibility", "Plausibility"),
    ("Adj Correctness", "Adjacency correctness"),
    ("Room Type Accuracy", "Room-type accuracy"),
    ("Size Proportionality", "Size proportionality"),
    ("Readability", "Layout readability"),
]


def load(workbook: str) -> dict[tuple[str, str], np.ndarray]:
    """Return {(dimension, source): stimuli x participants array of ratings}."""
    ws = openpyxl.load_workbook(workbook, data_only=True)["Raw Ratings"]
    rows = list(ws.iter_rows(min_row=1, values_only=True))
    n_participants = len([c for c in rows[0][3:] if c])
    out: dict[tuple[str, str], list] = {}
    for r in rows[1:]:
        if not r[0]:
            continue
        key = (r[1].strip(), r[2].strip())
        out.setdefault(key, []).append([int(x) for x in r[3:3 + n_participants]])
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


def holm(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values, in the input order."""
    m = len(pvalues)
    order = np.argsort(pvalues)
    adjusted = [0.0] * m
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, (m - k) * pvalues[i])
        adjusted[i] = min(running, 1.0)
    return adjusted


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workbook", required=True,
                    help="user_study_workbook_latest.xlsx from the Zenodo archive")
    args = ap.parse_args()

    data = load(args.workbook)
    rows, pvalues = [], []
    for sheet_name, paper_name in DIMENSIONS:
        pipeline = data[(sheet_name, "Pipeline")]
        truth = data[(sheet_name, "GT")]
        if pipeline.shape != truth.shape:
            raise SystemExit(f"{paper_name}: unbalanced design {pipeline.shape} vs {truth.shape}")

        # collapse the ten stimuli per condition into one score per participant
        per_participant_pipeline = pipeline.mean(axis=0)
        per_participant_truth = truth.mean(axis=0)

        test = stats.wilcoxon(per_participant_pipeline, per_participant_truth,
                              alternative="two-sided", zero_method="wilcox")

        # matched-pairs rank-biserial correlation; positive favours ground truth
        differences = per_participant_truth - per_participant_pipeline
        nonzero = differences[differences != 0]
        ranks = stats.rankdata(np.abs(nonzero))
        positive, negative = ranks[nonzero > 0].sum(), ranks[nonzero < 0].sum()
        effect = (positive - negative) / (positive + negative)

        rows.append((paper_name, pipeline.mean(), pipeline.std(ddof=1),
                     truth.mean(), truth.std(ddof=1), test.statistic, test.pvalue, effect))
        pvalues.append(test.pvalue)

    adjusted = holm(pvalues)

    print(f"Wilcoxon signed-rank, paired at participant level "
          f"(N = {data[('Plausibility', 'Pipeline')].shape[1]} participants)\n")
    header = f"{'Dimension':<22}{'Pipeline':>14}{'Ground truth':>15}{'W':>7}{'p':>9}{'p (Holm)':>10}{'r':>8}"
    print(header)
    print("-" * len(header))
    for (name, pm, psd, gm, gsd, w, p, r), padj in zip(rows, adjusted):
        print(f"{name:<22}{pm:>7.2f} ± {psd:.2f}{gm:>8.2f} ± {gsd:.2f}"
              f"{w:>7.1f}{p:>9.4f}{padj:>10.4f}{r:>8.3f}")
    print("\nNo dimension reaches significance after Holm correction for five comparisons.")


if __name__ == "__main__":
    main()
