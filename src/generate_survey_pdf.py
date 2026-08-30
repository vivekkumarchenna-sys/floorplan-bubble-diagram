"""
generate_survey_pdf.py - Create a survey PDF from generated stimuli
====================================================================
Reads survey_stimuli.csv and the corresponding images, then builds
a PDF with one stimulus per page: floor plan (left) + bubble diagram
(right) + rating section below.

Usage:
    python generate_survey_pdf.py
    python generate_survey_pdf.py --dir survey_stimuli/ --out survey.pdf
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

try:
    _SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    _SCRIPT_DIR = Path("/content/drive/MyDrive/bubble_diagram_project")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ── Likert scale questions ────────────────────────────────────────────────────
QUESTIONS = [
    "Q1. The bubble diagram accurately represents\n"
    "      the rooms in the floor plan.",
    "Q2. The room adjacencies (connections) are\n"
    "      correctly captured.",
    "Q3. The room sizes in the diagram are proportional\n"
    "      to the actual plan.",
    "Q4. The diagram is easy to read and interpret.",
    "Q5. Overall, I would rate this bubble diagram as\n"
    "      useful for understanding the layout.",
]

SCALE_HEADER = "1 = Strongly Disagree   2 = Disagree   3 = Neutral   4 = Agree   5 = Strongly Agree"


# ══════════════════════════════════════════════════════════════════════════════
# Cover page
# ══════════════════════════════════════════════════════════════════════════════

def _draw_cover(pdf):
    fig = plt.figure(figsize=(8.27, 11.69))  # A4
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # title block
    ax.text(0.5, 0.85, "Floor Plan Bubble Diagram",
            ha="center", va="center", fontsize=24, fontweight="bold")
    ax.text(0.5, 0.80, "User Evaluation Survey",
            ha="center", va="center", fontsize=18, color="#444444")

    # divider
    ax.plot([0.15, 0.85], [0.76, 0.76], color="#CCCCCC", linewidth=0.8)

    # instructions
    lines = [
        "Instructions",
        "",
        "This survey evaluates the quality of automatically generated bubble",
        "diagrams from architectural floor plan images.",
        "",
        "Each page presents:",
        "     - Left:  The original floor plan image",
        "     - Right: The generated bubble diagram",
        "",
        "For each stimulus, please rate the bubble diagram on a scale of",
        "1 (Strongly Disagree) to 5 (Strongly Agree) for each question.",
        "",
        "There are no right or wrong answers. Please provide your honest",
        "assessment based on your understanding of the floor plan.",
        "",
        "Thank you for your participation.",
    ]

    y = 0.70
    for line in lines:
        weight = "bold" if line == "Instructions" else "normal"
        size = 13 if line == "Instructions" else 10.5
        ax.text(0.12, y, line, ha="left", va="center",
                fontsize=size, fontweight=weight)
        y -= 0.027

    # participant info box
    ax.plot([0.15, 0.85], [0.28, 0.28], color="#D6B656", linewidth=0.8)

    info_lines = [
        "Participant Information",
        "",
        "Participant ID:  ____________________          Date:  ____________________",
        "",
        "Background:      [ ] Architect      [ ] Engineer      [ ] Student      [ ] Other: ____________",
        "",
        "Years of experience:  ____________________",
    ]

    y = 0.24
    for line in info_lines:
        weight = "bold" if line == "Participant Information" else "normal"
        size = 12 if line == "Participant Information" else 10
        ax.text(0.12, y, line, ha="left", va="center",
                fontsize=size, fontweight=weight)
        y -= 0.028

    pdf.savefig(fig, facecolor="white")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Stimulus page
# ══════════════════════════════════════════════════════════════════════════════

def _draw_stimulus_page(pdf, stim_id, fp_path, bubble_path):
    fig = plt.figure(figsize=(8.27, 11.69))  # A4

    # use a single axes as canvas for precise positioning
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ── header ────────────────────────────────────────────────────────────────
    ax.text(0.5, 0.965, f"Stimulus {stim_id}",
            ha="center", va="center", fontsize=14, fontweight="bold")
    ax.plot([0.08, 0.92], [0.95, 0.95], color="#CCCCCC", linewidth=0.8)

    # ── images (top 45% of page) ─────────────────────────────────────────────
    # floor plan: left half
    ax_fp = fig.add_axes([0.06, 0.58, 0.42, 0.35])
    if fp_path.exists():
        img_fp = mpimg.imread(str(fp_path))
        ax_fp.imshow(img_fp)
    ax_fp.set_title("Floor Plan", fontsize=11, fontweight="bold", pad=10)
    ax_fp.axis("off")

    # bubble diagram: right half
    ax_bb = fig.add_axes([0.52, 0.58, 0.42, 0.35])
    if bubble_path.exists():
        img_bb = mpimg.imread(str(bubble_path))
        ax_bb.imshow(img_bb)
    ax_bb.set_title("Bubble Diagram", fontsize=11, fontweight="bold", pad=10)
    ax_bb.axis("off")

    # ── divider between images and questions ──────────────────────────────────
    ax.plot([0.08, 0.92], [0.56, 0.56], color="#CCCCCC", linewidth=0.8)

    # ── rating scale header ──────────────────────────────────────────────────
    ax.text(0.5, 0.535, SCALE_HEADER,
            ha="center", va="center", fontsize=8, color="#666666",
            fontstyle="italic")

    # ── questions + circles ──────────────────────────────────────────────────
    y_start = 0.49
    q_spacing = 0.085

    for i, q in enumerate(QUESTIONS):
        y_q = y_start - i * q_spacing

        # question text (left side)
        ax.text(0.08, y_q, q, ha="left", va="center",
                fontsize=9, linespacing=1.3)

        # rating circles (right side, aligned)
        y_circ = y_q
        for val in range(1, 6):
            x_circ = 0.68 + (val - 1) * 0.055
            circle = plt.Circle((x_circ, y_circ), 0.01, fill=False,
                                edgecolor="#333333", linewidth=0.9)
            ax.add_patch(circle)
            ax.text(x_circ, y_circ - 0.018, str(val),
                    ha="center", va="center", fontsize=6.5, color="#666666")

    # ── divider before comments ──────────────────────────────────────────────
    y_comment_top = y_start - len(QUESTIONS) * q_spacing + 0.01
    ax.plot([0.08, 0.92], [y_comment_top, y_comment_top],
            color="#CCCCCC", linewidth=0.5)

    # ── comments section ─────────────────────────────────────────────────────
    y_label = y_comment_top - 0.025
    ax.text(0.08, y_label, "Comments (optional):",
            ha="left", va="center", fontsize=9)

    # comment lines
    for j in range(3):
        y_line = y_label - 0.035 - j * 0.03
        ax.plot([0.08, 0.92], [y_line, y_line],
                color="#E0E0E0", linewidth=0.5, linestyle="-")

    # ── page footer ──────────────────────────────────────────────────────────
    ax.text(0.5, 0.02, f"Page {stim_id.replace('S', '')}",
            ha="center", va="center", fontsize=7, color="#AAAAAA")

    pdf.savefig(fig, facecolor="white")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def generate_pdf(stim_dir, out_path):
    stim_dir = Path(stim_dir)
    csv_path = stim_dir / "survey_stimuli.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    stimuli = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            stimuli.append(row)

    print(f"[survey] {len(stimuli)} stimuli from {csv_path}")

    with PdfPages(out_path) as pdf:
        _draw_cover(pdf)

        for s in stimuli:
            fp_path = stim_dir / "floorplans" / s["Floor_Plan_File"]
            bubble_path = stim_dir / "bubbles" / s["Bubble_Diagram_File"]
            _draw_stimulus_page(pdf, s["Stimulus_ID"], fp_path, bubble_path)

    print(f"Saved -> {out_path}  ({len(stimuli) + 1} pages)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate survey PDF from stimuli images")
    parser.add_argument("--dir", type=str,
                        default=str(_SCRIPT_DIR / "survey_stimuli"),
                        help="Directory with floorplans/, bubbles/, survey_stimuli.csv")
    parser.add_argument("--out", type=str,
                        default=str(_SCRIPT_DIR / "survey.pdf"),
                        help="Output PDF path")
    args = parser.parse_args()

    generate_pdf(args.dir, args.out)


if __name__ == "__main__":
    main()
