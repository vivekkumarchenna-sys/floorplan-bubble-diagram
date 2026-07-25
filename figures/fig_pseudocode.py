"""
fig_pseudocode.py — Figure D.1: Graph Construction Pseudocode (Algorithm 1)
============================================================================
Renders typed connectivity graph construction algorithm as a publication-quality
pseudocode figure. No project dependencies — runs anywhere.

Usage:
    python fig_pseudocode.py
    python fig_pseudocode.py --out fig_d1_pseudocode.pdf
"""

import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ── Pseudocode lines ─────────────────────────────────────────────────────────
# Each entry: (indent_level, text, is_comment)
ALGORITHM = [
    (0, "Algorithm 1: Typed Connectivity Graph Construction", False),
    (0, "", False),
    (0, "Input:  Segmentation mask M of size H x W, class labels in {0, ..., 15}", False),
    (0, "Output: Typed room-adjacency graph G = (V, E)", False),
    (0, "Params: dilation d=15, door_min=15, wall_min=20, arch_min=30", False),
    (0, "", False),
    (0, "// Phase 1: Room Instance Extraction", True),
    (1, "V <- {}", False),
    (1, "for each room class c in {1, 2, ..., 9} do", False),
    (2, "B_c <- (M = c)                                    // binary mask for class c", False),
    (2, "components <- ConnectedComponents(B_c)", False),
    (2, "for each component K with area(K) >= 100 px do", False),
    (3, "v <- new node(class=c, mask=K, area=area(K), centroid=centroid(K))", False),
    (3, "V <- V + {v}", False),
    (0, "", False),
    (0, "// Phase 2: Door Mask Extraction", True),
    (1, "D <- (M = 11) OR (M = 13)                        // Door + FrontDoor pixels", False),
    (0, "", False),
    (0, "// Phase 3: Pairwise Adjacency Detection", True),
    (1, "E <- {}", False),
    (1, "for each room v_i in V do", False),
    (2, "R_i <- Dilate(mask(v_i), kernel=2d+1)             // square dilation", False),
    (1, "for each pair (v_i, v_j) in V x V, i < j do", False),
    (2, "O <- R_i AND R_j                                  // overlap region", False),
    (2, "if |O| < wall_min then continue                   // not adjacent", False),
    (0, "", False),
    (0, "// Phase 4: Edge Type Classification", True),
    (2, "d_px <- |O AND D|                                  // door pixels in overlap", False),
    (2, "S <- O \\ Wall \\ mask(v_i) \\ mask(v_j)              // opening region", False),
    (2, "w <- MinorAxis(LargestContour(S))                  // opening width", False),
    (2, "", False),
    (2, "if d_px >= door_min then", False),
    (3, 'type <- "door"', False),
    (2, "else if w >= arch_min then", False),
    (3, 'type <- "arch"', False),
    (2, "else", False),
    (3, 'type <- "shared-wall"', False),
    (2, "", False),
    (2, "E <- E + {(v_i, v_j, type, |O|, d_px, w)}", False),
    (0, "", False),
    (1, "return G = (V, E)", False),
]


def generate_pseudocode(save_path="fig_d1_pseudocode.pdf"):
    # calculate figure height based on number of lines
    n_lines = len(ALGORITHM)
    line_height = 0.032
    fig_height = max(4, n_lines * line_height + 0.8)

    fig, ax = plt.subplots(figsize=(8.5, fig_height))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # background box
    box_top = 0.97
    box_bot = 0.03
    ax.add_patch(plt.Rectangle(
        (0.02, box_bot), 0.96, box_top - box_bot,
        fill=True, facecolor="#FAFAFA", edgecolor="#333333",
        linewidth=1.2, zorder=1,
    ))

    # render lines
    y = box_top - 0.04
    indent_px = 0.03
    dy = (box_top - box_bot - 0.06) / max(n_lines - 1, 1)

    for indent, text, is_comment in ALGORITHM:
        if text == "":
            y -= dy * 0.5
            continue

        x = 0.05 + indent * indent_px

        # algorithm title
        if text.startswith("Algorithm 1"):
            ax.text(0.5, y, text, ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    fontfamily="serif", zorder=2)
            # underline
            ax.plot([0.08, 0.92], [y - dy * 0.35, y - dy * 0.35],
                    color="#333333", linewidth=0.6, zorder=2)
            y -= dy
            continue

        # Input/Output/Params lines
        if text.startswith(("Input:", "Output:", "Params:")):
            colon_idx = text.index(":")
            ax.text(x, y, text[:colon_idx + 1], ha="left", va="center",
                    fontsize=9, fontweight="bold",
                    fontfamily="serif", zorder=2)
            ax.text(x + 0.065, y, text[colon_idx + 1:], ha="left", va="center",
                    fontsize=9, fontfamily="serif", zorder=2)
            y -= dy
            continue

        if is_comment:
            ax.text(x, y, text, ha="left", va="center",
                    fontsize=9, fontstyle="italic",
                    color="#2E7D32", fontfamily="serif", zorder=2)
            y -= dy
            continue

        # split trailing inline comments (after //)
        if "  //" in text:
            parts = text.split("  //", 1)
            main_text = parts[0].rstrip()
            comment = "// " + parts[1].strip()

            ax.text(x, y, main_text, ha="left", va="center",
                    fontsize=9, fontfamily="monospace", zorder=2)
            ax.text(0.95, y, comment, ha="right", va="center",
                    fontsize=8, fontstyle="italic",
                    color="#888888", fontfamily="serif", zorder=2)
        else:
            ax.text(x, y, text, ha="left", va="center",
                    fontsize=9, fontfamily="monospace", zorder=2)

        y -= dy

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved -> {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="fig_d1_pseudocode.pdf")
    args = parser.parse_args()
    generate_pseudocode(args.out)
