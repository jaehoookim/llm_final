"""
Draw the AI Newsroom pipeline diagram for the slides (the `<pic 1>` asset).

  python architecture.py        # writes results/fig0_architecture.png

Schematic only (no data) — kept as code so the figure is reproducible and
matches the deck's clean black/white + red-accent style.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from utils import load_config

INK = "#111111"
GRAY = "#666666"
RED = "#d6312a"
FILL = "#f3f3f3"
IO_FILL = "#ffffff"


def _box(ax, cx, cy, w, h, title, sub, *, fill=FILL, edge=INK, accent=False):
    lw = 2.2 if accent else 1.3
    ec = RED if accent else edge
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.006,rounding_size=0.02",
        linewidth=lw, edgecolor=ec, facecolor=fill, zorder=2))
    ax.text(cx, cy + h * 0.16, title, ha="center", va="center",
            fontsize=14, fontweight="bold", color=INK, zorder=3)
    ax.text(cx, cy - h * 0.20, sub, ha="center", va="center",
            fontsize=9.5, color=GRAY, zorder=3, linespacing=1.25)


def _arrow(ax, x0, y0, x1, y1, *, color=INK, lw=1.6, rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1),
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-|>", mutation_scale=16,
        linewidth=lw, color=color, linestyle=ls, zorder=1))


def main():
    cfg = load_config()
    rd = cfg["results_dir"]

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    centers = [0.0875, 0.2525, 0.4175, 0.5825, 0.7475, 0.9125]
    y, w, h = 0.62, 0.118, 0.26
    hw = w / 2

    _box(ax, centers[0], y, w, h, "Sources",
         "HackerNews · arXiv\n5 articles", fill=IO_FILL)
    _box(ax, centers[1], y, w, h, "Scout", "collect &\ncluster")
    _box(ax, centers[2], y, w, h, "Reader", "summarize each\n(parallel · 3 lines)")
    _box(ax, centers[3], y, w, h, "Writer", "fuse into\none draft")
    _box(ax, centers[4], y, w, h, "Editor", "score 4 axes\n+ feedback", accent=True)
    _box(ax, centers[5], y, w, h, "Newsletter",
         "1 article", fill=IO_FILL)

    for a, b in zip(centers[:-1], centers[1:]):
        _arrow(ax, a + hw, y, b - hw, y)

    # Feedback loop: Editor -> Writer (below threshold, re-write).
    # Routed as a clear rectangular path under the two boxes.
    yb = y - h / 2            # box bottoms
    yl = yb - 0.20            # loop depth
    ax.plot([centers[4], centers[4]], [yb, yl], color=RED, lw=2.0, zorder=1)
    ax.plot([centers[4], centers[3]], [yl, yl], color=RED, lw=2.0, zorder=1)
    ax.add_patch(FancyArrowPatch(
        (centers[3], yl), (centers[3], yb),
        arrowstyle="-|>", mutation_scale=16,
        linewidth=2.0, color=RED, zorder=1))
    ax.text((centers[3] + centers[4]) / 2, yl - 0.055,
            "score < threshold  →  revise  (K ≤ 2)",
            ha="center", va="center", fontsize=10.5,
            color=RED, fontweight="bold")

    ax.text(0.5, 0.97, "AI Newsroom — four role agents + self-correction loop",
            ha="center", va="center", fontsize=13, fontweight="bold", color=INK)

    out = os.path.join(rd, "fig0_architecture.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[arch] wrote {out}")


if __name__ == "__main__":
    main()
