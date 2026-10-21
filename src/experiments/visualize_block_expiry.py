"""Chart for the block-expiry experiment: what freeing blocks costs a passage that is needed later.

Every point is one (model, context length, window, threshold) from the
dormant-then-needed test, so nothing is selected. The x axis is the share of the
request's blocks freed at the moment the passage is copied back, and the y axis
is the loss in copy accuracy against the full cache. Two rules free the same
number of blocks at each threshold: the RFC's mass-gated expiry, and a positional
sink-plus-recent window that ignores attention.

Palette and marks follow the project's data-viz method. Blue and orange are
categorical slots 1 and 2, validated for colour-vision separation; both series
are direct-labelled and the legend is kept because there are two.
"""

from __future__ import annotations

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.experiments.visualize_corrected import GRID, INK, INK_2, MUTED, S1, S2, SURFACE, _style

OUT_DIR = "reports/charts"
MODELS = {"q05b": ("Qwen2.5-0.5B", "o"), "q15b": ("Qwen2.5-1.5B", "D")}


def _points() -> dict[str, list[tuple[float, float, str]]]:
    """arm -> [(freed, accuracy loss, model tag)]"""
    out: dict[str, list] = {"mass": [], "oldest": []}
    for tag in MODELS:
        for path in sorted(glob.glob(f"reports/experiments/block_expiry_{tag}_n*_w*_*.json")):
            for t in json.load(open(path))["trials"]:
                c, m = t["config"], t["metrics"]
                if c["part"] != "C" or c["arm"] not in out:
                    continue
                out[c["arm"]].append((m["all_blocks_expired_at_copy_start"],
                                      m["copy_acc"] - m["copy_acc_full"], tag))
    return out


def chart_dormant_passage(out_dir: str = OUT_DIR) -> str | None:
    pts = _points()
    if not pts["mass"]:
        return None
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    series = [("oldest", S2, "sink + recent window", -0.012), ("mass", S1, "mass-gated (RFC rule)", 0.012)]
    for arm, colour, label, jitter in series:
        for tag, (name, marker) in MODELS.items():
            xs = [x * 100 + jitter * 100 for x, _, t in pts[arm] if t == tag]
            ys = [y for _, y, t in pts[arm] if t == tag]
            if not xs:
                continue
            ax.scatter(xs, ys, s=62, marker=marker, color=colour, alpha=0.78,
                       edgecolor=SURFACE, linewidth=1.5, zorder=3)
    ax.axhline(0, color=MUTED, linewidth=1.2, zorder=2)
    ax.annotate("no loss", (101, 0.0), xytext=(4, 4), textcoords="offset points",
                color=MUTED, fontsize=8.5, va="bottom")
    ax.set_xlim(-2, 106)
    ax.set_ylim(-0.62, 0.07)
    ax.set_xlabel("Share of the request's blocks freed when the passage is copied back (%)")
    ax.set_ylabel("Copy accuracy minus full-cache accuracy")
    _style(ax, "Freeing blocks costs a passage that went quiet, with or without mass-gating",
           "Share of the request's blocks freed when the passage is copied back (%)",
           "Copy accuracy minus full-cache accuracy")
    ax.title.set_fontsize(11.5)

    # Direct labels in the empty regions, so identity never rests on colour alone.
    ax.annotate("sink + recent window:\nloses the passage whenever\nit sits in the oldest blocks",
                xy=(2, -0.50), xytext=(2, -0.60), color=S2, fontsize=9, fontweight="bold",
                va="bottom")
    ax.annotate("mass-gated:\nharmless in most\nwindows up to\nabout 40% freed,\nbut a few lose\naccuracy",
                xy=(38.5, -0.025), xytext=(1.5, -0.285), color=S1, fontsize=9, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=S1, linewidth=1.1, alpha=0.7))

    handles = [plt.Line2D([], [], marker="s", linestyle="", color=S1, markersize=8, label="mass-gated (RFC rule)"),
               plt.Line2D([], [], marker="s", linestyle="", color=S2, markersize=8, label="sink + recent window"),
               plt.Line2D([], [], marker="o", linestyle="", color=INK_2, markersize=7, label="Qwen2.5-0.5B"),
               plt.Line2D([], [], marker="D", linestyle="", color=INK_2, markersize=7, label="Qwen2.5-1.5B")]
    leg = ax.legend(handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14),
                    ncol=4, fontsize=9)
    for t in leg.get_texts():
        t.set_color(INK_2)
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "block_expiry_dormant_passage.png")
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


if __name__ == "__main__":
    print(chart_dormant_passage() or "no data")
