"""Regenerate the three README charts that were still from the short-prompt runs.

All three now read 1024-token data and overwrite the old files in place, so the
README links stay valid:

  real_extreme_compression.png       per-layer retention at a 10% budget, from the
                                     1024-token profiler pass (0.5B and 7B)
  e2e_kv_logprob_distortion.png      median NLL increase over five 1024-token
                                     windows, 0.5B and 7B side by side
  e2e_kv_cache_greedy_similarity.png greedy similarity on the same five windows (0.5B)

Palette: the three policies the proxy can score (h2o, window, hybrid / sink) take
categorical slots 1-3, the set already validated for the corrected charts. The two
other policies are drawn in muted ink with distinct dash patterns rather than
slots 4-5, because five slots fail the normal-vision and colour-vision floors on
this surface. Every series is direct-labelled, so identity never rests on colour.
"""

from __future__ import annotations

import glob
import json
import os
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.experiments.visualize_corrected import INK_2, MUTED, S1, S2, S3, SURFACE, _style

OUT_DIR = "reports/charts"
EXP = "reports/experiments"
BUDGETS_PCT = [10, 20, 30, 50, 75, 90]
E2E_ORDER = ("h2o", "window", "adaptive", "hybrid", "snapkv")

# policy -> (colour, label, line style); sink is the proxy curve hybrid is scored on
STYLE = {
    "h2o": (S1, "h2o", "-"),
    "window": (S2, "window", "-"),
    "hybrid": (S3, "hybrid (sink + recent)", "-"),
    "sink": (S3, "sink + recent", "-"),
    "snapkv": (MUTED, "snapkv", "--"),
    "adaptive": (MUTED, "adaptive", ":"),
}


def _place_labels(ax, items, min_gap_px=13):
    """Direct labels at line ends, pushed apart vertically so none collide.

    items: [(x, y, text, colour)]. Spacing is resolved in display space after
    layout, so the gap holds whatever the axis range is.
    """
    ax.figure.canvas.draw()
    to_px = ax.transData
    placed = sorted((to_px.transform((x, y))[1], x, y, t, c) for x, y, t, c in items)
    ys = [p[0] for p in placed]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + min_gap_px)
    dpi_scale = 72 / ax.figure.dpi            # offset points from pixels
    for (py, x, y, text, colour), ny in zip(placed, ys):
        ax.annotate(text, (x, y), xytext=(7, (ny - py) * dpi_scale), textcoords="offset points",
                    color=colour if colour != MUTED else INK_2, fontsize=8.5,
                    va="center", fontweight="bold")


def _cells(trials, metric):
    out = {}
    for t in trials:
        c = t["config"]
        if c.get("policy") in (None, "full") or metric not in t["metrics"]:
            continue
        out.setdefault((c["policy"], c["budget_frac"]), []).append(t["metrics"][metric])
    return {k: median(v) for k, v in out.items()}


def _long5(prefix):
    """The 5-window 1024-token run of the 0.5B for a result prefix."""
    for f in sorted(glob.glob(f"{EXP}/{prefix}_qwen2.5_0.5b_2026091*.json"), reverse=True):
        d = json.load(open(f))
        if median(t["metrics"].get("prompt_tokens", 0) for t in d["trials"]) >= 1000:
            return d["trials"]
    return None


def _trials_7b():
    trials = []
    for w in range(5):
        fs = sorted(glob.glob(f"{EXP}/e2e_kv_logprob_qwen2.5_7b_long1024_w{w}_*.json"))
        if fs:
            trials += json.load(open(fs[-1]))["trials"]
    return trials or None


def _policy_lines(ax, cells, order):
    """Draw one line per policy; return the end points for direct labels."""
    labels = []
    for pol in order:
        pts = sorted((b, v) for (p, b), v in cells.items() if p == pol)
        if not pts:
            continue
        colour, label, ls = STYLE[pol]
        xs = [b * 100 for b, _ in pts]
        ys = [v for _, v in pts]
        ax.plot(xs, ys, ls, color=colour, linewidth=2, zorder=3)
        ax.scatter(xs, ys, s=36, color=colour, edgecolor=SURFACE, linewidth=1.5, zorder=4)
        labels.append((xs[-1], ys[-1], label, colour))
    return labels


def _legend(fig, policies):
    """Legend below the plot; the direct labels carry identity, this is the key."""
    handles = [plt.Line2D([], [], color=STYLE[p][0], linestyle=STYLE[p][2], linewidth=2,
                          marker="o", markersize=5, label=STYLE[p][1]) for p in policies]
    leg = fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(INK_2)


def _budget_axis(ax):
    ax.set_xlim(5, 124)
    ax.set_xticks(BUDGETS_PCT)


def _save(fig, name, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def chart_logprob(out_dir=OUT_DIR):
    panels = [("Qwen2.5-0.5B", _long5("e2e_kv_logprob")), ("Qwen2.5-7B", _trials_7b())]
    panels = [(n, t) for n, t in panels if t]
    if not panels:
        return None
    fig, axes = plt.subplots(1, len(panels), figsize=(6.4 * len(panels), 5.0), sharey=True)
    axes = [axes] if len(panels) == 1 else list(axes)
    pending = []
    for ax, (name, trials) in zip(axes, panels):
        pending.append(_policy_lines(ax, _cells(trials, "nll_delta_vs_full"), E2E_ORDER))
        ax.axhline(0, color=MUTED, linewidth=1.0, zorder=2)
        _style(ax, name, "KV cache budget (% of context kept)", "NLL increase over full cache")
        ax.title.set_fontsize(11.5)
        _budget_axis(ax)
    _legend(fig, E2E_ORDER)
    fig.suptitle("Fixed-target NLL at 1024 tokens, median of five wikitext windows (lower is better)",
                 color=INK_2, fontsize=10, x=0.01, ha="left", y=0.995)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    for ax, labels in zip(axes, pending):
        _place_labels(ax, labels)
    return _save(fig, "e2e_kv_logprob_distortion.png", out_dir)


def chart_greedy(out_dir=OUT_DIR):
    trials = _long5("e2e_kv_cache")
    if not trials:
        return None
    cells = {k: v * 100 for k, v in _cells(trials, "text_similarity").items()}
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    labels = _policy_lines(ax, cells, E2E_ORDER)
    _style(ax, "Greedy output similarity to the full cache, Qwen2.5-0.5B at 1024 tokens",
           "KV cache budget (% of context kept)", "Similarity to full-cache output (%), median of 5 windows")
    ax.title.set_fontsize(11.5)
    _budget_axis(ax)
    ax.set_ylim(-3, 105)
    _legend(fig, E2E_ORDER)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    _place_labels(ax, labels)
    return _save(fig, "e2e_kv_cache_greedy_similarity.png", out_dir)


def chart_extreme(out_dir=OUT_DIR):
    panels = []
    for name, pat in (("Qwen2.5-0.5B", "profile_qwen2_5-0_5b_*.json"),
                      ("Qwen2.5-7B", "profile_qwen2_5-7b_*.json")):
        fs = sorted(glob.glob(f"reports/profiles/{pat}"))
        if fs:
            panels.append((name, json.load(open(fs[-1]))))
    if not panels:
        return None
    fig, axes = plt.subplots(1, len(panels), figsize=(6.4 * len(panels), 4.8), sharey=True)
    axes = [axes] if len(panels) == 1 else list(axes)
    pending = []
    for ax, (name, prof) in zip(axes, panels):
        labels = []
        for pol in ("h2o", "sink", "window"):
            colour, label, ls = STYLE[pol]
            ys = [L["retention_at"][pol]["0.1"] for L in prof["layers"]]
            xs = list(range(len(ys)))
            ax.plot(xs, ys, ls, color=colour, linewidth=2, zorder=3)
            ax.scatter(xs, ys, s=22, color=colour, edgecolor=SURFACE, linewidth=1.2, zorder=4)
            labels.append((xs[-1], ys[-1], f"{label}  {median(ys):.0%}", colour))
        pending.append(labels)
        _style(ax, name, "Layer", "Attention mass retained at a 10% budget")
        ax.title.set_fontsize(11.5)
        ax.set_xlim(-1, len(prof["layers"]) + 9)
        ax.set_ylim(-0.03, 1.0)
    _legend(fig, ("h2o", "sink", "window"))
    fig.suptitle("90% eviction at 1024 tokens: retained attention mass per layer (label = median over layers)",
                 color=INK_2, fontsize=10, x=0.01, ha="left", y=0.995)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    for ax, labels in zip(axes, pending):
        _place_labels(ax, labels)
    return _save(fig, "real_extreme_compression.png", out_dir)


if __name__ == "__main__":
    for fn in (chart_extreme, chart_logprob, chart_greedy):
        print(fn() or f"{fn.__name__}: no data")
