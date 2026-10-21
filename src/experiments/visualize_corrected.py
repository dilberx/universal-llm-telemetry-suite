"""Charts for the corrected 2026-09 findings.

The charts in `reports/charts/` predate the length, architecture and proxy-audit
corrections, so several of them contradict the text they sit beside. These
three replace the ones that were wrong and add the two findings that had no
figure at all.

Palette and mark rules follow the project's data-viz method: categorical slots
assigned in fixed order (blue, orange, aqua), validated for colour-vision
separation on all pairs, every series direct-labelled because the aqua slot
sits below 3:1 contrast on the light surface.
"""

from __future__ import annotations

import glob
import json
import os
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "reports/charts"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8a84"
GRID = "#e4e4df"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"   # categorical slots 1-3
S1_LIGHT = "#a8c8ee"


def _style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.figure.patch.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=13, pad=14, loc="left", fontweight="bold")
    ax.set_xlabel(xlabel, color=INK_2, fontsize=10)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=10)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)


def _latest(pattern):
    hits = sorted(glob.glob(pattern))
    return json.load(open(hits[-1])) if hits else None


# ── 1. Short prompts inflate the ratio ────────────────────────────────

def chart_length_inflation(out_dir=OUT_DIR):
    """Before/after per model: the same measurement at 43-55 tokens and 1024."""
    d = _latest("reports/experiments/scale_validation_20260917*.json")
    if d is None:
        return None
    rows = [(t["config"], t["metrics"]) for t in d["trials"]]
    models = [("Qwen/Qwen2.5-0.5B", "Qwen2.5-0.5B\n0.5B  GQA"),
              ("microsoft/phi-2", "Phi-2\n2.7B  MHA"),
              ("mistralai/Mistral-7B-Instruct-v0.2", "Mistral-7B\n7.2B  GQA"),
              ("/home/dilbersha/models/llava-1.5-7b-hf", "Vicuna-7B\n6.7B  MHA")]
    vals = {}
    for c, m in rows:
        if c.get("budget_frac") == 0.1 and c.get("prompts") in ("short", "wikitext_1024"):
            vals.setdefault(c["model"], {})[c["prompts"]] = m["h2o_over_window_median"]

    fig, ax = plt.subplots(figsize=(9, 4.0))
    ys = range(len(models))
    for y, (key, label) in zip(ys, models):
        a, b = vals[key]["short"], vals[key]["wikitext_1024"]
        ax.plot([b, a], [y, y], color=S1_LIGHT, linewidth=2, zorder=1,
                solid_capstyle="round")
        ax.scatter([a], [y], s=70, color=S1_LIGHT, zorder=2,
                   edgecolor=SURFACE, linewidth=2)
        ax.scatter([b], [y], s=90, color=S1, zorder=3,
                   edgecolor=SURFACE, linewidth=2)
        ax.annotate(f"{a:.0f}x", (a, y), xytext=(6, 7), textcoords="offset points",
                    color=MUTED, fontsize=9)
        ax.annotate(f"{b:.0f}x", (b, y), xytext=(-6, 7), textcoords="offset points",
                    color=INK, fontsize=9, fontweight="bold", ha="right")
    ax.set_yticks(list(ys))
    ax.set_yticklabels([m[1] for m in models])
    ax.set_xscale("log")
    ax.set_xticks([10, 20, 50, 100, 200])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.invert_yaxis()
    _style(ax, "Short prompts overstate how badly window eviction fails",
           "Median H2O / window attention-mass retention at a 10% budget (log scale)", "")
    ax.set_xlim(right=340)   # room for the 207x label at the right edge
    ax.scatter([], [], s=70, color=S1_LIGHT, label="43-55 token prompts")
    ax.scatter([], [], s=90, color=S1, label="1024 tokens of wikitext")
    # Below the plot: every in-axes region is occupied by a row or its labels.
    leg = ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18),
                    ncol=2, fontsize=9)
    for t in leg.get_texts():
        t.set_color(INK_2)
    fig.tight_layout()
    path = os.path.join(out_dir, "corrected_length_inflation.png")
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


# ── 2. What actually moves the measurement ────────────────────────────

def chart_effect_ranking(out_dir=OUT_DIR):
    """Three effects, each measured with the other two held as fixed as possible."""
    effects = [
        ("Context length", 2.69, "43-token prompts vs 1024 tokens, Qwen2.5-0.5B"),
        ("Attention architecture", 1.72, "Mistral-7B GQA vs Vicuna-7B MHA, matched scale"),
        ("Model scale", 1.22, "0.5B vs 7.6B within Qwen2.5, same tokenizer"),
    ]
    fig, ax = plt.subplots(figsize=(10.5, 3.4))
    ys = range(len(effects))
    for y, (name, val, note) in zip(ys, effects):
        colour = S1 if y == 0 else MUTED
        ax.barh(y, val - 1, left=1, height=0.30, color=colour,
                edgecolor=SURFACE, linewidth=2, zorder=2)
        # Value and provenance on one line to the right of the bar, so neither
        # can collide with the bar or with the row below it.
        ax.annotate(f"{val:.2f}x", (val, y), xytext=(9, 0), textcoords="offset points",
                    va="center", color=INK, fontsize=11, fontweight="bold")
        ax.annotate(note, (val, y), xytext=(52, 0), textcoords="offset points",
                    va="center", color=MUTED, fontsize=8.5)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([e[0] for e in effects], fontsize=10)
    ax.set_xlim(1, 4.4)
    ax.set_xticks([1.0, 1.5, 2.0, 2.5, 3.0])
    ax.invert_yaxis()
    _style(ax, "Context length moves this more than architecture or scale",
           "Factor change in the H2O / window ratio at a 10% budget", "")
    fig.tight_layout()
    path = os.path.join(out_dir, "corrected_effect_ranking.png")
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


# ── 3. The proxy inverts the policy ranking ───────────────────────────

def chart_proxy_inversion(out_dir=OUT_DIR):
    """Retention against NLL. Within a policy the two track; across policies they do not."""
    prof = _latest("reports/profiles/profile_qwen2_5-0_5b_*.json")
    logp = _latest("reports/experiments/e2e_kv_logprob_qwen2.5_0.5b_20260916_2131*.json")
    if prof is None or logp is None:
        return None
    buds = [0.1, 0.2, 0.3, 0.5, 0.75, 0.9]
    nll = {}
    for t in logp["trials"]:
        c = t["config"]
        if c.get("policy") in (None, "full") or c.get("budget_frac") in (None, 1.0):
            continue
        if "nll" in t["metrics"]:
            nll.setdefault((c["policy"], c["budget_frac"]), []).append(t["metrics"]["nll"])
    nll = {k: median(v) for k, v in nll.items()}

    series = [("h2o", "h2o", S1, "H2O"),
              ("sink", "hybrid", S3, "sink + recent\n(hybrid)"),
              ("window", "window", S2, "window")]
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    for proxy_key, pol, colour, label in series:
        xs = [median([l["retention_at"][proxy_key][str(b)] for l in prof["layers"]])
              for b in buds]
        ys = [nll[(pol, b)] for b in buds]
        ax.plot(xs, ys, color=colour, linewidth=2, alpha=0.55, zorder=2,
                solid_capstyle="round")
        ax.scatter(xs, ys, s=64, color=colour, zorder=3,
                   edgecolor=SURFACE, linewidth=2, label=label)
        ax.annotate(label.replace("\n", " "), (xs[-1], ys[-1]), xytext=(10, 0),
                    textcoords="offset points", color=INK, fontsize=9,
                    fontweight="bold", va="center")
    _style(ax, "The proxy predicts quality within a policy, and inverts across policies",
           "Attention-mass retention  (the proxy, higher is better)",
           "NLL under the pruned cache  (quality, lower is better)")
    ax.set_xlim(0, 1.32)   # headroom so the direct labels do not overflow
    # Placed in the empty lower-left quadrant, pointing at the H2O/hybrid gap
    # at high retention, so the leader does not cross either line.
    ax.annotate("at the same retention, H2O's\ndistributions are far worse\nthan the sink policy's",
                xy=(0.92, 2.2), xytext=(0.13, 2.55), color=INK_2, fontsize=9,
                arrowprops=dict(arrowstyle="->", color=MUTED, linewidth=1.2,
                                connectionstyle="arc3,rad=-0.15"))
    leg = ax.legend(frameon=False, loc="center left", bbox_to_anchor=(0.005, 0.60),
                    fontsize=9, title="policy")
    leg.get_title().set_color(INK_2)
    leg.get_title().set_fontsize(9)
    for t in leg.get_texts():
        t.set_color(INK_2)
    fig.tight_layout()
    path = os.path.join(out_dir, "corrected_proxy_inversion.png")
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def generate_corrected_charts(out_dir: str = OUT_DIR) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    made = []
    for fn in (chart_length_inflation, chart_effect_ranking, chart_proxy_inversion):
        path = fn(out_dir)
        if path:
            made.append(path)
            print(f"  wrote {path}")
        else:
            print(f"  skipped {fn.__name__}: source data missing")
    return made


if __name__ == "__main__":
    generate_corrected_charts()
