"""
Real Model Findings Visualizer

Generates charts from Qwen2.5-0.5B real-model experiments,
highlighting findings that differ from synthetic experiments.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


plt.style.use("dark_background")
COLORS = {
    "primary": "#00D4FF",
    "secondary": "#FF6B6B",
    "tertiary": "#50FA7B",
    "quaternary": "#FFB86C",
    "purple": "#BD93F9",
    "pink": "#FF79C6",
    "gray": "#6272A4",
}
PALETTE = list(COLORS.values())


def _setup_chart(title, xlabel, ylabel, figsize=(10, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title, fontsize=16, fontweight="bold", pad=15, color="white")
    ax.set_xlabel(xlabel, fontsize=12, color="#ccc")
    ax.set_ylabel(ylabel, fontsize=12, color="#ccc")
    ax.tick_params(colors="#999")
    ax.grid(True, alpha=0.15, color="#555")
    for spine in ax.spines.values():
        spine.set_color("#444")
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")
    return fig, ax


def generate_real_model_charts(data_path: str, output_dir: str = "./reports/charts"):
    """Generate all charts from real model data."""
    with open(data_path) as f:
        data = json.load(f)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ═══════════════════════════════════════════════════════════════
    # CHART 1: H2O vs Window advantage per layer (the novel finding)
    # ═══════════════════════════════════════════════════════════════
    fig, ax = _setup_chart(
        "H2O vs StreamingLLM: Per-Layer Advantage (Qwen2.5-0.5B)",
        "Transformer Layer",
        "Quality Ratio (H2O ÷ Window)",
        figsize=(12, 6),
    )

    kv_trials = [t for t in data["trials"] if t["config"]["experiment"] == "kv_cache_real"]

    for i, budget in enumerate([0.2, 0.3, 0.5]):
        subset = [t for t in kv_trials if t["config"]["budget_frac"] == budget]
        layer_ratios = {}
        for t in subset:
            layer = t["config"]["layer"]
            if layer not in layer_ratios:
                layer_ratios[layer] = []
            layer_ratios[layer].append(t["metrics"]["h2o_vs_window"])

        layers = sorted(layer_ratios.keys())
        ratios = [sum(layer_ratios[l]) / len(layer_ratios[l]) for l in layers]

        ax.plot(layers, ratios, "o-", color=PALETTE[i],
                label=f"{budget:.0%} budget", linewidth=2, markersize=5)

    ax.axhline(y=1.0, color="#666", linestyle="--", alpha=0.5, label="Break-even")
    ax.set_yscale("log")
    ax.legend(fontsize=10, facecolor="#1a1a2e", edgecolor="#444")

    # Annotate peak layers
    ax.annotate("L11: 16×", xy=(11, 16), fontsize=9, color=COLORS["primary"],
                ha="center", va="bottom")
    ax.annotate("L16: 23×", xy=(16, 23), fontsize=9, color=COLORS["primary"],
                ha="center", va="bottom")

    fig.tight_layout()
    fig.savefig(out / "real_h2o_per_layer.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ real_h2o_per_layer.png")

    # ═══════════════════════════════════════════════════════════════
    # CHART 2: Real token confidence by prompt type
    # ═══════════════════════════════════════════════════════════════
    fig, ax = _setup_chart(
        "Real Token Confidence: Skip Rate by Task Type (Qwen2.5-0.5B)",
        "Task Type",
        "Skip Rate at 90% Threshold",
    )

    conf_trials = [t for t in data["trials"]
                   if t["config"]["experiment"] == "token_confidence_real"]
    conf_trials.sort(key=lambda t: t["metrics"]["skip_rate_90"], reverse=True)

    prompts = [t["config"]["prompt"] for t in conf_trials]
    skip_90 = [t["metrics"]["skip_rate_90"] * 100 for t in conf_trials]
    skip_95 = [t["metrics"]["skip_rate_95"] * 100 for t in conf_trials]

    x = np.arange(len(prompts))
    width = 0.35
    bars1 = ax.bar(x - width/2, skip_90, width, color=COLORS["primary"],
                   label="Skip @ 90%", alpha=0.9)
    bars2 = ax.bar(x + width/2, skip_95, width, color=COLORS["secondary"],
                   label="Skip @ 95%", alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(prompts, rotation=45, ha="right", fontsize=9, color="#ccc")
    ax.yaxis.set_major_formatter(ticker.PercentFormatter())
    ax.legend(fontsize=10, facecolor="#1a1a2e", edgecolor="#444")

    fig.tight_layout()
    fig.savefig(out / "real_token_confidence_by_task.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ real_token_confidence_by_task.png")

    # ═══════════════════════════════════════════════════════════════
    # CHART 3: Attention sink mass pattern
    # ═══════════════════════════════════════════════════════════════
    fig, ax = _setup_chart(
        "Attention Sink Mass vs Sparsity (Qwen2.5-0.5B)",
        "Attention Sparsity (fraction of positions with >1% weight)",
        "Sink Mass (attention on first 4 tokens)",
    )

    head_trials = [t for t in data["trials"]
                   if t["config"]["experiment"] == "head_analysis"]

    for i, t in enumerate(head_trials):
        m = t["metrics"]
        ax.scatter(m["sparsity_mean"], m["sink_mass_mean"],
                   s=150, color=PALETTE[i % len(PALETTE)],
                   edgecolors="white", linewidth=1, zorder=5)
        ax.annotate(t["config"]["prompt"], (m["sparsity_mean"], m["sink_mass_mean"]),
                    fontsize=8, color="#ccc", ha="left", va="bottom",
                    xytext=(5, 5), textcoords="offset points")

    ax.set_xlim(0, 0.5)
    ax.set_ylim(0.35, 1.0)
    fig.tight_layout()
    fig.savefig(out / "real_sink_vs_sparsity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ real_sink_vs_sparsity.png")

    # ═══════════════════════════════════════════════════════════════
    # CHART 4: H2O quality at 10% budget (extreme compression)
    # ═══════════════════════════════════════════════════════════════
    fig, ax = _setup_chart(
        "Extreme KV Cache Compression: 90% Eviction (Qwen2.5-0.5B)",
        "Transformer Layer",
        "Quality Retained",
    )

    for i, (policy, key) in enumerate([
        ("H2O", "h2o_quality"),
        ("Sink+Recent", "sink_quality"),
        ("Window", "window_quality"),
    ]):
        subset = [t for t in kv_trials if t["config"]["budget_frac"] == 0.1]
        layer_q = {}
        for t in subset:
            layer = t["config"]["layer"]
            if layer not in layer_q:
                layer_q[layer] = []
            layer_q[layer].append(t["metrics"][key])

        layers = sorted(layer_q.keys())
        quality = [sum(layer_q[l]) / len(layer_q[l]) for l in layers]

        ax.plot(layers, quality, "o-", color=PALETTE[i],
                label=policy, linewidth=2, markersize=5)

    ax.legend(fontsize=10, facecolor="#1a1a2e", edgecolor="#444")
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(out / "real_extreme_compression.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ real_extreme_compression.png")

    print(f"\n✅ Real model charts saved to {output_dir}/")


if __name__ == "__main__":
    import glob
    files = sorted(glob.glob("reports/experiments/real_model_analysis_*.json"))
    if files:
        generate_real_model_charts(files[-1])
    else:
        print("No real model data found. Run exp_real_model.py first.")
