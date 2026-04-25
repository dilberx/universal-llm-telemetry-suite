"""
Visualization suite for experiment results.

Generates publication-quality charts from experiment JSON data.
Run: python -m src.experiments.visualize
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


# ── Style ────────────────────────────────────────────────────────────
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


def _setup_chart(title: str, xlabel: str, ylabel: str, figsize=(10, 6)):
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


def plot_token_confidence(data_path: str, output_dir: str = "./reports/charts"):
    """Generate token confidence charts."""
    with open(data_path) as f:
        data = json.load(f)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Chart 1: Skip Rate vs Threshold (by profile) ─────────────
    fig, ax = _setup_chart(
        "Token Confidence: Skip Rate vs Threshold",
        "Confidence Threshold",
        "Skip Rate (%)",
    )

    for i, profile in enumerate(["confident", "natural", "uncertain"]):
        trials = [t for t in data["trials"]
                  if t["config"]["profile"] == profile
                  and t["config"]["temperature"] == 0.3
                  and t["config"]["vocab_size"] == 32000]
        trials.sort(key=lambda x: x["config"]["threshold"])

        thresholds = [t["config"]["threshold"] for t in trials]
        skip_rates = [t["metrics"]["skip_rate"] * 100 for t in trials]

        ax.plot(thresholds, skip_rates, "o-", color=PALETTE[i],
                label=f"{profile}", linewidth=2, markersize=6)

    ax.legend(fontsize=11, facecolor="#1a1a2e", edgecolor="#444")
    ax.yaxis.set_major_formatter(ticker.PercentFormatter())
    fig.tight_layout()
    fig.savefig(out / "token_confidence_skip_rate.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ token_confidence_skip_rate.png")

    # ── Chart 2: Skip Rate vs Temperature (at threshold=0.9) ─────
    fig, ax = _setup_chart(
        "Temperature Effect on Token Confidence",
        "Temperature",
        "Skip Rate (%)",
    )

    for i, profile in enumerate(["confident", "natural"]):
        trials = [t for t in data["trials"]
                  if t["config"]["profile"] == profile
                  and t["config"]["threshold"] == 0.9
                  and t["config"]["vocab_size"] == 32000]
        trials.sort(key=lambda x: x["config"]["temperature"])

        temps = [t["config"]["temperature"] for t in trials]
        skip_rates = [t["metrics"]["skip_rate"] * 100 for t in trials]

        ax.plot(temps, skip_rates, "s-", color=PALETTE[i],
                label=f"{profile}", linewidth=2, markersize=8)

    ax.legend(fontsize=11, facecolor="#1a1a2e", edgecolor="#444")
    ax.yaxis.set_major_formatter(ticker.PercentFormatter())
    fig.tight_layout()
    fig.savefig(out / "token_confidence_temperature.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ token_confidence_temperature.png")

    # ── Chart 3: Vocab Size Impact ────────────────────────────────
    fig, ax = _setup_chart(
        "Vocabulary Size Impact on Confidence",
        "Confidence Threshold",
        "Skip Rate (%)",
    )

    for i, (vs, label) in enumerate([(32000, "32K (Llama)"), (128256, "128K (GPT-4)")]):
        trials = [t for t in data["trials"]
                  if t["config"]["profile"] == "confident"
                  and t["config"]["temperature"] == 0.3
                  and t["config"]["vocab_size"] == vs]
        trials.sort(key=lambda x: x["config"]["threshold"])

        thresholds = [t["config"]["threshold"] for t in trials]
        skip_rates = [t["metrics"]["skip_rate"] * 100 for t in trials]

        ax.plot(thresholds, skip_rates, "D-", color=PALETTE[i],
                label=f"vocab={label}", linewidth=2, markersize=6)

    ax.legend(fontsize=11, facecolor="#1a1a2e", edgecolor="#444")
    ax.yaxis.set_major_formatter(ticker.PercentFormatter())
    fig.tight_layout()
    fig.savefig(out / "token_confidence_vocab_size.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ token_confidence_vocab_size.png")


def plot_head_pruning(data_path: str, output_dir: str = "./reports/charts"):
    """Generate attention head pruning charts."""
    with open(data_path) as f:
        data = json.load(f)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Chart: Quality vs Prune Fraction (by model size) ──────────
    fig, ax = _setup_chart(
        "Attention Head Pruning: Quality vs Heads Removed",
        "Fraction of Heads Pruned",
        "Quality Score (cosine similarity)",
    )

    for i, model in enumerate(["small", "medium", "large"]):
        trials = [t for t in data["trials"] if t["config"]["model"] == model]
        trials.sort(key=lambda x: x["config"]["prune_fraction"])

        fractions = [t["config"]["prune_fraction"] for t in trials]
        quality = [t["metrics"]["quality_score"] for t in trials]

        ax.plot(fractions, quality, "o-", color=PALETTE[i],
                label=f'{model} ({trials[0]["config"]["layers"]}L/{trials[0]["config"]["heads"]}H)',
                linewidth=2, markersize=6)

    # Mark the "cliff" region
    ax.axvspan(0, 0.15, alpha=0.1, color=PALETTE[1], label="Critical zone")
    ax.axhline(y=0.46, color="#666", linestyle="--", alpha=0.5, label="Quality plateau")

    ax.legend(fontsize=10, facecolor="#1a1a2e", edgecolor="#444")
    ax.set_xlim(-0.02, 0.82)
    fig.tight_layout()
    fig.savefig(out / "head_pruning_quality.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ head_pruning_quality.png")

    # ── Chart: KL Divergence vs Prune Fraction ────────────────────
    fig, ax = _setup_chart(
        "Information Loss from Head Pruning",
        "Fraction of Heads Pruned",
        "KL Divergence (lower = better)",
    )

    for i, model in enumerate(["small", "medium", "large"]):
        trials = [t for t in data["trials"]
                  if t["config"]["model"] == model
                  and "kl_divergence" in t["metrics"]
                  and t["config"]["prune_fraction"] > 0]
        trials.sort(key=lambda x: x["config"]["prune_fraction"])

        fractions = [t["config"]["prune_fraction"] for t in trials]
        kl = [t["metrics"]["kl_divergence"] for t in trials]

        ax.plot(fractions, kl, "^-", color=PALETTE[i],
                label=f'{model}', linewidth=2, markersize=6)

    ax.legend(fontsize=11, facecolor="#1a1a2e", edgecolor="#444")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(out / "head_pruning_kl_div.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ head_pruning_kl_div.png")


def plot_kv_cache(data_path: str, output_dir: str = "./reports/charts"):
    """Generate KV cache eviction benchmark charts."""
    with open(data_path) as f:
        data = json.load(f)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Chart: Quality vs Memory Saved (by policy) ────────────────
    fig, ax = _setup_chart(
        "KV Cache Eviction: Quality vs Memory Saved",
        "Memory Saved (%)",
        "Quality Retained (attention mass)",
    )

    policy_colors = {
        "full": COLORS["gray"],
        "window": COLORS["secondary"],
        "h2o": COLORS["primary"],
        "snapkv": COLORS["purple"],
        "pyramid": COLORS["quaternary"],
        "adaptive": COLORS["tertiary"],
    }

    for policy_name in ["window", "h2o", "snapkv", "pyramid", "adaptive"]:
        trials = [r for r in data["results"]
                  if r["policy_name"] == policy_name
                  and r["memory_saved_pct"] > 0]

        if not trials:
            continue

        mem_saved = [r["memory_saved_pct"] for r in trials]
        quality = [r["quality_score"] for r in trials]

        ax.scatter(mem_saved, quality,
                   color=policy_colors.get(policy_name, "#fff"),
                   label=policy_name, s=60, alpha=0.8, edgecolors="white", linewidth=0.5)

    ax.legend(fontsize=11, facecolor="#1a1a2e", edgecolor="#444")
    ax.set_xlim(-5, 100)
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(out / "kv_cache_quality_vs_memory.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ kv_cache_quality_vs_memory.png")


def generate_all_charts():
    """Generate all visualization charts."""
    print("Generating experiment visualizations...")
    print()

    # Find latest experiment files
    exp_dir = Path("./reports/experiments")
    charts_dir = "./reports/charts"

    tc_files = sorted(exp_dir.glob("token_confidence_*.json"))
    ah_files = sorted(exp_dir.glob("attention_head_*.json"))
    kv_file = Path("./reports/kv_cache/initial_results.json")

    if tc_files:
        print("Token Confidence:")
        plot_token_confidence(str(tc_files[-1]), charts_dir)

    if ah_files:
        print("\nAttention Head Pruning:")
        plot_head_pruning(str(ah_files[-1]), charts_dir)

    if kv_file.exists():
        print("\nKV Cache Eviction:")
        plot_kv_cache(str(kv_file), charts_dir)

    print(f"\n✅ All charts saved to {charts_dir}/")


if __name__ == "__main__":
    generate_all_charts()
