"""
E2E KV Cache Visualization Script.

Plots text similarity and logprob perplexity delta across different policies
(window, h2o, hybrid, snapkv, pyramid, adaptive).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


plt.style.use("dark_background")
COLORS = {
    "window": "#E06262",
    "h2o": "#E86A0C",
    "hybrid": "#5A9BE7",
    "snapkv": "#BD93F9",
    "pyramid": "#FFB86C",
    "adaptive": "#48C774",
    "full": "#FFFFFF",
}


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


def plot_e2e_greedy(data_path: str, output_dir: str = "./reports/charts"):
    """Plot text similarity vs budget fraction for greedy generation."""
    with open(data_path) as f:
        data = json.load(f)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    trials = data.get("trials", [])
    if not trials:
        print("  ! No trials found in greedy E2E file.")
        return None

    # Group by policy and budget_frac
    policies = sorted({t["config"].get("policy") for t in trials if t["config"].get("policy") != "full"})
    budgets = sorted({t["config"].get("budget_frac") for t in trials})

    fig, ax = _setup_chart(
        "End-to-End Generation Quality under Cache Eviction",
        "KV Cache Budget Fraction",
        "Text Similarity to Full Cache (%)",
    )

    for policy in policies:
        policy_trials = [t for t in trials if t["config"].get("policy") == policy]
        policy_budgets = []
        policy_similarities = []

        for b in budgets:
            subset = [t for t in policy_trials if t["config"].get("budget_frac") == b]
            if subset:
                sims = [t["metrics"].get("text_similarity", 0.0) * 100 for t in subset]
                policy_budgets.append(b)
                policy_similarities.append(np.mean(sims))

        if policy_budgets:
            ax.plot(
                policy_budgets,
                policy_similarities,
                "o-",
                color=COLORS.get(policy, "#50FA7B"),
                label=policy,
                linewidth=2.5,
                markersize=6,
            )

    ax.legend(fontsize=11, facecolor="#1a1a2e", edgecolor="#444")
    ax.set_ylim(-5, 105)
    fig.tight_layout()
    path = out / "e2e_kv_cache_greedy_similarity.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ e2e_kv_cache_greedy_similarity.png")
    return path


def plot_e2e_logprob(data_path: str, output_dir: str = "./reports/charts"):
    """Plot perplexity and NLL delta vs budget fraction."""
    with open(data_path) as f:
        data = json.load(f)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    trials = data.get("trials", [])
    if not trials:
        print("  ! No trials found in logprob E2E file.")
        return None

    policies = sorted({t["config"].get("policy") for t in trials if t["config"].get("policy") != "full"})
    budgets = sorted({t["config"].get("budget_frac") for t in trials if t["config"].get("budget_frac") < 1.0})

    fig, ax = _setup_chart(
        "Distribution Distortion under Cache Eviction",
        "KV Cache Budget Fraction",
        "Negative Log-Likelihood Delta",
    )

    for policy in policies:
        policy_trials = [t for t in trials if t["config"].get("policy") == policy]
        policy_budgets = []
        policy_deltas = []

        for b in budgets:
            subset = [t for t in policy_trials if t["config"].get("budget_frac") == b]
            if subset:
                deltas = [t["metrics"].get("nll_delta_vs_full", 0.0) for t in subset]
                policy_budgets.append(b)
                policy_deltas.append(np.mean(deltas))

        if policy_budgets:
            ax.plot(
                policy_budgets,
                policy_deltas,
                "o-",
                color=COLORS.get(policy, "#50FA7B"),
                label=policy,
                linewidth=2.5,
                markersize=6,
            )

    ax.legend(fontsize=11, facecolor="#1a1a2e", edgecolor="#444")
    fig.tight_layout()
    path = out / "e2e_kv_logprob_distortion.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ e2e_kv_logprob_distortion.png")
    return path


def generate_e2e_charts(output_dir: str = "./reports/charts"):
    """Generate all E2E charts from the latest experiment reports."""
    exp_dir = Path("./reports/experiments")
    if not exp_dir.exists():
        print("No experiments directory found.")
        return

    greedy_files = sorted(exp_dir.glob("e2e_kv_cache_*.json"))
    logprob_files = sorted(exp_dir.glob("e2e_kv_logprob_*.json"))

    if greedy_files:
        print(f"Plotting greedy E2E using {greedy_files[-1].name}...")
        plot_e2e_greedy(str(greedy_files[-1]), output_dir)
    else:
        print("No greedy E2E result file found.")

    if logprob_files:
        print(f"Plotting logprob E2E using {logprob_files[-1].name}...")
        plot_e2e_logprob(str(logprob_files[-1]), output_dir)
    else:
        print("No logprob E2E result file found.")


if __name__ == "__main__":
    generate_e2e_charts()
