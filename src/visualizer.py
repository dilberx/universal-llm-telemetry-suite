"""
visualizer.py — LLM Inference Telemetry Suite — Cross-Platform Dashboard

Generates an 8-panel dashboard from benchmark CSVs. Supports:
  - Single-GPU mode: point at one results subdirectory
  - Aggregate mode  : --aggregate flag merges all results/*/production_benchmarks.csv
                      for cross-hardware comparison (multi-GPU / Apple Silicon / etc.)

Usage:
  # Single hardware
  python src/visualizer.py --results-dir results/rtx_3080

  # Cross-platform aggregate (Global Leaderboard mode)
  python src/visualizer.py --results-dir results --aggregate
"""

import argparse
import os
import re
import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

def extract_quantization(model_name: str) -> str:
    m = re.search(r"(q\d[a-z0-9_]*)\.gguf", model_name, re.IGNORECASE)
    return m.group(1).upper() if m else "Unknown"


def extract_model_size(model_name: str) -> float:
    m = re.search(r"([\d.]+)(b)", model_name, re.IGNORECASE)
    return float(m.group(1)) if m else 7.0


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

_REQUIRED_COLS = {
    "model", "family", "context_length", "run_number",
    "tokens_per_sec", "max_vram_mb", "avg_power_watts",
    "tokens_per_joule", "avg_temp_c", "avg_clock_mhz", "perplexity",
}

_NEW_COLS_DEFAULTS = {
    "gpu_name":        "Unknown GPU",
    "total_vram_gb":   0.0,
    "base_clock_mhz":  0.0,
    "memory_type":     "discrete",
    "isolation_level": "gpu_isolated",
    "cold_start_sec":  0.0,
}


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Backwards-compat: fill new columns if absent
    for col, default in _NEW_COLS_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default
    return df


def load_aggregate(results_root: str) -> pd.DataFrame:
    """
    Glob all production_benchmarks.csv files under results_root/*/
    and concatenate them. Derives gpu_slug from the parent directory name
    when gpu_name column is absent/generic.
    """
    pattern = os.path.join(results_root, "*", "production_benchmarks.csv")
    paths   = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No CSVs found matching: {pattern}")

    frames = []
    for path in paths:
        df = load_csv(path)
        # If gpu_name not populated, derive from directory name
        parent = os.path.basename(os.path.dirname(path))
        df["gpu_name"] = df["gpu_name"].replace("Unknown GPU", parent).fillna(parent)
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Subplot helpers
# ---------------------------------------------------------------------------

def _add_ci_bars(ax, data, x, y, hue):
    """Wrapper so we can switch seaborn errorbar kwarg once."""
    return sns.barplot(
        data=data, x=x, y=y, hue=hue,
        errorbar=("ci", 95), capsize=0.1, ax=ax,
    )


def _rotate_xlabels(ax, rotation=40):
    ax.tick_params(axis="x", rotation=rotation)
    ax.set_xticklabels(ax.get_xticklabels(), ha="right", fontsize=9)


# ---------------------------------------------------------------------------
# Main dashboard
# ---------------------------------------------------------------------------

def create_dashboard(results_dir: str, aggregate: bool = False) -> None:

    # --- Load data -------------------------------------------------------
    if aggregate:
        df = load_aggregate(results_dir)
    else:
        csv_file = os.path.join(results_dir, "production_benchmarks.csv")
        if not os.path.exists(csv_file):
            print(f"❌ CSV not found: {csv_file}")
            return
        df = load_csv(csv_file)

    thermal_csv = os.path.join(results_dir, "thermal_log.csv")

    # --- Derived columns -------------------------------------------------
    df["quantization"]  = df["model"].apply(extract_quantization)
    df["model_size_b"]  = df["model"].apply(extract_model_size)

    # 95% CI for TPS, merged back onto full df
    ci_df = (
        df.groupby(["model", "context_length"])
        .agg(std_tps=("tokens_per_sec", "std"), count=("tokens_per_sec", "count"))
        .reset_index()
    )
    ci_df["tps_ci"] = 1.96 * (ci_df["std_tps"] / np.sqrt(ci_df["count"]))
    df = pd.merge(df, ci_df[["model", "context_length", "tps_ci"]], on=["model", "context_length"], how="left")

    # Standard 2048-context slice used by most charts
    df_2048 = df[df["context_length"] == 2048].copy()

    # Hue column: use gpu_name in aggregate mode, else family
    hue_col = "gpu_name" if aggregate else "family"

    # --- Aesthetics ------------------------------------------------------
    sns.set_theme(
        style="whitegrid",
        rc={
            "font.size": 11,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.titlesize": 13,
        },
    )
    sns.set_palette("magma")

    # 4 rows × 2 cols
    fig, axes = plt.subplots(4, 2, figsize=(26, 32))
    fig.suptitle(
        "LLM Inference Telemetry Suite — Performance Dashboard",
        fontsize=16, fontweight="bold", y=0.995,
    )

    # ─────────────────────────────────────────────────────────────────────
    # Row 0 — Throughput | VRAM / Unified Memory usage
    # ─────────────────────────────────────────────────────────────────────
    ax = axes[0, 0]
    _add_ci_bars(ax, df_2048, x="model", y="tokens_per_sec", hue=hue_col)
    ax.set_title("Throughput with 95% CI (Tokens/Sec) [ctx=2048]")
    ax.set_ylabel("Tokens / Second")
    ax.set_xlabel("")
    _rotate_xlabels(ax)

    ax = axes[0, 1]
    mem_label = "Memory Usage (MB)" + (" [Unified + Discrete]" if aggregate else " [VRAM MB]")
    _add_ci_bars(ax, df_2048, x="model", y="max_vram_mb", hue=hue_col)
    ax.set_title(f"Peak {mem_label} with 95% CI [ctx=2048]")
    ax.set_ylabel("Memory (MB)")
    ax.set_xlabel("")
    _rotate_xlabels(ax)

    # ─────────────────────────────────────────────────────────────────────
    # Row 1 — Energy Efficiency | VRAM vs. Context Window
    # ─────────────────────────────────────────────────────────────────────
    ax = axes[1, 0]
    if "tokens_per_joule" in df_2048.columns:
        _add_ci_bars(ax, df_2048, x="model", y="tokens_per_joule", hue=hue_col)
        ax.set_title("Energy Efficiency (Tokens / Joule) [ctx=2048]")
        ax.set_ylabel("Tokens / Joule")
        ax.set_xlabel("")
        _rotate_xlabels(ax)
    else:
        ax.axis("off")

    ax = axes[1, 1]
    context_df = df[df["context_length"].isin([512, 2048, 8192])]
    if not context_df.empty:
        sns.lineplot(
            data=context_df, x="context_length", y="max_vram_mb",
            hue="model", marker="o", errorbar=("ci", 95), ax=ax,
        )
        ax.set_title("Memory Footprint vs. Context Window (95% CI)")
        ax.set_xticks([512, 2048, 8192])
        ax.set_xlabel("Context Length (Tokens)")
        ax.set_ylabel("Peak Memory (MB)")
    else:
        ax.axis("off")

    # ─────────────────────────────────────────────────────────────────────
    # Row 2 — Efficiency Frontier | Unified Memory Efficiency (NEW)
    # ─────────────────────────────────────────────────────────────────────
    ax = axes[2, 0]
    frontier_src = df_2048[df_2048["perplexity"] > 0.0]
    if not frontier_src.empty:
        frontier_df = (
            frontier_src
            .groupby(["model", "family", "quantization", "model_size_b", "gpu_name", "memory_type"])
            .mean(numeric_only=True)
            .reset_index()
        )
        sns.scatterplot(
            data=frontier_df,
            x="perplexity", y="tokens_per_joule",
            hue="family", size="model_size_b",
            sizes=(100, 600), style="memory_type", ax=ax,
        )
        for _, row in frontier_df.iterrows():
            ax.text(
                row["perplexity"] + 0.08, row["tokens_per_joule"],
                f"{row['model_size_b']:.1f}B {row['quantization']}",
                fontsize=8, color="#333333",
            )
        ax.set_title("Efficiency Frontier: Perplexity vs. Energy")
        ax.set_xlabel("WikiText-2 Perplexity (↓ better)")
        ax.set_ylabel("Energy Efficiency (T/J)")
    else:
        ax.axis("off")

    # ── Unified Memory Efficiency chart (NEW) ─────────────────────────
    ax = axes[2, 1]
    if "memory_type" in df_2048.columns and df_2048["memory_type"].nunique() > 0:
        mem_eff = (
            df_2048
            .groupby(["gpu_name", "model", "family", "memory_type"])
            .agg(tj=("tokens_per_joule", "mean"), tps=("tokens_per_sec", "mean"))
            .reset_index()
        )
        # Strip to the per-GPU peak T/J for clarity
        gpu_peak = (
            mem_eff.sort_values("tj", ascending=False)
            .groupby(["gpu_name", "memory_type"]).head(6)
        )
        if not gpu_peak.empty:
            _add_ci_bars(ax, df_2048, x="gpu_name", y="tokens_per_joule", hue="memory_type")
            ax.set_title("Unified vs. Discrete Memory: T/J by Hardware")
            ax.set_xlabel("GPU / SoC")
            ax.set_ylabel("Tokens / Joule (avg across models)")
            _rotate_xlabels(ax)
            # Annotate the memory_type distinction
            ax.axhline(y=df_2048["tokens_per_joule"].mean(), color="grey",
                       linestyle="--", alpha=0.5, label="Dataset mean")
            ax.legend(title="Memory Type")
        else:
            ax.axis("off")
    else:
        ax.set_title("Unified Memory Efficiency (no cross-platform data yet)")
        ax.text(0.5, 0.5, "Run on Apple Silicon and submit results\nto populate this chart.",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=11, color="grey")
        ax.axis("off")

    # ─────────────────────────────────────────────────────────────────────
    # Row 3 — Cold Start Latency (NEW) | Thermal / SoC Power over time
    # ─────────────────────────────────────────────────────────────────────

    # ── Cold Start Latency ────────────────────────────────────────────
    ax = axes[3, 0]
    if "cold_start_sec" in df.columns and df["cold_start_sec"].max() > 0:
        cold_df = (
            df.groupby(["model", "gpu_name", "memory_type"])
            ["cold_start_sec"].mean()
            .reset_index()
        )
        sns.barplot(
            data=cold_df, x="model", y="cold_start_sec",
            hue="memory_type", ax=ax,
        )
        ax.set_title("Cold Start Latency — Model Load → First Token")
        ax.set_xlabel("")
        ax.set_ylabel("Cold Start (seconds)")
        _rotate_xlabels(ax)
        ax.legend(title="Memory Type")
    else:
        ax.set_title("Cold Start Latency (pending data)")
        ax.text(0.5, 0.5, "cold_start_sec = 0 in current dataset.\n"
                "Re-run orchestrator.py to capture this metric.",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=10, color="grey")
        ax.axis("off")

    # ── Thermal / SoC Power over time ────────────────────────────────
    ax = axes[3, 1]
    if os.path.exists(thermal_csv):
        try:
            tdf = pd.read_csv(thermal_csv)
            if not tdf.empty:
                tdf["time_sec"] = tdf["timestamp"] - tdf["timestamp"].min()

                # Detect whether this is Apple Silicon (clock=0) or NVIDIA
                has_clock = "clock_mhz" in tdf.columns and tdf["clock_mhz"].max() > 0
                is_apple  = not has_clock

                # Primary: temperature
                sns.lineplot(
                    data=tdf, x="time_sec", y="temp_c",
                    color="tomato", label="Temp (°C)", alpha=0.75, ax=ax,
                )
                ax.set_ylabel("Temperature (°C)", color="tomato")
                ax.tick_params(axis="y", labelcolor="tomato")

                ax2 = ax.twinx()
                if is_apple:
                    # Apple Silicon: plot SoC power on secondary axis
                    power_col = "power_w" if "power_w" in tdf.columns else None
                    if power_col:
                        sns.lineplot(
                            data=tdf, x="time_sec", y=power_col,
                            color="steelblue", label="SoC Power (W)", alpha=0.75, ax=ax2,
                        )
                        ax2.set_ylabel("SoC Package Power (W)", color="steelblue")
                        ax2.tick_params(axis="y", labelcolor="steelblue")
                    ax.set_title("Thermal & SoC Power Profile (Apple Silicon)")
                else:
                    # NVIDIA: plot SM clock speed
                    sns.lineplot(
                        data=tdf, x="time_sec", y="clock_mhz",
                        color="steelblue", label="SM Clock (MHz)", alpha=0.75, ax=ax2,
                    )
                    ax2.set_ylabel("SM Clock Speed (MHz)", color="steelblue")
                    ax2.tick_params(axis="y", labelcolor="steelblue")

                    # Dynamic base-clock from CSV, fallback to 1440 for RTX 3080 compat
                    base_clock = (
                        df["base_clock_mhz"].max()
                        if "base_clock_mhz" in df.columns and df["base_clock_mhz"].max() > 0
                        else 1440.0
                    )
                    throttled = tdf[tdf["clock_mhz"] < base_clock]
                    if not throttled.empty:
                        ax2.scatter(
                            throttled["time_sec"], throttled["clock_mhz"],
                            color="black", marker="x", s=40,
                            label=f"Sub-Base (<{base_clock:.0f} MHz)", zorder=5,
                        )
                    ax.set_title("Thermal Decay & Throttling Analysis (NVIDIA)")

                # Unified legend
                l1, lab1 = ax.get_legend_handles_labels()
                l2, lab2 = ax2.get_legend_handles_labels()
                ax2.legend(l1 + l2, lab1 + lab2, loc="lower left", fontsize=8)
                if ax.get_legend():
                    ax.get_legend().remove()

                ax.set_xlabel("Continuous Runtime (Seconds)")
            else:
                ax.axis("off")
        except pd.errors.EmptyDataError:
            ax.axis("off")
    else:
        ax.axis("off")

    # --- Save ------------------------------------------------------------
    plt.tight_layout(rect=[0, 0, 1, 0.995])
    output_img = os.path.join(results_dir, "dashboard.png")
    plt.savefig(output_img, bbox_inches="tight", dpi=300)
    print(f"✅ Dashboard saved → {output_img}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM Inference Telemetry Suite — Visualization Engine"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=os.path.join(os.path.abspath(os.path.dirname(__file__)), "../results"),
        help="Path to a results directory (single GPU) or the root results/ folder.",
    )
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help=(
            "Merge all results/*/production_benchmarks.csv files for "
            "cross-platform comparison. Use with --results-dir pointing "
            "at the root results/ folder."
        ),
    )
    args = parser.parse_args()
    create_dashboard(
        results_dir=os.path.abspath(args.results_dir),
        aggregate=args.aggregate,
    )


if __name__ == "__main__":
    main()