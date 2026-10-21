"""Experiment 2A: Does the single-pass profiler reproduce the grid sweep?

The claim under test is that `src.kv_profiler.curves` returns the same retention
numbers as `exp_real_model.analyze_real_kv_cache`, at every budget the grid
evaluates, while reading the attention tensor once instead of once per budget.

This is the fidelity check that has to pass before any curve produced by the
profiler is trusted. The grid sweep is the incumbent and is treated as ground
truth. Both paths run against the same extracted attention in the same process,
so any difference is implementation, not sampling.

A second section measures the SHARDS-style key sampling against the exact curve,
to find the rate at which the approximation stops being usable.
"""

from __future__ import annotations

import time

import torch

from src.experiments.exp_real_model import (
    PROMPTS,
    analyze_real_kv_cache,
    extract_attention_patterns,
    load_model,
)
from src.experiments.runner import ExperimentRunner
from src.kv_profiler.curves import key_mass, layer_curves, sampled_key_mass, POLICY_BUILDERS

GRID = [0.1, 0.2, 0.3, 0.5, 0.75]
SAMPLE_RATES = [0.5, 0.25, 0.1, 0.05]


def run_profiler_validation(model_name: str = "Qwen/Qwen2.5-0.5B", prompts: list[str] | None = None):
    runner = ExperimentRunner(
        name="profiler_validation",
        description=(
            "Fidelity and cost of single-pass retention curves against the incumbent "
            "grid sweep, on real attention from "
            f"{model_name}. Also measures SHARDS-style key sampling error."
        ),
    )

    model, tokenizer = load_model(model_name)
    prompt_names = prompts or list(PROMPTS)

    for prompt_name in prompt_names:
        print(f"\n=== {prompt_name} ===")
        data = extract_attention_patterns(model, tokenizer, PROMPTS[prompt_name], max_new_tokens=0)
        attentions = data["attentions"]
        n_layers = len(attentions)
        seq_len = attentions[0].shape[-1]
        print(f"  {n_layers} layers, seq_len {seq_len}")

        # --- incumbent grid sweep ---------------------------------------
        t0 = time.perf_counter()
        grid_rows = analyze_real_kv_cache(attentions, GRID)
        grid_s = time.perf_counter() - t0

        # --- single pass ------------------------------------------------
        t0 = time.perf_counter()
        curves = [layer_curves(attentions[i][0], layer=i) for i in range(n_layers)]
        single_s = time.perf_counter() - t0

        print(f"  grid sweep {grid_s*1000:7.1f} ms   single pass {single_s*1000:7.1f} ms   "
              f"speedup {grid_s/single_s:.2f}x")

        # --- fidelity ----------------------------------------------------
        worst = 0.0
        by_policy = {"h2o": 0.0, "window": 0.0, "sink": 0.0}
        for row in grid_rows:
            layer, frac = row["layer"], row["budget_frac"]
            for policy, key in [("h2o", "h2o_quality"), ("window", "window_quality"), ("sink", "sink_quality")]:
                got = curves[layer][policy].at_fraction(frac)
                err = abs(got - row[key])
                by_policy[policy] = max(by_policy[policy], err)
                worst = max(worst, err)

        print(f"  max abs error vs grid: " +
              "  ".join(f"{p}={e:.3e}" for p, e in by_policy.items()))

        with runner.trial({"prompt": prompt_name, "check": "fidelity", "seq_len": seq_len}) as trial:
            trial.record("max_abs_error", worst)
            for p, e in by_policy.items():
                trial.record(f"max_abs_error_{p}", e)
            trial.record("grid_seconds", grid_s)
            trial.record("single_pass_seconds", single_s)
            trial.record("speedup", grid_s / single_s if single_s > 0 else 0.0)
            trial.record("grid_points", float(len(GRID)))
            trial.record("curve_points", float(seq_len + 1))
            trial.record("n_layers", float(n_layers))

        # --- knees, which the grid cannot resolve -------------------------
        for policy in ("h2o", "window"):
            knees = [curves[i][policy].knee() for i in range(n_layers)]
            budgets = [k / seq_len for k, _ in knees]
            with runner.trial({"prompt": prompt_name, "check": "knee", "policy": policy}) as trial:
                trial.record("knee_frac_mean", sum(budgets) / len(budgets))
                trial.record("knee_frac_min", min(budgets))
                trial.record("knee_frac_max", max(budgets))
                trial.record("knee_retention_mean", sum(r for _, r in knees) / len(knees))
                trial.record_meta("knee_frac_by_layer", [round(b, 4) for b in budgets])

        # --- sampling fidelity --------------------------------------------
        for rate in SAMPLE_RATES:
            errs = []
            for i in range(n_layers):
                exact = curves[i]["h2o"]
                s_mass, realised = sampled_key_mass(attentions[i][0], rate)
                approx = POLICY_BUILDERS["h2o"](s_mass, i)
                for frac in GRID:
                    errs.append(abs(approx.at_fraction(frac) - exact.at_fraction(frac)))
            with runner.trial({"prompt": prompt_name, "check": "sampling", "rate": rate}) as trial:
                trial.record("mean_abs_error", sum(errs) / len(errs))
                trial.record("max_abs_error", max(errs))
                trial.record("keys_kept", float(round(seq_len * rate)))
            print(f"  sample rate {rate:<5} mean err {sum(errs)/len(errs):.4f}  max err {max(errs):.4f}")

    # --- resolution scaling -------------------------------------------
    # The structural property: grid cost is linear in the number of budget
    # points, single-pass cost is independent of it. Synthetic attention keeps
    # this section free of model and hardware variance.
    print("\n=== resolution scaling (synthetic attention, 16 heads, seq 1024) ===")
    torch.manual_seed(0)
    synth = torch.rand(16, 1024, 1024)
    for n_points in [5, 10, 20, 50]:
        budgets = [(i + 1) / (n_points + 1) for i in range(n_points)]
        t0 = time.perf_counter()
        analyze_real_kv_cache([synth.unsqueeze(0)], budgets)
        grid_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        layer_curves(synth, layer=0)
        single_s = time.perf_counter() - t0
        print(f"  {n_points:>3} budget points   grid {grid_s*1000:7.1f} ms   "
              f"single pass {single_s*1000:6.1f} ms   speedup {grid_s/single_s:5.2f}x")
        with runner.trial({"check": "resolution_scaling", "grid_points": n_points}) as trial:
            trial.record("grid_seconds", grid_s)
            trial.record("single_pass_seconds", single_s)
            trial.record("speedup", grid_s / single_s if single_s > 0 else 0.0)
            trial.record("seq_len", 1024.0)

    runner.save()
    runner.to_csv()
    print(runner.report.summary())
    return runner


if __name__ == "__main__":
    run_profiler_validation()
