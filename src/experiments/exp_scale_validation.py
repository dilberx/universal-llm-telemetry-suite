"""Experiment 2D: does the H2O over window gap keep widening at 7B?

The repository's headline empirical claim is that real attention makes window
eviction fail far worse than synthetic benchmarks suggest, and that the gap
grows with model size: a median H2O over window retention ratio near 36x on
Qwen2.5-0.5B and near 207x on Phi-2 2.7B at a 10% KV budget. Two points is a
direction, not a trend, and the obvious question from anyone reading it is what
happens at 7B.

This adds a third point. Mistral-7B-Instruct-v0.2 does not fit in 10GB, so it
runs with its weights split across GPU and CPU. That is affordable only because
the profiler needs one forward pass per prompt to produce the whole curve;
the incumbent grid sweep would need one pass per budget point.

All three models are measured the same way in the same run: bfloat16 weights,
eager attention, single-pass mass collection, identical prompts. Single-pass
collection is exact, so no chunking error enters these numbers.

Updated 2026-09-16. `exp_family_scaling` found that this ratio depends far more
on input length than on model size: one model moves 36.6x to 13.6x between
43-token prompts and 1024 tokens of natural text. The original short prompts
average 43 to 55 tokens, which is not a regime where anyone evicts a KV cache,
so the cross-model comparison is now run at a fixed 1024 tokens as well.

The three tokenizers differ, so a fixed token count means each model reads a
slightly different span of the same corpus. Length is the variable that
dominates, so it is the one held fixed. Four non-overlapping windows give a
sense of spread.
"""

from __future__ import annotations

import gc
from statistics import median

import torch

from src.experiments.exp_family_scaling import _wikitext
from src.experiments.exp_real_model import PROMPTS
from src.experiments.runner import ExperimentRunner
from src.kv_profiler.collect import collect_key_mass, reference_key_mass
from src.kv_profiler.curves import h2o_curve, window_curve

BUDGETS = [0.1, 0.2, 0.3, 0.5, 0.75]
FIXED_LEN = 1024
N_WINDOWS = 4

# Vicuna-7B is reached through the llava-1.5-7b-hf checkpoint, whose text tower
# is exactly it. It is here because the architecture comparison otherwise rests
# on Phi-2 alone, and Phi-2 is 2.7x smaller than the GQA model it is compared
# against. Vicuna gives an MHA model at the same scale as Mistral-7B, which
# separates attention design from size.
MODELS = [
    ("Qwen/Qwen2.5-0.5B", 0.5, None),
    ("microsoft/phi-2", 2.7, None),
    ("mistralai/Mistral-7B-Instruct-v0.2", 7.2, "auto"),
    ("/home/dilbersha/models/llava-1.5-7b-hf", 6.7, "auto"),
]


def _load(name, max_memory):
    """Load a checkpoint and return the text decoder plus its tokenizer.

    Placement comes from free VRAM at load time rather than the `max_memory`
    column, which is now advisory: this GPU is shared and a fixed map OOMs
    whenever something else is running.
    """
    from transformers import AutoTokenizer

    from src.kv_profiler.loading import load_for_profiling, unwrap_causal_lm

    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    model = load_for_profiling(name)
    return unwrap_causal_lm(model), tok


def run_scale_validation():
    runner = ExperimentRunner(
        name="scale_validation",
        description=(
            "H2O over window retention ratio across three model scales on identical "
            "prompts, measured with single-pass retention curves. Adds Mistral-7B as a "
            "third scale point to the 0.5B and 2.7B results. bfloat16, eager attention."
        ),
    )

    wiki_text = _wikitext(min_chars=80000)
    summary = {}
    for name, params_b, max_memory in MODELS:
        print(f"\n{'='*66}\n{name} ({params_b}B)\n{'='*66}")
        model, tok = _load(name, max_memory)
        n_layers = model.config.num_hidden_layers
        n_heads = model.config.num_attention_heads
        device = next(model.parameters()).device
        print(f"  {n_layers} layers, {n_heads} heads")

        ratios_at = {b: [] for b in BUDGETS}
        knee_fracs = []
        prompt_lens = []
        exactness_checked = False

        for prompt_name, prompt_text in PROMPTS.items():
            ids = tok(prompt_text, return_tensors="pt").input_ids.to(device)
            seq_len = ids.shape[1]
            if seq_len < 8:
                continue
            prompt_lens.append(seq_len)
            mass = collect_key_mass(model, ids)

            # Confirm the single-pass path is exact on this model too, once.
            if not exactness_checked:
                err = float((reference_key_mass(model, ids) - mass).abs().max())
                print(f"  single-pass vs reference: max abs err {err:.1e}")
                with runner.trial({"model": name, "check": "exactness"}) as trial:
                    trial.record("max_abs_error", err)
                    trial.record("n_layers", float(n_layers))
                    trial.record("params_b", params_b)
                exactness_checked = True

            for layer in range(n_layers):
                h = h2o_curve(mass[layer], layer)
                w = window_curve(mass[layer], layer)
                knee_fracs.append(h.knee()[0] / seq_len)
                for b in BUDGETS:
                    hv, wv = h.at_fraction(b), w.at_fraction(b)
                    ratios_at[b].append(hv / max(wv, 1e-10))

        for b in BUDGETS:
            vals = ratios_at[b]
            with runner.trial({"model": name, "params_b": params_b,
                               "prompts": "short", "budget_frac": b}) as trial:
                trial.record("h2o_over_window_median", median(vals))
                trial.record("h2o_over_window_mean", sum(vals) / len(vals))
                trial.record("h2o_over_window_max", max(vals))
                trial.record("n_observations", float(len(vals)))
        with runner.trial({"model": name, "params_b": params_b,
                           "prompts": "short", "check": "knee"}) as trial:
            trial.record("knee_frac_mean", sum(knee_fracs) / len(knee_fracs))
            trial.record("knee_frac_min", min(knee_fracs))
            trial.record("knee_frac_max", max(knee_fracs))
        print(f"  short prompts (mean {sum(prompt_lens)/len(prompt_lens):.0f} tok)  " +
              "  ".join(f"{b:.0%}={median(ratios_at[b]):.1f}x" for b in BUDGETS))

        # --- fixed 1024 tokens, the regime eviction is actually for ---------
        wiki_ids = tok(wiki_text, return_tensors="pt").input_ids
        long_ratios = {b: [] for b in BUDGETS}
        long_knees = []
        per_window = {b: [] for b in BUDGETS}
        h2o_terms, win_terms = [], []
        for w in range(N_WINDOWS):
            lo = w * FIXED_LEN
            if wiki_ids.shape[1] < lo + FIXED_LEN:
                break
            ids = wiki_ids[:, lo:lo + FIXED_LEN].to(device)
            mass = collect_key_mass(model, ids)
            win_ratios = {b: [] for b in BUDGETS}
            for layer in range(n_layers):
                h = h2o_curve(mass[layer], layer)
                wc = window_curve(mass[layer], layer)
                long_knees.append(h.knee()[0] / FIXED_LEN)
                for b in BUDGETS:
                    hv, wv = h.at_fraction(b), wc.at_fraction(b)
                    win_ratios[b].append(hv / max(wv, 1e-10))
                    if b == 0.1:
                        h2o_terms.append(hv)
                        win_terms.append(wv)
            for b in BUDGETS:
                long_ratios[b].extend(win_ratios[b])
                per_window[b].append(median(win_ratios[b]))

        for b in BUDGETS:
            with runner.trial({"model": name, "params_b": params_b,
                               "prompts": f"wikitext_{FIXED_LEN}", "budget_frac": b}) as trial:
                trial.record("h2o_over_window_median", median(long_ratios[b]))
                trial.record("h2o_over_window_mean", sum(long_ratios[b]) / len(long_ratios[b]))
                trial.record("window_median_min", min(per_window[b]))
                trial.record("window_median_max", max(per_window[b]))
                trial.record("n_windows", float(len(per_window[b])))
                trial.record("n_observations", float(len(long_ratios[b])))
        with runner.trial({"model": name, "params_b": params_b,
                           "prompts": f"wikitext_{FIXED_LEN}", "check": "knee"}) as trial:
            trial.record("knee_frac_mean", sum(long_knees) / len(long_knees))
            trial.record("knee_frac_min", min(long_knees))
            trial.record("knee_frac_max", max(long_knees))
            trial.record("h2o_retention_median_10pct", median(h2o_terms))
            trial.record("window_retention_median_10pct", median(win_terms))
        print(f"  wikitext {FIXED_LEN} x{len(per_window[0.1])} windows           " +
              "  ".join(f"{b:.0%}={median(long_ratios[b]):.1f}x" for b in BUDGETS))
        print(f"    per-window median at 10%: "
              f"{min(per_window[0.1]):.1f}x to {max(per_window[0.1]):.1f}x")

        summary[name] = (params_b, n_layers,
                         {b: median(ratios_at[b]) for b in BUDGETS},
                         {b: median(long_ratios[b]) for b in BUDGETS},
                         sum(knee_fracs) / len(knee_fracs),
                         sum(long_knees) / len(long_knees),
                         median(h2o_terms), median(win_terms))

        del model
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n{'='*78}\nCROSS-MODEL, median H2O over window at a 10% budget\n{'='*78}")
    print(f"{'model':<30}{'params':>8}{'43-55 tok prompts':>20}{'wikitext 1024':>16}")
    for name, (p, L, short_med, long_med, sk, lk, h, w) in summary.items():
        print(f"{name.split('/')[-1]:<30}{p:>7.1f}B{short_med[0.1]:>19.1f}x{long_med[0.1]:>15.1f}x")
    print(f"\n{'model':<30}{'knee short':>12}{'knee 1024':>12}{'H2O@10%':>10}{'window@10%':>12}")
    for name, (p, L, short_med, long_med, sk, lk, h, w) in summary.items():
        print(f"{name.split('/')[-1]:<30}{sk:>12.3f}{lk:>12.3f}{h:>10.3f}{w:>12.4f}")

    runner.save()
    runner.to_csv()
    print(runner.report.summary())
    return runner


if __name__ == "__main__":
    run_scale_validation()
