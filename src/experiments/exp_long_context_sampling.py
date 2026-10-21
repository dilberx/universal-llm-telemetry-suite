"""Experiment 2C: does sampling give usable retention curves on real long context?

`exp_profiler_validation` left the sampling question open. The real-model
prompts there are 7 to 121 tokens, where sampling 10% leaves a handful of keys,
and the only long-sequence numbers came from synthetic attention that is far
less skewed than the real thing. Real attention follows a power law, which is
exactly the regime where a sample can miss the heavy hitters that carry the mass.

This runs real text through a real model at 1k to 4k tokens, which the
incremental collector made affordable, and measures both sampling axes against
the exact curve:

  query sampling   Use a fraction of query rows. This is the axis where work
                   could be saved, because attention costs H * S^2 in the
                   queries while the mass vector it produces is only S long.

  key sampling     Subsample the mass vector after the fact, SHARDS style. This
                   saves the sort and the storage of a length-S vector, both of
                   which are trivial next to H * S^2, so it cannot save real
                   work here. Included because it is the obvious port of SHARDS
                   and it is worth knowing that the obvious port is the wrong
                   one.

What is measured here is statistical fidelity, not speed. This implementation
computes full attention and then selects query rows, so a lower rate does not
run faster; at seq 2048 it is 1.25s at rate 1.0 and 1.19s at rate 0.1, with the
selection overhead cancelling the smaller reduction. Realising the saving needs
an attention kernel that computes only the sampled rows of QK^T, which is not
done here. The question this answers is whether such a kernel would produce a
usable curve, and for query sampling the answer is yes.

Text is wikitext-2, read from the local parquet so the experiment needs no
network. Attention is float32, as the chunked accumulation requires.
"""

from __future__ import annotations

import glob
import os

import torch

from src.experiments.runner import ExperimentRunner
from src.kv_profiler.collect import collect_key_mass
from src.kv_profiler.curves import POLICY_BUILDERS, h2o_curve, sampled_key_mass, window_curve

SEQ_LENGTHS = [1024, 2048, 4096]
QUERY_RATES = [0.5, 0.25, 0.1, 0.05]
KEY_RATES = [0.5, 0.25, 0.1, 0.05]
GRID = [0.1, 0.2, 0.3, 0.5, 0.75]
CHUNK = 128
SEEDS = [0, 1, 2]


def _wikitext(min_chars: int) -> str:
    """Real natural text from the local wikitext-2 parquet."""
    import pandas as pd

    pattern = os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--wikitext/snapshots/*/wikitext-2-raw-v1/train-*.parquet"
    )
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(
            "wikitext-2 parquet not found in the HF cache. "
            "Fetch it with datasets.load_dataset('wikitext', 'wikitext-2-raw-v1')."
        )
    df = pd.read_parquet(matches[0])
    parts, total = [], 0
    for line in df["text"]:
        line = str(line).strip()
        if len(line) < 80:  # skip headings and blanks
            continue
        parts.append(line)
        total += len(line)
        if total >= min_chars:
            break
    return "\n".join(parts)


def _curve_errors(exact, approx, seq_len: int) -> dict[str, float]:
    """Error at the grid fractions, plus how far the knee moves."""
    at = [abs(approx.at_fraction(f) - exact.at_fraction(f)) for f in GRID]
    ke, _ = exact.knee()
    ka, _ = approx.knee()
    return {
        "mean_abs_error": sum(at) / len(at),
        "max_abs_error": max(at),
        "knee_shift_frac": abs(ke - ka) / seq_len,
    }


def run_long_context_sampling(model_name: str = "Qwen/Qwen2.5-0.5B"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    runner = ExperimentRunner(
        name="long_context_sampling",
        description=(
            "Query and key sampling error against exact retention curves on real "
            f"wikitext attention from {model_name} at 1k to 4k tokens. Float32 attention, "
            f"chunked prefill at C={CHUNK}."
        ),
    )

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, device_map="cuda",
        trust_remote_code=True, attn_implementation="eager",
    ).eval()

    text = _wikitext(min_chars=30000)
    all_ids = tok(text, return_tensors="pt").input_ids.to("cuda")
    print(f"wikitext tokenised to {all_ids.shape[1]} tokens")

    for seq_len in SEQ_LENGTHS:
        if all_ids.shape[1] < seq_len:
            print(f"skipping {seq_len}, only {all_ids.shape[1]} tokens")
            continue
        ids = all_ids[:, :seq_len]
        print(f"\n=== seq_len {seq_len} ===")

        exact_mass = collect_key_mass(model, ids, chunk_size=CHUNK)
        n_layers = exact_mass.shape[0]
        exact = {
            "h2o": [h2o_curve(exact_mass[L], L) for L in range(n_layers)],
            "window": [window_curve(exact_mass[L], L) for L in range(n_layers)],
        }

        # Concentration of the real mass, which is what makes sampling hard or easy.
        top1 = float(sorted(exact_mass[0].tolist(), reverse=True)[0] / exact_mass[0].sum())
        frac_for_half = min(
            (b for b in range(1, seq_len + 1) if float(exact["h2o"][0].values[b]) >= 0.5),
            default=seq_len) / seq_len
        print(f"  layer 0: top key holds {top1:.3f} of mass; "
              f"{frac_for_half:.3f} of keys hold half the mass")

        for rate in QUERY_RATES:
            per_seed = []
            for seed in SEEDS:
                approx_mass = collect_key_mass(model, ids, chunk_size=CHUNK,
                                               query_rate=rate, query_seed=seed)
                for policy, builder in (("h2o", h2o_curve), ("window", window_curve)):
                    for L in range(n_layers):
                        per_seed.append((policy, _curve_errors(
                            exact[policy][L], builder(approx_mass[L], L), seq_len)))
            for policy in ("h2o", "window"):
                rows = [e for p, e in per_seed if p == policy]
                mean_err = sum(r["mean_abs_error"] for r in rows) / len(rows)
                max_err = max(r["max_abs_error"] for r in rows)
                knee = sum(r["knee_shift_frac"] for r in rows) / len(rows)
                with runner.trial({"seq_len": seq_len, "axis": "query",
                                   "rate": rate, "policy": policy}) as trial:
                    trial.record("mean_abs_error", mean_err)
                    trial.record("max_abs_error", max_err)
                    trial.record("mean_knee_shift_frac", knee)
                    trial.record("sampled_query_fraction", rate)
                    trial.record("n_seeds", float(len(SEEDS)))
                if policy == "h2o":
                    print(f"  query rate {rate:<5} h2o  mean err {mean_err:.4f}  "
                          f"max {max_err:.4f}  knee shift {knee:.4f}")

        for rate in KEY_RATES:
            errs, knees = [], []
            for L in range(n_layers):
                s_mass, _ = sampled_key_mass(exact_mass[L].unsqueeze(0).unsqueeze(0), rate)
                approx = POLICY_BUILDERS["h2o"](s_mass, L)
                e = _curve_errors(exact["h2o"][L], approx, seq_len)
                errs.append(e["mean_abs_error"])
                knees.append(e["knee_shift_frac"])
            with runner.trial({"seq_len": seq_len, "axis": "key", "rate": rate, "policy": "h2o"}) as trial:
                trial.record("mean_abs_error", sum(errs) / len(errs))
                trial.record("max_abs_error", max(errs))
                trial.record("mean_knee_shift_frac", sum(knees) / len(knees))
                trial.record("sampled_query_fraction", 1.0)
            print(f"  key   rate {rate:<5} h2o  mean err {sum(errs)/len(errs):.4f}  "
                  f"max {max(errs):.4f}")

    runner.save()
    runner.to_csv()
    print(runner.report.summary())
    return runner


if __name__ == "__main__":
    run_long_context_sampling()
