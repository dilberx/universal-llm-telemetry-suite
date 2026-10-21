"""Experiment 2B: peak memory of incremental mass accumulation.

Three ways to get the same `[layers, keys]` mass matrix:

  reference   `output_attentions=True`, keep every layer, then reduce.
              Peak attention memory L * H * S^2.
  one-shot    Reduce each layer as its attention module returns and drop the
              tensor there. Peak H * S^2.
  chunked     Feed the prompt in chunks of C with a KV cache, reducing each
              chunk's attention as it appears. Peak H * C * S.

All three must return the same numbers. This experiment checks that and
measures what each one costs.

Attention is computed in float32. bfloat16 is not accurate enough for the
chunked path: a chunked matmul accumulates the QK product in a different order
than a single full-sequence matmul, and with 8 mantissa bits that shifts mass
between near-tied keys in a peaked softmax. Measured on Qwen2.5-0.5B at
sequence length 145, chunking in bfloat16 gives up to 29% relative error on
per-key mass against 1.4e-05 in float32. The profiler is calibration tooling
that runs once, so paying for float32 here is the right trade.
"""

from __future__ import annotations

import gc

import torch

from src.experiments.runner import ExperimentRunner
from src.kv_profiler.collect import collect_key_mass, reference_key_mass
from src.kv_profiler.curves import h2o_curve

SEQ_LENGTHS = [256, 512, 1024]
CHUNK_SIZES = [128, 64]
FILLER = "The transformer architecture was introduced in 2017 and changed how sequence models work. "


def _peak_mb(fn):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    result = fn()
    peak = torch.cuda.max_memory_allocated() - before
    return result, peak / 1024 ** 2


def run_profiler_memory(model_name: str = "Qwen/Qwen2.5-0.5B"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    runner = ExperimentRunner(
        name="profiler_memory",
        description=(
            "Peak GPU memory and fidelity for three ways of computing per-key attention "
            f"mass on {model_name}: keeping every layer's attention, reducing each layer "
            "in a hook, and chunked prefill. Attention in float32."
        ),
    )

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, device_map="cuda",
        trust_remote_code=True, attn_implementation="eager",
    ).eval()
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads

    long_text = FILLER * 400
    all_ids = tok(long_text, return_tensors="pt").input_ids.to("cuda")

    for seq_len in SEQ_LENGTHS:
        if all_ids.shape[1] < seq_len:
            print(f"skipping {seq_len}, only {all_ids.shape[1]} tokens available")
            continue
        ids = all_ids[:, :seq_len]
        print(f"\n=== seq_len {seq_len} ===")

        ref, ref_mb = _peak_mb(lambda: reference_key_mass(model, ids))
        print(f"  reference (all layers held)   {ref_mb:8.1f} MB")

        one, one_mb = _peak_mb(lambda: collect_key_mass(model, ids))
        err_one = float((ref - one).abs().max())
        print(f"  one-shot reduction            {one_mb:8.1f} MB   "
              f"{ref_mb/max(one_mb,1e-9):5.1f}x less   max abs err {err_one:.1e}")

        with runner.trial({"seq_len": seq_len, "method": "reference"}) as trial:
            trial.record("peak_mb", ref_mb)
            trial.record("max_abs_error", 0.0)
            trial.record("predicted_attn_mb", n_layers * n_heads * seq_len ** 2 * 4 / 1024 ** 2)
        with runner.trial({"seq_len": seq_len, "method": "one_shot_hook"}) as trial:
            trial.record("peak_mb", one_mb)
            trial.record("max_abs_error", err_one)
            trial.record("reduction_vs_reference", ref_mb / max(one_mb, 1e-9))
            trial.record("predicted_attn_mb", n_heads * seq_len ** 2 * 4 / 1024 ** 2)

        for chunk in CHUNK_SIZES:
            if chunk >= seq_len:
                continue
            got, mb = _peak_mb(lambda c=chunk: collect_key_mass(model, ids, chunk_size=c))
            rel = float(((ref - got).abs() / ref.abs().clamp(min=1e-12)).max())
            curve_err = max(
                float((h2o_curve(ref[L]).values - h2o_curve(got[L]).values).abs().max())
                for L in range(ref.shape[0])
            )
            print(f"  chunked C={chunk:<4}                 {mb:8.1f} MB   "
                  f"{ref_mb/max(mb,1e-9):5.1f}x less   max rel err {rel:.1e}   "
                  f"curve err {curve_err:.1e}")
            with runner.trial({"seq_len": seq_len, "method": "chunked", "chunk_size": chunk}) as trial:
                trial.record("peak_mb", mb)
                trial.record("max_rel_error", rel)
                trial.record("max_curve_error", curve_err)
                trial.record("reduction_vs_reference", ref_mb / max(mb, 1e-9))
                trial.record("predicted_attn_mb", n_heads * chunk * seq_len * 4 / 1024 ** 2)

    runner.save()
    runner.to_csv()
    print(runner.report.summary())
    return runner


if __name__ == "__main__":
    run_profiler_memory()
