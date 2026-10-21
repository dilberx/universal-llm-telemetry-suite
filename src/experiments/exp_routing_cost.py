"""Measured per-token cost of the SEER model pair under batched serving.

The routing rerun measured a cost ratio of 0.844 (Qwen2.5-0.5B against Phi-2) in a
batch-1 eager-attention loop with per-token Python overhead, where the two models
cost about the same. The README then quoted 38% and 44% savings from an assumed
ratio. This replaces the assumption with a measurement: each model decodes the
routing prompts at several batch sizes with SDPA attention and the KV cache, and
the per-sequence-token time gives the cost ratio at that batch size.

Each ratio is then applied to the measured routing trials
(`prompt_routing_20260916_133528.json`) at the README's operating point (escalate
when mean top-probability over the first K=3 tokens is below 0.55), with the same
cost model as `exp_routing.py`.

Run: python -m src.experiments.exp_routing_cost
"""

from __future__ import annotations

import gc
import glob
import json
import time

import torch

from src.experiments.exp_routing import K, ROUTING_PROMPTS
from src.experiments.runner import ExperimentRunner

MODELS = ("Qwen/Qwen2.5-0.5B", "microsoft/phi-2")
BATCH_SIZES = (1, 4, 16, 32)
NEW_TOKENS = 20          # matches MAX_TOKENS in the routing run
REPEATS = 3
THRESHOLD = 0.55         # the README's operating point
ROUTING_FILE = "reports/experiments/prompt_routing_20260916_133528.json"


@torch.no_grad()
def _decode_ms_per_token(model, tok, batch_size: int) -> tuple[float, float]:
    """Median over repeats of wall time per generated sequence-token, and of prefill.

    Prompts cycle through the routing set to fill the batch. Generation is forced
    to exactly NEW_TOKENS so every batch does the same amount of decode work.
    """
    names = list(ROUTING_PROMPTS)
    texts = [ROUTING_PROMPTS[names[i % len(names)]] for i in range(batch_size)]
    enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
    kw = dict(do_sample=False, pad_token_id=tok.pad_token_id)

    def timed(n_new):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.generate(**enc, max_new_tokens=n_new, min_new_tokens=n_new, **kw)
        torch.cuda.synchronize()
        return time.perf_counter() - t0

    timed(NEW_TOKENS)                                   # warm-up: kernels, allocator
    totals = sorted(timed(NEW_TOKENS) for _ in range(REPEATS))
    prefills = sorted(timed(1) for _ in range(REPEATS))
    total, prefill = totals[REPEATS // 2], prefills[REPEATS // 2]
    decode_per_token = (total - prefill) / ((NEW_TOKENS - 1) * batch_size)
    return decode_per_token * 1000, prefill * 1000


def _savings(cost_ratio: float, trials: list[dict]) -> tuple[float, int]:
    """Mean savings against always calling the large model, and prompts kept local."""
    total_routed = total_large = 0.0
    kept = 0
    for t in trials:
        m = t["metrics"]
        n_small, n_large = m["small_tokens"], m["large_tokens"]
        n_k = min(K, n_small)
        if m["avg_k_prob"] < THRESHOLD:
            total_routed += n_k * cost_ratio + n_large
        else:
            total_routed += n_small * cost_ratio
            kept += 1
        total_large += n_large
    return 1.0 - total_routed / total_large, kept


def run_routing_cost():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    runner = ExperimentRunner(
        name="routing_cost",
        description=(
            "Per-token decode cost of Qwen2.5-0.5B and Phi-2 at batch 1 to 32, SDPA attention, "
            "bfloat16, and the SEER savings implied by each measured cost ratio."
        ),
    )
    ms = {}
    for name in MODELS:
        tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=torch.bfloat16, device_map="cuda",
            trust_remote_code=True, attn_implementation="sdpa").eval()
        for bs in BATCH_SIZES:
            decode, prefill = _decode_ms_per_token(model, tok, bs)
            ms[(name, bs)] = decode
            print(f"  {name:<22} batch {bs:>3}: {decode:7.2f} ms per sequence-token, "
                  f"prefill {prefill:7.1f} ms", flush=True)
            with runner.trial({"model": name, "batch_size": bs, "part": "latency"}) as trial:
                trial.record("decode_ms_per_seq_token", decode)
                trial.record("prefill_ms", prefill)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    trials = json.load(open(ROUTING_FILE))["trials"]
    print(f"\nSavings at threshold {THRESHOLD} on {len(trials)} measured routing trials:")
    for bs in BATCH_SIZES:
        ratio = ms[(MODELS[0], bs)] / ms[(MODELS[1], bs)]
        saving, kept = _savings(ratio, trials)
        print(f"  batch {bs:>3}: cost ratio {ratio:.3f}, savings {saving:+.1%} "
              f"({kept} of {len(trials)} kept local)", flush=True)
        with runner.trial({"batch_size": bs, "part": "savings", "threshold": THRESHOLD}) as trial:
            trial.record("cost_ratio", ratio)
            trial.record("savings_vs_large_only", saving)
            trial.record("kept_local", float(kept))
    runner.save()
    runner.to_csv()
    return runner


if __name__ == "__main__":
    run_routing_cost()
