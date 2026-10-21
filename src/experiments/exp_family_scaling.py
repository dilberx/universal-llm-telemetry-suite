"""Experiment 2E: does the H2O over window gap widen with scale, within one family?

`exp_scale_validation` compared Qwen2.5-0.5B, Phi-2 and Mistral-7B and found the
ratio was not monotone in parameter count. Those three differ in architecture,
attention type, tokenizer and instruction tuning, so they cannot separate scale
from anything else. Phi-2, the high outlier, is the only multi-head model.

This holds everything else fixed. Qwen2.5 ships base checkpoints at 0.5B, 1.5B,
3B and 7B with the same tokenizer, the same grouped-query attention and the same
training recipe. Same tokenizer matters most: every model sees a byte-identical
token sequence, so sequence length is controlled exactly rather than
approximately.

Two prompt sets are used. The eight prompts from `exp_real_model` keep continuity
with the earlier numbers. Fixed-length wikitext windows at 512 and 1024 tokens
give a controlled measurement at a length worth caring about, which the short
prompts (43 tokens mean) do not. `exp_scale_validation` established that the
ratio depends far more on input length than on model size, so the 1024-token
column is the one to read.
"""

from __future__ import annotations

import gc
import glob
import json
import os
from statistics import median

import torch

from src.experiments.exp_real_model import PROMPTS
from src.experiments.runner import ExperimentRunner
from src.kv_profiler.collect import collect_key_mass
from src.kv_profiler.curves import h2o_curve, window_curve

BUDGETS = [0.1, 0.2, 0.3, 0.5, 0.75]
FIXED_LENGTHS = [512, 1024]

# The intended run. Checkpoints whose weights are not present locally are
# skipped with a warning rather than failing the run, so this fills in as more
# of the family becomes available.
# Approximate bf16 weight footprint in GiB, used to decide placement at load
# time. This GPU is shared with other work, so free VRAM is not knowable in
# advance and a fixed device_map would OOM whenever something else is running.
WEIGHT_GIB = {"Qwen2.5-0.5B": 1.0, "Qwen2.5-1.5B": 3.1, "Qwen2.5-3B": 6.2,
              "Qwen2.5-7B": 15.2, "Qwen2.5-Coder-1.5B": 3.1}
# Headroom for activations, the KV cache and one layer's attention tensor.
# At 1024 tokens the largest of those is about 56MB, so 1.5 GiB is generous.
HEADROOM_GIB = 1.5


def _placement(short_name: str) -> dict | None:
    """Decide device placement from VRAM actually free right now.

    Returns None to load wholly on the GPU, or a max_memory dict for
    accelerate to split across GPU and CPU. Querying at load time rather than
    hardcoding matters because this card is shared: a run that fits when
    started may not fit an hour later.
    """
    free_b, _ = torch.cuda.mem_get_info()
    free_gib = free_b / 1024 ** 3
    need = WEIGHT_GIB.get(short_name, 99.0)
    if need + HEADROOM_GIB <= free_gib:
        print(f"  placement: all on GPU ({need:.1f} GiB weights, {free_gib:.1f} GiB free)")
        return None
    gpu_budget = max(0.5, free_gib - HEADROOM_GIB)
    print(f"  placement: split, {gpu_budget:.1f} GiB GPU + CPU "
          f"({need:.1f} GiB weights, {free_gib:.1f} GiB free)")
    return {0: f"{gpu_budget:.1f}GiB", "cpu": "11GiB"}


# Base checkpoints in size order, then Qwen2.5-Coder-1.5B as a control. Coder
# is the same size and architecture as Qwen2.5-1.5B but trained on a code-heavy
# mix, so the difference between those two is training data at fixed scale,
# which bounds how much of any trend below is really about size.
FAMILY = [
    ("Qwen/Qwen2.5-0.5B", 0.5, None),
    ("Qwen/Qwen2.5-1.5B", 1.5, None),
    ("Qwen/Qwen2.5-3B", 3.1, None),
    ("Qwen/Qwen2.5-7B", 7.6, "auto"),
    ("Qwen/Qwen2.5-Coder-1.5B", 1.5, None),
]


LOCAL_ROOT = "/home/dilbersha/models"


def _shard_complete(path: str) -> bool:
    """True if a safetensors shard is fully written.

    A partially downloaded shard still matches a glob, and loading one fails
    deep inside transformers with an opaque error. The header declares the byte
    range of every tensor, so the file is complete exactly when it is at least
    as large as the end of the last range.
    """
    try:
        with open(path, "rb") as fh:
            n = int.from_bytes(fh.read(8), "little")
            if n <= 0 or n > 200_000_000:
                return False
            header = json.loads(fh.read(n))
        end = max(v["data_offsets"][1] for k, v in header.items() if k != "__metadata__")
        return os.path.getsize(path) >= 8 + n + end
    except Exception:
        return False


def _resolve(name: str) -> str | None:
    """Where to load a checkpoint from, or None if its weights are missing.

    Prefers a plain directory under LOCAL_ROOT, then the HF cache. Config and
    tokenizer files are kilobytes and are often present for models whose
    weights are not, so the presence of a repo directory proves nothing: this
    checks for an actual weights file either way.
    """
    local = os.path.join(LOCAL_ROOT, name.split("/")[-1])
    shards = glob.glob(os.path.join(local, "*.safetensors"))
    if shards and all(_shard_complete(f) for f in shards):
        return local
    slug = "models--" + name.replace("/", "--")
    root = os.path.expanduser(f"~/.cache/huggingface/hub/{slug}")
    if glob.glob(os.path.join(root, "**", "*.safetensors"), recursive=True) or \
       glob.glob(os.path.join(root, "**", "*.bin"), recursive=True):
        return name
    return None


def _wikitext(min_chars: int = 40000) -> str:
    import pandas as pd

    pattern = os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--wikitext/snapshots/*/wikitext-2-raw-v1/train-*.parquet"
    )
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError("wikitext-2 parquet not in the HF cache")
    df = pd.read_parquet(matches[0])
    parts, total = [], 0
    for line in df["text"]:
        line = str(line).strip()
        if len(line) < 80:
            continue
        parts.append(line)
        total += len(line)
        if total >= min_chars:
            break
    return "\n".join(parts)


def _stats(mass, seq_len):
    """Per-layer H2O and window retention at each budget, their ratio, and knees.

    Both terms are kept, not just the ratio. A ratio that moves with scale could
    mean H2O improved or window degraded, and those imply different things.
    """
    ratios = {b: [] for b in BUDGETS}
    h2o_vals = {b: [] for b in BUDGETS}
    win_vals = {b: [] for b in BUDGETS}
    knees = []
    for layer in range(mass.shape[0]):
        h = h2o_curve(mass[layer], layer)
        w = window_curve(mass[layer], layer)
        knees.append(h.knee()[0] / seq_len)
        for b in BUDGETS:
            hv, wv = h.at_fraction(b), w.at_fraction(b)
            h2o_vals[b].append(hv)
            win_vals[b].append(wv)
            ratios[b].append(hv / max(wv, 1e-10))
    return ratios, knees, h2o_vals, win_vals


def _load_with_fallback(cls, src, short, kwargs):
    """Load a checkpoint, retrying with more CPU offload if the GPU is full.

    This card is shared with other jobs whose usage moves by several GiB, so a
    placement that fits when chosen can fail moments later. Each retry halves
    the GPU budget rather than giving up on the run.
    """
    placement = _placement(short)
    for attempt in range(4):
        try:
            if placement:
                return cls.from_pretrained(src, device_map="auto",
                                           max_memory=placement, **kwargs)
            return cls.from_pretrained(src, device_map="cuda", **kwargs)
        except torch.cuda.OutOfMemoryError:
            gc.collect()
            torch.cuda.empty_cache()
            free_gib = torch.cuda.mem_get_info()[0] / 1024 ** 3
            budget = max(0.5, (free_gib - HEADROOM_GIB) / (2 ** attempt))
            placement = {0: f"{budget:.1f}GiB", "cpu": "11GiB"}
            print(f"  OOM on attempt {attempt + 1}, retrying with "
                  f"{budget:.1f} GiB on GPU ({free_gib:.1f} GiB free)")
    raise RuntimeError(f"could not fit {short} after 4 attempts")


def run_family_scaling():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    runner = ExperimentRunner(
        name="family_scaling",
        description=(
            "H2O over window retention ratio across Qwen2.5 base checkpoints at 0.5B, "
            "1.5B, 3B and 7B. Same tokenizer, same grouped-query attention, same "
            "training recipe, byte-identical token sequences. bfloat16, eager attention, "
            "single-pass collection."
        ),
    )

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    wiki_ids = tok(_wikitext(), return_tensors="pt").input_ids
    print(f"wikitext tokenised to {wiki_ids.shape[1]} tokens (shared across the family)")

    resolved = {n: _resolve(n) for n, _, _ in FAMILY}
    available = [(n, p, m) for n, p, m in FAMILY if resolved[n]]
    missing = [n for n, _, _ in FAMILY if not resolved[n]]
    if missing:
        print(f"\nSKIPPING (weights not in local cache): {', '.join(missing)}")
        print("  Fetch them to complete the family, then rerun.\n")
    if len(available) < 2:
        raise RuntimeError(f"need at least two checkpoints, found {len(available)}")

    summary = {}
    for name, params_b, max_memory in available:
        short = name.split("/")[-1]
        print(f"\n{'='*70}\n{short} ({params_b}B)\n{'='*70}")
        src = resolved[name]
        kwargs = dict(torch_dtype=torch.bfloat16, attn_implementation="eager")
        model = _load_with_fallback(AutoModelForCausalLM, src, short, kwargs)
        model.eval()
        cfg = model.config
        device = next(model.parameters()).device
        print(f"  L={cfg.num_hidden_layers} heads={cfg.num_attention_heads} "
              f"kv={cfg.num_key_value_heads} hidden={cfg.hidden_size}")

        # --- the original short prompts, for continuity ------------------
        short_ratios = {b: [] for b in BUDGETS}
        short_knees = []
        for text in PROMPTS.values():
            ids = tok(text, return_tensors="pt").input_ids.to(device)
            if ids.shape[1] < 8:
                continue
            r, k, _, _ = _stats(collect_key_mass(model, ids), ids.shape[1])
            for b in BUDGETS:
                short_ratios[b].extend(r[b])
            short_knees.extend(k)
        for b in BUDGETS:
            with runner.trial({"model": short, "params_b": params_b,
                               "prompts": "short", "budget_frac": b}) as trial:
                trial.record("h2o_over_window_median", median(short_ratios[b]))
                trial.record("h2o_over_window_mean", sum(short_ratios[b]) / len(short_ratios[b]))
                trial.record("n_observations", float(len(short_ratios[b])))
        print("  short prompts   " +
              "  ".join(f"{b:.0%}={median(short_ratios[b]):.1f}x" for b in BUDGETS))

        # --- fixed-length wikitext, the controlled measurement ------------
        for seq_len in FIXED_LENGTHS:
            ids = wiki_ids[:, :seq_len].to(device)
            mass = collect_key_mass(model, ids)
            r, k, hv, wv = _stats(mass, seq_len)
            for b in BUDGETS:
                with runner.trial({"model": short, "params_b": params_b,
                                   "prompts": f"wikitext_{seq_len}", "budget_frac": b}) as trial:
                    trial.record("h2o_over_window_median", median(r[b]))
                    trial.record("h2o_over_window_mean", sum(r[b]) / len(r[b]))
                    trial.record("h2o_over_window_max", max(r[b]))
                    trial.record("h2o_retention_median", median(hv[b]))
                    trial.record("window_retention_median", median(wv[b]))
                    trial.record("n_observations", float(len(r[b])))
            with runner.trial({"model": short, "params_b": params_b,
                               "prompts": f"wikitext_{seq_len}", "check": "knee"}) as trial:
                trial.record("knee_frac_mean", sum(k) / len(k))
                trial.record("knee_frac_min", min(k))
                trial.record("knee_frac_max", max(k))
                trial.record("n_layers", float(mass.shape[0]))
            print(f"  wikitext {seq_len:<5}  " +
                  "  ".join(f"{b:.0%}={median(r[b]):.1f}x" for b in BUDGETS) +
                  f"   knee {sum(k)/len(k):.3f}")
            if seq_len == FIXED_LENGTHS[-1]:
                summary[short] = (params_b, cfg.num_hidden_layers,
                                  {b: median(r[b]) for b in BUDGETS}, sum(k) / len(k),
                                  {b: (median(hv[b]), median(wv[b])) for b in BUDGETS})

        del model
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n{'='*70}\nQWEN2.5 FAMILY, median H2O over window at {FIXED_LENGTHS[-1]} tokens\n{'='*70}")
    print(f"{'model':<20}{'params':>8}{'layers':>8}" + "".join(f"{b:>10.0%}" for b in BUDGETS) + f"{'knee':>9}")
    for short, (p, L, med, knee, terms) in summary.items():
        print(f"{short:<20}{p:>7.1f}B{L:>8}" +
              "".join(f"{med[b]:>9.1f}x" for b in BUDGETS) + f"{knee:>9.3f}")
    print(f"\n{'='*70}\nWHICH TERM MOVES, at 10% budget\n{'='*70}")
    print(f"{'model':<20}{'params':>8}{'H2O retention':>16}{'window retention':>18}")
    for short, (p, L, med, knee, terms) in summary.items():
        h, w = terms[0.1]
        print(f"{short:<20}{p:>7.1f}B{h:>16.3f}{w:>18.4f}")

    runner.save()
    runner.to_csv()
    print(runner.report.summary())
    return runner


if __name__ == "__main__":
    run_family_scaling()
