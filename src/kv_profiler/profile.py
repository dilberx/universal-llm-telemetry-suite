"""Run a calibration pass and emit per-layer KV retention curves.

This is the measurement half of the profiler. It reports what each layer's
attention mass looks like as a function of cache budget, and stops there. It
deliberately does not recommend a budget, a policy or a configuration: turning
curves into a deployed config is a separate decision that depends on the
serving target, and it does not belong in a measurement tool.

One forward pass per calibration window yields the whole curve at every budget,
so profiling a model costs about the same as running it once.
"""

from __future__ import annotations

import glob
import json
import os
import time
from datetime import datetime

import torch

from src.kv_profiler.collect import collect_key_mass
from src.kv_profiler.curves import POLICY_BUILDERS

BUDGET_FRACTIONS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 0.9]
CURVE_POINTS = 101  # curve stored at 1% budget intervals

FALLBACK_TEXT = (
    "The attention mechanism computes a weighted sum over value vectors, where the "
    "weights come from a softmax over query-key dot products. Because every query "
    "attends to every preceding key, the cost of exact attention grows quadratically "
    "with sequence length, and the key-value cache that avoids recomputing keys and "
    "values grows linearly with it. Cache eviction policies decide which entries to "
    "discard when that growth exceeds available memory. "
)


def calibration_text(min_chars: int = 40000) -> str:
    """Real text if wikitext-2 is in the local HF cache, otherwise a fallback.

    The fallback is repetitive, which makes attention less representative than
    natural prose. Profiles built from it carry a flag saying so.
    """
    pattern = os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--wikitext/snapshots/*/wikitext-2-raw-v1/train-*.parquet"
    )
    matches = glob.glob(pattern)
    if matches:
        try:
            import pandas as pd
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
            if parts:
                return "\n".join(parts)
        except Exception:
            pass
    return FALLBACK_TEXT * (min_chars // len(FALLBACK_TEXT) + 1)


def _downsample(curve, n_points: int = CURVE_POINTS) -> list[float]:
    """Curve at evenly spaced budget fractions, so files stay a sane size."""
    seq = curve.seq_len
    return [round(float(curve.values[min(seq, int(round(seq * i / (n_points - 1))))]), 6)
            for i in range(n_points)]


def profile_model(src: str, tokens: int = 1024, chunk: int = 128,
                  windows: int = 1, text: str | None = None,
                  policies: tuple[str, ...] = ("h2o", "window", "sink")) -> dict:
    """Profile one model and return per-layer retention curves.

    Args:
        src: HF model id or a local directory.
        tokens: calibration sequence length.
        chunk: prefill chunk size; lower means less peak memory, same result.
        windows: number of non-overlapping calibration windows to average.
        text: calibration text; natural prose from the local cache by default.
        policies: which eviction policies to build curves for.
    """
    from transformers import AutoTokenizer

    from src.kv_profiler.loading import load_for_profiling

    print(f"Profiling {src}")
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(src, trust_remote_code=True)
    model = load_for_profiling(src)
    cfg = model.config
    device = next(model.parameters()).device

    used_fallback = text is None and not glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--wikitext/snapshots/*/wikitext-2-raw-v1/train-*.parquet"))
    body = text or calibration_text()
    ids_all = tok(body, return_tensors="pt").input_ids
    if ids_all.shape[1] < tokens:
        raise ValueError(f"calibration text is {ids_all.shape[1]} tokens, need {tokens}")

    n_layers = cfg.num_hidden_layers
    summed = torch.zeros(n_layers, tokens, dtype=torch.float64)
    used = 0
    for w in range(windows):
        lo = w * tokens
        if ids_all.shape[1] < lo + tokens:
            break
        ids = ids_all[:, lo:lo + tokens].to(device)
        summed += collect_key_mass(model, ids, chunk_size=chunk)
        used += 1
        print(f"  window {w + 1}/{windows} done")
    if used == 0:
        raise ValueError("no calibration window fitted in the text")
    mass = summed / used

    layers = []
    for i in range(n_layers):
        curves = {p: POLICY_BUILDERS[p](mass[i], i) for p in policies}
        knee_budget, knee_ret = curves["h2o"].knee()
        layers.append({
            "layer": i,
            "knee_budget_frac": round(knee_budget / tokens, 6),
            "knee_retention": round(knee_ret, 6),
            "retention_at": {p: {str(f): round(c.at_fraction(f), 6)
                                 for f in BUDGET_FRACTIONS}
                             for p, c in curves.items()},
            "curve": {p: _downsample(c) for p, c in curves.items()},
        })

    elapsed = time.perf_counter() - t0
    print(f"  {n_layers} layers profiled in {elapsed:.1f}s")
    del model
    torch.cuda.empty_cache()

    return {
        "schema": "kv-retention-profile/1",
        "model": src,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "architecture": {
            "layers": n_layers,
            "attention_heads": cfg.num_attention_heads,
            "kv_heads": getattr(cfg, "num_key_value_heads", cfg.num_attention_heads),
            "hidden_size": cfg.hidden_size,
        },
        "calibration": {
            "tokens": tokens,
            "windows": used,
            "chunk_size": chunk,
            "dtype": "bfloat16",
            "text_source": "fallback (repetitive, not representative)" if used_fallback
                           else "wikitext-2",
            "seconds": round(elapsed, 1),
        },
        "budget_fractions": BUDGET_FRACTIONS,
        "curve_points": CURVE_POINTS,
        "note": ("Attention-mass retention is a proxy for output quality, not a "
                 "measurement of it. This file reports measurements only and makes "
                 "no configuration recommendation."),
        "layers": layers,
    }


def print_summary(profile: dict, top_n: int = 5) -> None:
    """Human-readable digest of a profile."""
    arch, cal = profile["architecture"], profile["calibration"]
    print(f"\n{'=' * 68}")
    print(f"KV RETENTION PROFILE: {profile['model']}")
    print(f"{'=' * 68}")
    print(f"{arch['layers']} layers, {arch['attention_heads']} heads "
          f"({arch['kv_heads']} KV), hidden {arch['hidden_size']}")
    print(f"calibration: {cal['tokens']} tokens x{cal['windows']} window(s), "
          f"{cal['text_source']}, {cal['seconds']}s")

    layers = profile["layers"]
    print(f"\nRetention at each budget, median across layers:")
    print(f"{'budget':>8}" + "".join(f"{p:>12}" for p in ("h2o", "window", "sink")))
    for f in profile["budget_fractions"]:
        row = f"{f:>7.0%} "
        for p in ("h2o", "window", "sink"):
            vals = sorted(l["retention_at"][p][str(f)] for l in layers)
            row += f"{vals[len(vals) // 2]:>12.3f}"
        print(row)

    knees = sorted(layers, key=lambda l: l["knee_budget_frac"])
    mean_knee = sum(l["knee_budget_frac"] for l in layers) / len(layers)
    print(f"\nH2O knee, the budget past which extra cache buys little:")
    print(f"  mean {mean_knee:.3f} of sequence, range "
          f"{knees[0]['knee_budget_frac']:.3f} to {knees[-1]['knee_budget_frac']:.3f}")
    print(f"  cheapest layers: " +
          ", ".join(f"L{l['layer']} ({l['knee_budget_frac']:.3f})" for l in knees[:top_n]))
    print(f"  costliest layers: " +
          ", ".join(f"L{l['layer']} ({l['knee_budget_frac']:.3f})" for l in knees[-top_n:]))
    print(f"\n{profile['note']}")


def write_profile(profile: dict, out_dir: str = "reports/profiles") -> str:
    os.makedirs(out_dir, exist_ok=True)
    slug = profile["model"].rstrip("/").split("/")[-1].replace(".", "_").lower()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"profile_{slug}_{stamp}.json")
    with open(path, "w") as fh:
        json.dump(profile, fh, indent=2)
    return path
