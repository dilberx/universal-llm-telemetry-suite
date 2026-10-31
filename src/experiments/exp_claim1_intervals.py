"""Bootstrap intervals for the three effect sizes behind the H2O-over-window ratio claim.

The scale and family runs store only the pooled median of the ratio and the min and max of the per-window
medians, so an interval cannot be recovered from them. This reruns the measurement on 20 non-overlapping
1024-token wikitext windows (windows 0-3 are the spans the original runs used) and keeps every per-layer ratio
at a 10% budget, then resamples windows.

Effects (each a ratio of two pooled-median ratios; intervals resample windows, or prompts for the short set):
  length        Qwen2.5-0.5B, short prompts / 1024-token windows
  architecture  Vicuna-7B (MHA) / Mistral-7B (GQA), 1024-token windows, resampled independently
  scale         Qwen2.5-7B / Qwen2.5-0.5B, 1024-token windows; same tokenizer, so windows are resampled together

Run:  python -m src.experiments.exp_claim1_intervals
"""

from __future__ import annotations

import gc
import json
import os
import random
import time
from statistics import median

from src.experiments.exp_family_scaling import _wikitext
from src.experiments.exp_real_model import PROMPTS
from src.kv_profiler.collect import collect_key_mass
from src.kv_profiler.curves import h2o_curve, window_curve

BUDGET = 0.1
FIXED_LEN = 1024
N_WINDOWS = 20
MODELS = {
    "qwen0.5b": "Qwen/Qwen2.5-0.5B",
    "qwen7b": "/home/dilbersha/models/Qwen2.5-7B",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
    "vicuna": "/home/dilbersha/models/llava-1.5-7b-hf",
}
OUT = "reports/experiments"


def _ratios(model, ids, n_layers):
    mass = collect_key_mass(model, ids)
    out = []
    for layer in range(n_layers):
        h, w = h2o_curve(mass[layer], layer), window_curve(mass[layer], layer)
        out.append(h.at_fraction(BUDGET) / max(w.at_fraction(BUDGET), 1e-10))
    return out


def measure(key, name, wiki_text):
    from transformers import AutoTokenizer

    from src.kv_profiler.loading import load_for_profiling, unwrap_causal_lm

    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    model = unwrap_causal_lm(load_for_profiling(name))
    n_layers, device = model.config.num_hidden_layers, next(model.parameters()).device
    wiki_ids = tok(wiki_text, return_tensors="pt").input_ids
    windows = []
    for w in range(N_WINDOWS):
        lo = w * FIXED_LEN
        if wiki_ids.shape[1] < lo + FIXED_LEN:
            break
        windows.append(_ratios(model, wiki_ids[:, lo:lo + FIXED_LEN].to(device), n_layers))
        print(f"  {key} window {w}: median {median(windows[-1]):.1f}x", flush=True)
    short = []
    if key == "qwen0.5b":
        for text in PROMPTS.values():
            ids = tok(text, return_tensors="pt").input_ids.to(device)
            if ids.shape[1] >= 8:
                short.append(_ratios(model, ids, n_layers))
    del model
    gc.collect()
    import torch
    torch.cuda.empty_cache()
    return {"model": name, "windows": windows, "short": short}


def pooled(groups, idx):
    return median(v for i in idx for v in groups[i])


def boot(effect, B=4000, seed=0):
    rng = random.Random(seed)
    vals = sorted(effect(rng) for _ in range(B))
    return vals[int(.025 * B)], vals[int(.975 * B)]


def analyse(d):
    w = {k: v["windows"] for k, v in d.items()}
    s = d["qwen0.5b"]["short"]

    def draw(rng, n):
        return [rng.randrange(n) for _ in range(n)]

    effects = {
        "length (0.5B short / 1024)": (
            pooled(s, range(len(s))) / pooled(w["qwen0.5b"], range(len(w["qwen0.5b"]))),
            lambda r: pooled(s, draw(r, len(s))) / pooled(w["qwen0.5b"], draw(r, len(w["qwen0.5b"])))),
        "architecture (Vicuna MHA / Mistral GQA)": (
            pooled(w["vicuna"], range(len(w["vicuna"]))) / pooled(w["mistral"], range(len(w["mistral"]))),
            lambda r: pooled(w["vicuna"], draw(r, len(w["vicuna"]))) / pooled(w["mistral"], draw(r, len(w["mistral"])))),
    }
    n = min(len(w["qwen7b"]), len(w["qwen0.5b"]))
    effects["scale (Qwen 7B / 0.5B)"] = (
        pooled(w["qwen7b"], range(n)) / pooled(w["qwen0.5b"], range(n)),
        lambda r: (lambda i: pooled(w["qwen7b"], i) / pooled(w["qwen0.5b"], i))(draw(r, n)))
    lines = [f"windows per model: " + ", ".join(f"{k} {len(v)}" for k, v in w.items()) + f"; short prompts {len(s)}",
             "pooled median H2O/window ratio at a 10% budget, 1024 tokens (95% CI over windows):"]
    for k in w:
        pt = pooled(w[k], range(len(w[k])))
        lo, hi = boot(lambda r, k=k: pooled(w[k], draw(r, len(w[k]))))
        lines.append(f"  {k:<10}{pt:7.1f}x  [{lo:.1f}, {hi:.1f}]")
    pt = pooled(s, range(len(s)))
    lo, hi = boot(lambda r: pooled(s, draw(r, len(s))))
    lines.append(f"  qwen0.5b short prompts {pt:.1f}x  [{lo:.1f}, {hi:.1f}]")
    lines.append("effect sizes:")
    for k, (pt, f) in effects.items():
        lo, hi = boot(f)
        lines.append(f"  {k:<42}{pt:5.2f}x  [{lo:.2f}, {hi:.2f}]")
    return lines


def run():
    wiki_text = _wikitext(min_chars=130000)
    data = {}
    for key, name in MODELS.items():
        path = f"{OUT}/claim1_windows_{key}.json"
        if os.path.exists(path):
            data[key] = json.load(open(path))
            print(f"{key}: loaded {path}", flush=True)
            continue
        print(f"{key}: measuring {name}", flush=True)
        data[key] = measure(key, name, wiki_text)
        json.dump(data[key], open(path, "w"))
    lines = analyse(data)
    print("\n".join(lines))
    stamp = time.strftime("%Y%m%d_%H%M%S")
    open(f"{OUT}/claim1_intervals_{stamp}.txt", "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    run()
