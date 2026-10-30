"""Does the quality ranking from NLL replay survive a downstream task? Needle retrieval under KV eviction.

The NLL audit scores the full-cache greedy continuation. This asks something a user cares about: can the
model still retrieve one fact after the cache is pruned. A secret passphrase (two words and four digits,
drawn at random so it cannot be guessed) is inserted at a chosen depth into a wikitext window. The prompt
ends with the question and the start of the answer. The answer tokens are teacher-forced through the
replay's own `score_fixed_targets_with_cache_policy`, so every policy sees exactly what it sees in the NLL
audit: a full prefill, then pruning at each decode step (SnapKV compresses once at prefill).

Metrics per (policy, budget):
  exact   all answer tokens are the argmax under the pruned cache (greedy decoding would print the passphrase)
  nll     mean negative log-likelihood of the answer tokens

Samples are depth x window. Intervals resample windows (each window carries one needle per depth).
Prefill uses full attention with output_attentions, so context length is limited by the attention tensor
(layers x heads x n^2): about 2K tokens on the 0.5B and about 1.5K on a 7B with offload. Run:
    python -m src.experiments.exp_needle_task --model Qwen/Qwen2.5-0.5B --length 2048 --windows 8
"""

from __future__ import annotations

import argparse
import json
import random
import time

import torch

from src.experiments.exp_e2e_kv_cache import _select_prompts, score_fixed_targets_with_cache_policy

ADJECTIVES = ["amber", "cobalt", "crimson", "ivory", "jade", "scarlet", "silver", "violet", "golden", "copper"]
NOUNS = ["falcon", "harbor", "lantern", "meadow", "orchard", "pillar", "quartz", "river", "summit", "willow"]
DEPTHS = (0.1, 0.3, 0.5, 0.7, 0.9)
BUDGETS = (0.1, 0.2, 0.3, 0.5)
POLICIES = ("window", "h2o", "hybrid", "snapkv", "snapkv_real")
NEEDLE = "The secret passphrase for the vault is {code}."
QUESTION = "\n\nQuestion: What is the secret passphrase for the vault?\nAnswer: The secret passphrase for the vault is"


def build_sample(tokenizer, haystack: str, depth: float, rng: random.Random):
    """Prompt text with the needle at `depth` and the teacher-forced answer tokens, or None if tokenisation splits."""
    code = f"{rng.choice(ADJECTIVES)}-{rng.choice(NOUNS)}-{rng.randrange(1000, 10000)}"
    ids = tokenizer(haystack)["input_ids"]
    cut = int(depth * len(ids))
    text = (tokenizer.decode(ids[:cut]) + "\n" + NEEDLE.format(code=code) + "\n"
            + tokenizer.decode(ids[cut:]) + QUESTION)
    prompt_ids = tokenizer(text)["input_ids"]
    full_ids = tokenizer(text + " " + code)["input_ids"]
    if full_ids[:len(prompt_ids)] != prompt_ids:          # answer merged into the prompt's last token
        return None
    return {"prompt": text, "code": code, "targets": full_ids[len(prompt_ids):], "prompt_tokens": len(prompt_ids)}


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5
    return ((c - h) / d, (c + h) / d)


def summarise(records: list[dict], windows: list[int], seed: int = 0, B: int = 2000) -> list[str]:
    rng = random.Random(seed)
    lines = []
    full_ok = {(r["window"], r["depth"]) for r in records if r["policy"] == "full" and r["exact"]}
    n_full = sum(1 for r in records if r["policy"] == "full")
    lines.append(f"full cache: {len(full_ok)} of {n_full} needles retrieved exactly")
    cells = sorted({(r["policy"], r["budget"]) for r in records if r["policy"] != "full"})

    def stat(rs, ws, key):
        pick = [r for w in ws for r in rs if r["window"] == w]
        return sum(r[key] for r in pick) / len(pick) if pick else float("nan")

    lines.append(f"{'policy':<12}{'budget':>7}{'exact':>8}   95% CI over windows    {'exact | full ok':>16}{'nll':>8}")
    for pol, bud in cells:
        rs = [r for r in records if r["policy"] == pol and r["budget"] == bud]
        point = stat(rs, windows, "exact")
        boots = sorted(stat(rs, [rng.choice(windows) for _ in windows], "exact") for _ in range(B))
        cond = [r for r in rs if (r["window"], r["depth"]) in full_ok]
        cond_acc = sum(r["exact"] for r in cond) / len(cond) if cond else float("nan")
        lines.append(f"{pol:<12}{bud:>7.0%}{point:>8.3f}   [{boots[int(.025 * B)]:.3f}, {boots[int(.975 * B)]:.3f}]"
                     f"{cond_acc:>18.3f}{stat(rs, windows, 'nll'):>8.2f}")
    return lines


def run(model_name="Qwen/Qwen2.5-0.5B", length=2048, n_windows=8, offload=False, headroom=3.5,
        budgets=BUDGETS, policies=POLICIES, depths=DEPTHS, out_dir="reports/experiments", seed=0):
    if offload:
        from transformers import AutoTokenizer
        from src.kv_profiler.loading import limit_gpu_memory, load_for_profiling
        limit_gpu_memory()
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = load_for_profiling(model_name, dtype=torch.bfloat16, headroom_gib=headroom)
    else:
        from src.experiments.exp_real_model import load_model
        model, tokenizer = load_model(model_name)

    hay_tokens = length - 80                                  # room for the needle and the question
    haystacks = _select_prompts("long", tokenizer, (hay_tokens,), long_windows=n_windows)
    rng = random.Random(seed)
    records, t0 = [], time.time()
    short = model_name.rstrip("/").split("/")[-1].lower()
    for name, text in haystacks.items():
        w = int(name.rsplit("_w", 1)[1])
        for depth in depths:
            sample = build_sample(tokenizer, text, depth, rng)
            if sample is None:
                print(f"  window {w} depth {depth}: tokenisation merged the answer, skipped", flush=True)
                continue
            for pol, bud in [("full", 1.0)] + [(p, b) for b in budgets for p in policies]:
                r = score_fixed_targets_with_cache_policy(
                    model, tokenizer, sample["prompt"], sample["targets"], pol, bud, len(sample["targets"]))
                records.append({"window": w, "depth": depth, "policy": pol, "budget": bud, "code": sample["code"],
                                "prompt_tokens": r["prompt_tokens"], "n_answer_tokens": r["scored_tokens"],
                                "exact": int(r["top1_match_rate"] == 1.0), "nll": r["nll"],
                                "avg_cache_tokens": r["avg_cache_tokens"]})
            json.dump({"model": model_name, "length": length, "partial": True, "records": records},
                      open(f"{out_dir}/needle_{short}_{length}tok_partial.json", "w"))
            done = [x for x in records if x["window"] == w and x["depth"] == depth]
            print(f"  window {w} depth {depth}: {sample['prompt_tokens']} tokens, "
                  f"full exact={done[0]['exact']}, {time.time() - t0:.0f}s", flush=True)
    windows = sorted({r["window"] for r in records})
    lines = summarise(records, windows)
    print("\n".join(lines))
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = f"{out_dir}/needle_{short}_{length}tok_{stamp}"
    json.dump({"model": model_name, "length": length, "windows": windows, "depths": list(depths),
               "records": records}, open(path + ".json", "w"), indent=1)
    open(path + ".txt", "w").write("\n".join(lines) + "\n")
    print(f"saved {path}.json")
    return records


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--length", type=int, default=2048)
    ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--offload", action="store_true")
    ap.add_argument("--headroom", type=float, default=3.5)
    a = ap.parse_args()
    run(a.model, a.length, a.windows, a.offload, a.headroom)
