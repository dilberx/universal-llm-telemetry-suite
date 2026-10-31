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

import torch.nn.functional as F

from src.experiments.exp_e2e_kv_cache import (
    SNAPKV_WINDOW, _attention_importance, _cache_seq_len, _prune_state, _select_prompts,
    _snapkv_compress, score_fixed_targets_with_cache_policy)

ADJECTIVES = ["amber", "cobalt", "crimson", "ivory", "jade", "scarlet", "silver", "violet", "golden", "copper"]
NOUNS = ["falcon", "harbor", "lantern", "meadow", "orchard", "pillar", "quartz", "river", "summit", "willow"]
DEPTHS = (0.1, 0.3, 0.5, 0.7, 0.9)
BUDGETS = (0.1, 0.2, 0.3, 0.5)
POLICIES = ("window", "h2o", "hybrid", "snapkv", "snapkv_real")
NEEDLE = "The secret passphrase for the vault is {code}."
QUESTION = "\n\nQuestion: What is the secret passphrase for the vault?\nAnswer: The secret passphrase for the vault is"
# question-agnostic variant: the question precedes the document and the prompt ends on a bare "Answer:", so the
# last queries (SnapKV's observation window) carry neither the question nor the needle's wording
PREFIX_Q = "Question: What is the secret passphrase for the vault?\n\nDocument:\n"
SUFFIX_Q = "\n\nAnswer:"


def build_sample(tokenizer, haystack: str, depth: float, rng: random.Random, agnostic: bool = False):
    """Prompt text with the needle at `depth` and the teacher-forced answer tokens, or None if tokenisation splits."""
    code = f"{rng.choice(ADJECTIVES)}-{rng.choice(NOUNS)}-{rng.randrange(1000, 10000)}"
    ids = tokenizer(haystack)["input_ids"]
    cut = int(depth * len(ids))
    text = (tokenizer.decode(ids[:cut]) + "\n" + NEEDLE.format(code=code) + "\n"
            + tokenizer.decode(ids[cut:]))
    text = PREFIX_Q + text + SUFFIX_Q if agnostic else text + QUESTION
    prompt_ids = tokenizer(text)["input_ids"]
    full_ids = tokenizer(text + " " + code)["input_ids"]
    if full_ids[:len(prompt_ids)] != prompt_ids:          # answer merged into the prompt's last token
        return None
    return {"prompt": text, "code": code, "targets": full_ids[len(prompt_ids):], "prompt_tokens": len(prompt_ids)}


def boundary_split(tokenizer, sample):
    """Token ids of the document, the question tail, or None if the document's last token merges with the question."""
    doc_text = sample["prompt"][:-len(QUESTION)]
    full = tokenizer(sample["prompt"])["input_ids"]
    doc = tokenizer(doc_text)["input_ids"]
    if full[:len(doc)] != doc:
        return None
    return {"doc": doc, "question": full[len(doc):]}


def _scoring_cache_lengths(n, n_feed, n_targets, frac, min_keep):
    """Cache length each target is scored at for a per-step policy (mirrors `score_split`)."""
    cache, seen, lens = n, n, []
    if n_feed == 0:
        cache = min(cache, max(min_keep, int(seen * frac)))
    for k in range(n_feed + n_targets):
        if k >= n_feed:
            lens.append(cache)
        if k == n_feed + n_targets - 1:
            break
        cache = min(cache, max(min_keep, int(seen * frac))) + 1
        seen += 1
    return lens


def _matched_snapkv_budget(n, n_feed, n_targets, frac, min_keep):
    """Prefill budget that gives SnapKV the same mean cache over the scored targets as the per-step policies."""
    lens = _scoring_cache_lengths(n, n_feed, n_targets, frac, min_keep)
    return max(2, round(sum(lens) / len(lens) - n_feed - (n_targets - 1) / 2))


@torch.no_grad()
def score_split(model, ids_prefill, ids_feed, targets, policy, budget_frac, min_keep=4):
    """Teacher-forced target scoring where `ids_prefill` is prefilled and `ids_feed` is appended token by token.

    With ids_feed empty this is the question-in-prefill replay with every policy pruned before the first
    target is scored. With the question as ids_feed, the prefill, SnapKV's observation window and the
    accumulated importance see only the document: the pruned cache is built before the question arrives.
    """
    device = model.device
    n, n_feed = len(ids_prefill), len(ids_feed)
    ids = torch.tensor([ids_prefill], device=device)
    out = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=True, output_attentions=True)
    cache = out.past_key_values
    importance = _attention_importance(out.attentions).to(device)
    positions = torch.arange(n, device=device)
    logits = out.logits[:, -1, :]
    seen = n
    if policy == "snapkv_real":
        cache = _snapkv_compress(cache, out.attentions,
                                 _matched_snapkv_budget(n, n_feed, len(targets), budget_frac, min_keep))
    fed = list(ids_feed) + list(targets[:-1])
    logprobs, top1, lens = [], 0, []
    for k in range(len(fed) + 1):
        if k == 0 and n_feed == 0 and policy != "snapkv_real":
            cache, positions, importance = _prune_state(policy, budget_frac, min_keep, cache, positions,
                                                        importance, seen, raw_attentions=out.attentions)
        if k >= n_feed:
            t = targets[k - n_feed]
            logprobs.append(float(F.log_softmax(logits.float(), dim=-1)[0, t].item()))
            top1 += int(int(logits.argmax(dim=-1).item()) == t)
            lens.append(_cache_seq_len(cache))
        if k == len(fed):
            break
        if policy != "snapkv_real":
            cache, positions, importance = _prune_state(policy, budget_frac, min_keep, cache, positions,
                                                        importance, seen, raw_attentions=out.attentions)
        out = model(input_ids=torch.tensor([[fed[k]]], device=device), past_key_values=cache, use_cache=True,
                    output_attentions=True, position_ids=torch.tensor([[seen]], device=device),
                    cache_position=torch.tensor([seen], device=device))
        cache, logits = out.past_key_values, out.logits[:, -1, :]
        if policy != "snapkv_real":
            positions = torch.cat([positions, torch.tensor([seen], device=device)])
            importance = torch.cat([importance, torch.zeros(1, device=device, dtype=importance.dtype)])
            importance = importance + _attention_importance(out.attentions).to(device)
        seen += 1
    nll = -sum(logprobs) / len(logprobs)
    del out, cache
    return {"prompt_tokens": n + n_feed, "scored_tokens": len(logprobs), "top1_match_rate": top1 / len(logprobs),
            "nll": nll, "avg_cache_tokens": sum(lens) / len(lens)}


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
        budgets=BUDGETS, policies=POLICIES, depths=DEPTHS, out_dir="reports/experiments", seed=0, agnostic=False, mode="replay"):
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
    short = model_name.rstrip("/").split("/")[-1].lower() + ("_agnostic" if agnostic else "") + ("" if mode == "replay" else "_" + mode)
    for name, text in haystacks.items():
        w = int(name.rsplit("_w", 1)[1])
        for depth in depths:
            sample = build_sample(tokenizer, text, depth, rng, agnostic)
            if sample is None:
                print(f"  window {w} depth {depth}: tokenisation merged the answer, skipped", flush=True)
                continue
            split = boundary_split(tokenizer, sample) if mode != "replay" else None
            if mode != "replay" and split is None:
                print(f"  window {w} depth {depth}: document end merged with the question, skipped", flush=True)
                continue
            for pol, bud in [("full", 1.0)] + [(p, b) for b in budgets for p in policies]:
                if mode == "replay":
                    r = score_fixed_targets_with_cache_policy(
                        model, tokenizer, sample["prompt"], sample["targets"], pol, bud, len(sample["targets"]))
                elif mode == "prefill_q":
                    r = score_split(model, split["doc"] + split["question"], [], sample["targets"], pol, bud)
                else:
                    r = score_split(model, split["doc"], split["question"], sample["targets"], pol, bud)
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
    ap.add_argument("--agnostic", action="store_true", help="question first, prompt ends on a bare 'Answer:'")
    ap.add_argument("--mode", choices=("replay", "prefill_q", "boundary"), default="replay",
                    help="replay: original scoring; prefill_q: question prefilled, every policy pruned before the first "
                         "target; boundary: prune on the document alone, then append the question")
    a = ap.parse_args()
    run(a.model, a.length, a.windows, a.offload, a.headroom, agnostic=a.agnostic, mode=a.mode)
