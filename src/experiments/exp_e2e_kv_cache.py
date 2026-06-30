"""
End-to-end KV cache generation benchmark.

This experiment physically prunes the HuggingFace DynamicCache during greedy
generation and compares generated output against a full-cache baseline.
"""

from __future__ import annotations

import gc
import math
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

from src.experiments.exp_real_model import PROMPTS, load_model
from src.experiments.runner import ExperimentRunner


E2E_PROMPTS = {
    name: PROMPTS[name]
    for name in ("code_complex", "reasoning", "creative", "instruction", "multilingual")
}


def _select_prompts(
    prompt_set: str,
    tokenizer,
    long_context_tokens: tuple[int, ...],
) -> dict[str, str]:
    if prompt_set == "default":
        return E2E_PROMPTS
    if prompt_set != "long":
        raise ValueError(f"Unknown prompt set: {prompt_set}")

    text_path = Path(__file__).parents[1] / "wikitext-2.txt"
    raw_text = text_path.read_text(encoding="utf-8")
    clean_text = " ".join(raw_text.split())
    max_target = max(long_context_tokens)
    # Bound tokenization so long-prompt setup does not tokenize the entire
    # WikiText fixture and trigger max-length warnings before we slice.
    sample_text = clean_text[:max(20_000, max_target * 12)]
    token_ids = tokenizer(sample_text, add_special_tokens=False)["input_ids"]

    prompts: dict[str, str] = {}
    for target_len in long_context_tokens:
        if target_len >= len(token_ids):
            raise ValueError(
                f"Requested {target_len} tokens, but only {len(token_ids)} tokens are available"
            )
        prompts[f"wikitext_{target_len}tok"] = tokenizer.decode(
            token_ids[:target_len],
            skip_special_tokens=True,
        )
    return prompts


def _sync_if_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _peak_memory_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024**2)


def _cache_seq_len(cache: DynamicCache) -> int:
    return int(cache[0][0].shape[2])


def _attention_importance(attentions: tuple[torch.Tensor, ...] | list[torch.Tensor]) -> torch.Tensor:
    """Return cumulative key importance for the cache positions in attentions."""
    importance = None
    for attn in attentions:
        # [batch, heads, query, key] -> [key]
        layer_score = attn.detach().float().sum(dim=(0, 1, 2))
        importance = layer_score if importance is None else importance + layer_score
    if importance is None:
        raise ValueError("No attention tensors returned")
    return importance


def _slice_cache(cache: DynamicCache, keep_idx: torch.Tensor) -> DynamicCache:
    """Physically keep only selected time positions in every layer."""
    legacy = []
    for key, value in cache.to_legacy_cache():
        idx = keep_idx.to(key.device)
        legacy.append((
            key.index_select(2, idx).contiguous(),
            value.index_select(2, idx).contiguous(),
        ))
    return DynamicCache.from_legacy_cache(tuple(legacy))


def _keep_indices(
    policy: str,
    importance: torch.Tensor,
    budget: int,
    positions: torch.Tensor,
    sink_tokens: int = 4,
    raw_attentions: tuple[torch.Tensor, ...] | list[torch.Tensor] | None = None,
) -> torch.Tensor:
    seq_len = int(importance.numel())
    if policy == "full" or budget >= seq_len:
        return torch.arange(seq_len, device=importance.device)

    budget = max(1, min(budget, seq_len))
    if policy == "window":
        keep = torch.arange(seq_len - budget, seq_len, device=importance.device)
    elif policy == "h2o":
        keep = torch.topk(importance, k=budget).indices
    elif policy == "sink_h2o":
        keep = _merge_sink_and_ranked(positions, importance, budget, sink_tokens)
    elif policy == "hybrid":
        keep = _hybrid_keep_indices(positions, importance, budget, sink_tokens)
    elif policy == "snapkv":
        if raw_attentions is not None:
            total_var = torch.zeros(seq_len, device=importance.device)
            obs_window = 16
            for attn in raw_attentions:
                obs_len = min(attn.shape[2], obs_window)
                obs = attn.detach().float()[:, :, -obs_len:, :]
                var = obs.var(dim=1).mean(dim=(0, 1))
                if var.numel() == seq_len:
                    total_var += var.to(importance.device)
            keep = torch.topk(total_var, k=budget).indices
        else:
            keep = torch.topk(importance, k=budget).indices
    elif policy == "adaptive":
        if raw_attentions is not None:
            total_entropy = 0.0
            for attn in raw_attentions:
                mean_attn = attn[0].detach().float().mean(dim=0)
                mean_attn = mean_attn.clamp(min=1e-10)
                ent = -torch.sum(mean_attn * torch.log(mean_attn), dim=-1).mean().item()
                total_entropy += ent
            avg_entropy = total_entropy / len(raw_attentions)
            max_entropy = math.log(raw_attentions[0].shape[-1])
            normalized = avg_entropy / max_entropy if max_entropy > 0 else 0.0
            
            min_ratio = 0.25
            max_ratio = 1.0
            ratio = min_ratio + normalized * (max_ratio - min_ratio)
            adaptive_budget = max(4, int(budget * ratio))
            adaptive_budget = min(adaptive_budget, seq_len)
            keep = torch.topk(importance, k=adaptive_budget).indices
        else:
            keep = torch.topk(importance, k=budget).indices
    elif policy == "pyramid":
        keep = torch.topk(importance, k=budget).indices
    else:
        raise ValueError(f"Unknown policy: {policy}")

    return keep.sort().values


def _merge_sink_and_ranked(
    positions: torch.Tensor,
    importance: torch.Tensor,
    budget: int,
    sink_tokens: int,
) -> torch.Tensor:
    """Keep original prompt sinks, then fill the rest by attention importance."""
    device = importance.device
    sink = torch.nonzero(positions.to(device) < sink_tokens, as_tuple=False).flatten()
    if sink.numel() >= budget:
        return sink[:budget]

    keep_mask = torch.zeros(importance.numel(), dtype=torch.bool, device=device)
    keep_mask[sink] = True
    remaining = budget - int(sink.numel())
    candidate_scores = importance.masked_fill(keep_mask, float("-inf"))
    fill = torch.topk(candidate_scores, k=remaining).indices
    return torch.cat([sink, fill])


def _hybrid_keep_indices(
    positions: torch.Tensor,
    importance: torch.Tensor,
    budget: int,
    sink_tokens: int,
) -> torch.Tensor:
    """Keep sinks + a recent window + H2O heavy hitters for the remaining slots."""
    device = importance.device
    seq_len = int(importance.numel())
    keep_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)

    sink = torch.nonzero(positions.to(device) < sink_tokens, as_tuple=False).flatten()
    keep_mask[sink[:budget]] = True

    remaining = budget - int(keep_mask.sum().item())
    if remaining <= 0:
        return torch.nonzero(keep_mask, as_tuple=False).flatten()

    recent_count = min(remaining, max(1, budget // 3))
    recent = torch.arange(seq_len - recent_count, seq_len, device=device)
    keep_mask[recent] = True

    remaining = budget - int(keep_mask.sum().item())
    if remaining > 0:
        candidate_scores = importance.masked_fill(keep_mask, float("-inf"))
        fill = torch.topk(candidate_scores, k=remaining).indices
        keep_mask[fill] = True

    return torch.nonzero(keep_mask, as_tuple=False).flatten()


def _prune_state(
    policy: str,
    budget_frac: float,
    min_keep: int,
    cache: DynamicCache,
    positions: torch.Tensor,
    importance: torch.Tensor,
    total_positions_seen: int,
    raw_attentions: tuple[torch.Tensor, ...] | list[torch.Tensor] | None = None,
) -> tuple[DynamicCache, torch.Tensor, torch.Tensor]:
    if policy == "full":
        return cache, positions, importance

    seq_len = _cache_seq_len(cache)
    budget = max(min_keep, int(total_positions_seen * budget_frac))
    budget = min(budget, seq_len)

    keep_idx = _keep_indices(
        policy=policy,
        importance=importance,
        budget=budget,
        positions=positions,
        sink_tokens=4,
        raw_attentions=raw_attentions,
    )

    return (
        _slice_cache(cache, keep_idx),
        positions.index_select(0, keep_idx.to(positions.device)),
        importance.index_select(0, keep_idx.to(importance.device)),
    )


@torch.no_grad()
def generate_with_cache_policy(
    model,
    tokenizer,
    prompt: str,
    policy: str,
    budget_frac: float,
    max_new_tokens: int,
    min_keep: int = 4,
) -> dict[str, Any]:
    device = model.device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_tokens = int(inputs["input_ids"].shape[1])

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    _sync_if_cuda()
    start = time.perf_counter()

    outputs = model(
        **inputs,
        use_cache=True,
        output_attentions=True,
    )
    cache = outputs.past_key_values
    importance = _attention_importance(outputs.attentions).to(device)
    positions = torch.arange(prompt_tokens, device=device)
    next_token = outputs.logits[:, -1:].argmax(dim=-1)

    generated: list[int] = []
    cache_lengths = [_cache_seq_len(cache)]
    total_positions_seen = prompt_tokens

    for step in range(max_new_tokens):
        token_id = int(next_token.item())
        generated.append(token_id)
        if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
            break
        if step == max_new_tokens - 1:
            break

        cache, positions, importance = _prune_state(
            policy=policy,
            budget_frac=budget_frac,
            min_keep=min_keep,
            cache=cache,
            positions=positions,
            importance=importance,
            total_positions_seen=total_positions_seen,
            raw_attentions=outputs.attentions,
        )

        position_id = torch.tensor([[total_positions_seen]], device=device)
        cache_position = torch.tensor([total_positions_seen], device=device)
        outputs = model(
            input_ids=next_token,
            past_key_values=cache,
            use_cache=True,
            output_attentions=True,
            position_ids=position_id,
            cache_position=cache_position,
        )
        cache = outputs.past_key_values
        step_importance = _attention_importance(outputs.attentions).to(device)

        positions = torch.cat([
            positions,
            torch.tensor([total_positions_seen], device=device),
        ])
        importance = torch.cat([
            importance,
            torch.zeros(1, device=device, dtype=importance.dtype),
        ])
        if step_importance.numel() != importance.numel():
            raise RuntimeError(
                f"Attention/cache length mismatch: attention={step_importance.numel()} "
                f"importance={importance.numel()}"
            )
        importance = importance + step_importance

        next_token = outputs.logits[:, -1:].argmax(dim=-1)
        total_positions_seen += 1
        cache_lengths.append(_cache_seq_len(cache))

    _sync_if_cuda()
    elapsed = time.perf_counter() - start

    text = tokenizer.decode(generated, skip_special_tokens=True)
    token_count = len(generated)

    del outputs, cache
    gc.collect()

    return {
        "generated_tokens": generated,
        "generated_text": text,
        "prompt_tokens": prompt_tokens,
        "new_tokens": token_count,
        "wall_time_s": elapsed,
        "tokens_per_second": token_count / elapsed if elapsed > 0 else 0.0,
        "peak_memory_mb": _peak_memory_mb(),
        "avg_cache_tokens": sum(cache_lengths) / len(cache_lengths),
        "max_cache_tokens": max(cache_lengths),
        "final_cache_tokens": cache_lengths[-1],
    }


def _compare_to_baseline(result: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    tokens = result["generated_tokens"]
    base_tokens = baseline["generated_tokens"]
    denom = max(1, len(base_tokens))
    matches = sum(1 for a, b in zip(tokens, base_tokens) if a == b)

    first_mismatch = -1
    for idx, (a, b) in enumerate(zip(tokens, base_tokens)):
        if a != b:
            first_mismatch = idx
            break
    if first_mismatch == -1 and len(tokens) != len(base_tokens):
        first_mismatch = min(len(tokens), len(base_tokens))

    return {
        "token_match_rate": matches / denom,
        "exact_token_match": float(tokens == base_tokens),
        "first_mismatch_index": float(first_mismatch),
        "text_similarity": SequenceMatcher(
            None,
            result["generated_text"],
            baseline["generated_text"],
        ).ratio(),
    }


@torch.no_grad()
def score_fixed_targets_with_cache_policy(
    model,
    tokenizer,
    prompt: str,
    target_tokens: list[int],
    policy: str,
    budget_frac: float,
    max_target_tokens: int,
    min_keep: int = 4,
) -> dict[str, Any]:
    """Score full-cache target tokens while physically pruning the KV cache."""
    device = model.device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_tokens = int(inputs["input_ids"].shape[1])
    target_tokens = target_tokens[:max_target_tokens]

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    _sync_if_cuda()
    start = time.perf_counter()

    outputs = model(
        **inputs,
        use_cache=True,
        output_attentions=True,
    )
    cache = outputs.past_key_values
    importance = _attention_importance(outputs.attentions).to(device)
    positions = torch.arange(prompt_tokens, device=device)
    logits = outputs.logits[:, -1, :]
    cache_lengths = [_cache_seq_len(cache)]
    total_positions_seen = prompt_tokens

    logprobs: list[float] = []
    top1_matches = 0

    for idx, token_id in enumerate(target_tokens):
        token = torch.tensor([[token_id]], device=device)
        log_probs = F.log_softmax(logits.float(), dim=-1)
        logprob = float(log_probs[0, token_id].item())
        logprobs.append(logprob)
        top1_matches += int(int(logits.argmax(dim=-1).item()) == token_id)

        if idx == len(target_tokens) - 1:
            break

        cache, positions, importance = _prune_state(
            policy=policy,
            budget_frac=budget_frac,
            min_keep=min_keep,
            cache=cache,
            positions=positions,
            importance=importance,
            total_positions_seen=total_positions_seen,
            raw_attentions=outputs.attentions,
        )

        position_id = torch.tensor([[total_positions_seen]], device=device)
        cache_position = torch.tensor([total_positions_seen], device=device)
        outputs = model(
            input_ids=token,
            past_key_values=cache,
            use_cache=True,
            output_attentions=True,
            position_ids=position_id,
            cache_position=cache_position,
        )
        cache = outputs.past_key_values
        logits = outputs.logits[:, -1, :]
        step_importance = _attention_importance(outputs.attentions).to(device)

        positions = torch.cat([
            positions,
            torch.tensor([total_positions_seen], device=device),
        ])
        importance = torch.cat([
            importance,
            torch.zeros(1, device=device, dtype=importance.dtype),
        ])
        if step_importance.numel() != importance.numel():
            raise RuntimeError(
                f"Attention/cache length mismatch: attention={step_importance.numel()} "
                f"importance={importance.numel()}"
            )
        importance = importance + step_importance

        total_positions_seen += 1
        cache_lengths.append(_cache_seq_len(cache))

    _sync_if_cuda()
    elapsed = time.perf_counter() - start

    token_count = len(logprobs)
    nll = -sum(logprobs) / token_count if token_count else float("inf")
    perplexity = math.exp(min(nll, 50.0)) if token_count else float("inf")

    del outputs, cache
    gc.collect()

    return {
        "target_tokens": target_tokens,
        "prompt_tokens": prompt_tokens,
        "scored_tokens": token_count,
        "total_logprob": sum(logprobs),
        "mean_logprob": sum(logprobs) / token_count if token_count else float("-inf"),
        "nll": nll,
        "perplexity": perplexity,
        "top1_match_rate": top1_matches / token_count if token_count else 0.0,
        "wall_time_s": elapsed,
        "tokens_per_second": token_count / elapsed if elapsed > 0 else 0.0,
        "peak_memory_mb": _peak_memory_mb(),
        "avg_cache_tokens": sum(cache_lengths) / len(cache_lengths),
        "max_cache_tokens": max(cache_lengths),
        "final_cache_tokens": cache_lengths[-1],
        "token_logprobs": logprobs,
    }


def _compare_logprob_to_baseline(
    result: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, float]:
    pairs = list(zip(result["token_logprobs"], baseline["token_logprobs"]))
    avg_logprob_delta = (
        sum(score - base for score, base in pairs) / len(pairs)
        if pairs else float("-inf")
    )
    return {
        "nll_delta_vs_full": result["nll"] - baseline["nll"],
        "perplexity_ratio_vs_full": (
            result["perplexity"] / baseline["perplexity"]
            if baseline["perplexity"] > 0 else float("inf")
        ),
        "avg_logprob_delta_vs_full": avg_logprob_delta,
        "top1_match_delta_vs_full": (
            result["top1_match_rate"] - baseline["top1_match_rate"]
        ),
    }


def run_e2e_kv_cache_experiment(
    model_name: str = "Qwen/Qwen2.5-0.5B",
    max_new_tokens: int = 32,
    budgets: tuple[float, ...] = (0.3, 0.5),
    policies: tuple[str, ...] = ("window", "h2o"),
    prompt_set: str = "default",
    long_context_tokens: tuple[int, ...] = (256, 512, 768),
) -> ExperimentRunner:
    short_name = model_name.split("/")[-1].lower().replace("-", "_")
    runner = ExperimentRunner(
        name=f"e2e_kv_cache_{short_name}",
        description=(
            f"End-to-end greedy generation on {model_name} with physically pruned "
            "KV caches compared against a full-cache baseline."
        ),
    )

    model, tokenizer = load_model(model_name)
    prompts = _select_prompts(prompt_set, tokenizer, long_context_tokens)
    baselines: dict[str, dict[str, Any]] = {}

    try:
        for prompt_name, prompt_text in prompts.items():
            print(f"\n=== E2E prompt: {prompt_name} ===")
            baseline = generate_with_cache_policy(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt_text,
                policy="full",
                budget_frac=1.0,
                max_new_tokens=max_new_tokens,
            )
            baselines[prompt_name] = baseline

            with runner.trial({
                "experiment": "e2e_kv_cache",
                "model": model_name,
                "prompt": prompt_name,
                "policy": "full",
                "budget_frac": 1.0,
                "max_new_tokens": max_new_tokens,
            }) as trial:
                for key in (
                    "prompt_tokens",
                    "new_tokens",
                    "wall_time_s",
                    "tokens_per_second",
                    "peak_memory_mb",
                    "avg_cache_tokens",
                    "max_cache_tokens",
                    "final_cache_tokens",
                ):
                    trial.record(key, float(baseline[key]))
                for key, value in _compare_to_baseline(baseline, baseline).items():
                    trial.record(key, value)
                trial.record_meta("generated_text", baseline["generated_text"])

            for budget in budgets:
                for policy in policies:
                    result = generate_with_cache_policy(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=prompt_text,
                        policy=policy,
                        budget_frac=budget,
                        max_new_tokens=max_new_tokens,
                    )
                    quality = _compare_to_baseline(result, baseline)

                    with runner.trial({
                        "experiment": "e2e_kv_cache",
                        "model": model_name,
                        "prompt": prompt_name,
                        "policy": policy,
                        "budget_frac": budget,
                        "max_new_tokens": max_new_tokens,
                    }) as trial:
                        for key in (
                            "prompt_tokens",
                            "new_tokens",
                            "wall_time_s",
                            "tokens_per_second",
                            "peak_memory_mb",
                            "avg_cache_tokens",
                            "max_cache_tokens",
                            "final_cache_tokens",
                        ):
                            trial.record(key, float(result[key]))
                        for key, value in quality.items():
                            trial.record(key, value)
                        trial.record_meta("generated_text", result["generated_text"])
                        trial.record_meta("baseline_text", baseline["generated_text"])
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    runner.save()
    runner.to_csv()
    print(runner.report.summary())
    return runner


def run_e2e_kv_logprob_experiment(
    model_name: str = "Qwen/Qwen2.5-0.5B",
    max_new_tokens: int = 32,
    budgets: tuple[float, ...] = (0.75, 0.9, 0.98),
    policies: tuple[str, ...] = ("window", "h2o", "hybrid"),
    prompt_set: str = "default",
    long_context_tokens: tuple[int, ...] = (256, 512, 768),
) -> ExperimentRunner:
    short_name = model_name.split("/")[-1].lower().replace("-", "_")
    runner = ExperimentRunner(
        name=f"e2e_kv_logprob_{short_name}",
        description=(
            f"Fixed-target logprob replay on {model_name}: full-cache greedy target "
            "tokens are scored under physically pruned KV caches."
        ),
    )

    model, tokenizer = load_model(model_name)
    prompts = _select_prompts(prompt_set, tokenizer, long_context_tokens)

    try:
        for prompt_name, prompt_text in prompts.items():
            print(f"\n=== E2E logprob prompt: {prompt_name} ===")
            generated = generate_with_cache_policy(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt_text,
                policy="full",
                budget_frac=1.0,
                max_new_tokens=max_new_tokens,
            )
            target_tokens = generated["generated_tokens"]

            baseline = score_fixed_targets_with_cache_policy(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt_text,
                target_tokens=target_tokens,
                policy="full",
                budget_frac=1.0,
                max_target_tokens=max_new_tokens,
            )

            with runner.trial({
                "experiment": "e2e_kv_logprob",
                "model": model_name,
                "prompt": prompt_name,
                "policy": "full",
                "budget_frac": 1.0,
                "max_new_tokens": max_new_tokens,
            }) as trial:
                _record_logprob_metrics(trial, baseline)
                for key, value in _compare_logprob_to_baseline(baseline, baseline).items():
                    trial.record(key, value)
                trial.record_meta("target_text", generated["generated_text"])
                trial.record_meta("target_tokens", target_tokens)

            for budget in budgets:
                for policy in policies:
                    result = score_fixed_targets_with_cache_policy(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=prompt_text,
                        target_tokens=target_tokens,
                        policy=policy,
                        budget_frac=budget,
                        max_target_tokens=max_new_tokens,
                    )
                    deltas = _compare_logprob_to_baseline(result, baseline)

                    with runner.trial({
                        "experiment": "e2e_kv_logprob",
                        "model": model_name,
                        "prompt": prompt_name,
                        "policy": policy,
                        "budget_frac": budget,
                        "max_new_tokens": max_new_tokens,
                    }) as trial:
                        _record_logprob_metrics(trial, result)
                        for key, value in deltas.items():
                            trial.record(key, value)
                        trial.record_meta("target_text", generated["generated_text"])
                        trial.record_meta("target_tokens", target_tokens)
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    runner.save()
    runner.to_csv()
    print(runner.report.summary())
    return runner


def _record_logprob_metrics(trial, result: dict[str, Any]) -> None:
    for key in (
        "prompt_tokens",
        "scored_tokens",
        "total_logprob",
        "mean_logprob",
        "nll",
        "perplexity",
        "top1_match_rate",
        "wall_time_s",
        "tokens_per_second",
        "peak_memory_mb",
        "avg_cache_tokens",
        "max_cache_tokens",
        "final_cache_tokens",
    ):
        trial.record(key, float(result[key]))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run end-to-end KV cache generation benchmark")
    parser.add_argument("model", nargs="?", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--mode", choices=["greedy", "logprob"], default="greedy")
    parser.add_argument("--prompt-set", choices=["default", "long"], default="default")
    parser.add_argument("--long-context-tokens", type=int, nargs="+", default=[256, 512, 768])
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--budgets", type=float, nargs="+", default=[0.3, 0.5])
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["window", "h2o"],
        choices=["window", "h2o", "sink_h2o", "hybrid", "snapkv", "pyramid", "adaptive"],
    )
    args = parser.parse_args()

    if args.mode == "logprob":
        run_e2e_kv_logprob_experiment(
            model_name=args.model,
            max_new_tokens=args.max_new_tokens,
            budgets=tuple(args.budgets),
            policies=tuple(args.policies),
            prompt_set=args.prompt_set,
            long_context_tokens=tuple(args.long_context_tokens),
        )
    else:
        run_e2e_kv_cache_experiment(
            model_name=args.model,
            max_new_tokens=args.max_new_tokens,
            budgets=tuple(args.budgets),
            policies=tuple(args.policies),
            prompt_set=args.prompt_set,
            long_context_tokens=tuple(args.long_context_tokens),
        )
