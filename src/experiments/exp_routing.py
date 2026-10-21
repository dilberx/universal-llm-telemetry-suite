"""
Experiment 1A: Prompt Complexity Routing

Question: Can you predict which prompts need a big model vs a small one?
If yes, you can route a share of traffic to a cheap model and save money.

Design note (rewritten 2026-09-16)
----------------------------------
The previous version of this experiment was invalid. Three defects:

1. Both models loaded in fp16. Phi-2 produces non-finite logits in fp16, so
   argmax returned token 0 ("!") on every step and both models emitted
   "!!!!!!!!!!!!!!!!!!!!". Fixed by loading in bfloat16 with an explicit
   finite-logit guard that fails loudly instead of degrading silently.

2. The quality metric was tautological. On abort the code set
   routed_text = large_text and then scored SequenceMatcher(routed_text,
   large_text), which is 1.0 by construction. Every aborted trial scored
   perfect quality while measuring nothing.

3. Policy and measurement were fused. The loop broke at step K when the
   hand-set thresholds fired, so the small model's actual output was never
   observed on aborted prompts and the thresholds could not be re-examined
   without a rerun.

This version separates measurement from policy. Every prompt is generated to
completion on both models, the trajectory signals are recorded per step, and
the routing rule is applied afterwards. Thresholds can then be swept offline
against a real quality target rather than asserted up front.
"""

from __future__ import annotations

import gc
import time
from difflib import SequenceMatcher

import torch
import torch.nn.functional as F

from src.experiments.runner import ExperimentRunner


# Diverse prompts covering the spectrum from trivial to hard
ROUTING_PROMPTS = {
    # === Trivially easy (small model should handle) ===
    "greeting": "Hello! How are you today?",
    "simple_math": "What is 15 + 27?",
    "yes_no": "Is the sky blue?",
    "definition": "What is a database?",
    "format": "Convert 72°F to Celsius.",
    "list": "List 3 primary colors.",
    "complete": "The capital of France is",
    "emoji": "Respond with a thumbs up emoji.",

    # === Medium (small model might work) ===
    "explain_simple": "Explain what an API is in 2 sentences.",
    "code_easy": "Write a Python function that returns the maximum of two numbers.",
    "summarize": "Summarize this in one sentence: Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed.",
    "rewrite": "Rewrite this more formally: Hey, the meeting's moved to 3pm tomorrow.",
    "translate_common": "Translate to Spanish: Where is the nearest hospital?",
    "classify": "Is this review positive or negative? 'The food was amazing and the service was great!'",

    # === Hard (probably needs a bigger model) ===
    "reason_multi": "If all roses are flowers, and some flowers fade quickly, can we conclude that some roses fade quickly? Explain your reasoning.",
    "code_hard": "Write a Python function that finds the longest palindromic substring in a string. Include edge cases.",
    "math_word": "A train leaves Station A at 60 mph. Another train leaves Station B, 300 miles away, at 90 mph heading toward Station A. When and where do they meet?",
    "creative_long": "Write the opening paragraph of a mystery novel set in a space station orbiting Jupiter.",
    "debate": "Give arguments both for and against universal basic income.",
    "translate_rare": "Translate to Mandarin Chinese and explain the cultural nuances: 'The squeaky wheel gets the grease.'",
    "multi_step": "I have a CSV with columns: name, age, salary, department. Write Python code to find the average salary per department, then plot it as a bar chart.",
    "ambiguous": "What's the best programming language? Justify your answer considering different use cases.",
}

EASY = ["greeting", "simple_math", "yes_no", "definition", "format", "list", "complete", "emoji"]
MEDIUM = ["explain_simple", "code_easy", "summarize", "rewrite", "translate_common", "classify"]

# SEER evaluation window and the thresholds from the original design.
# Kept so the original rule can be scored, not because they are calibrated.
K = 3
TAU = 0.85
ETA = 2.0
MAX_TOKENS = 20

# Relative cost weight of a small-model token against a large-model token.
# Qwen2.5-0.5B against Phi-2 2.7B is roughly 5x on parameter count. The
# measured wall-clock ratio is recorded per run and used for the real number.
DTYPE = torch.bfloat16


def _difficulty(name: str) -> str:
    if name in MEDIUM:
        return "medium"
    if name in EASY:
        return "easy"
    return "hard"


def _assert_finite(logits: torch.Tensor, model_name: str, prompt_name: str, step: int) -> None:
    """Fail loudly on non-finite logits instead of silently emitting token 0."""
    if not torch.isfinite(logits).all():
        n_nan = int(torch.isnan(logits).sum())
        n_inf = int(torch.isinf(logits).sum())
        raise RuntimeError(
            f"Non-finite logits from {model_name} on prompt '{prompt_name}' at step {step}: "
            f"{n_nan} NaN, {n_inf} Inf. This is the fp16 failure that invalidated the "
            f"2026-07-01 run. Check the model dtype."
        )


@torch.no_grad()
def _generate_with_trajectory(model, tokenizer, prompt_text, prompt_name, model_name, max_tokens=MAX_TOKENS):
    """Greedy generation that records per-step top probability and entropy.

    Always runs to completion or EOS. The routing rule is applied afterwards,
    so the small model's real output is observed even on prompts it would fail.
    """
    inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")
    input_len = inputs["input_ids"].shape[1]
    eos_id = tokenizer.eos_token_id

    gen_tokens: list[int] = []
    probs_seq: list[float] = []
    entropy_seq: list[float] = []
    past = None
    next_input = inputs["input_ids"]

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for step in range(max_tokens):
        outputs = model(input_ids=next_input, past_key_values=past, use_cache=True)
        past = outputs.past_key_values

        logits = outputs.logits[0, -1].float()
        _assert_finite(logits, model_name, prompt_name, step)

        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        probs_seq.append(probs.max().item())
        entropy_seq.append(-(probs * log_probs).sum().item())

        next_token = int(logits.argmax().item())
        gen_tokens.append(next_token)
        if eos_id is not None and next_token == eos_id:
            break
        next_input = torch.tensor([[next_token]], device="cuda")

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    return {
        "text": tokenizer.decode(gen_tokens, skip_special_tokens=True),
        "tokens": gen_tokens,
        "probs": probs_seq,
        "entropies": entropy_seq,
        "input_len": input_len,
        "wall_time_s": elapsed,
        "n_generated": len(gen_tokens),
    }


@torch.no_grad()
def run_routing_experiment():
    """Measure SEER trajectory signals against real small-model quality."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    runner = ExperimentRunner(
        name="prompt_routing",
        description=(
            "SEER (Speculative Early-Exit Routing) on Qwen2.5-0.5B vs Phi-2 2.7B. "
            "Both models generate to completion in bfloat16; trajectory signals are "
            "recorded per step and the routing rule is scored afterwards against "
            "measured small-model quality. Replaces the invalid 2026-07-01 fp16 run."
        ),
    )

    # ---- Small model pass -------------------------------------------------
    small_model_name = "Qwen/Qwen2.5-0.5B"
    print(f"Loading small model: {small_model_name} ({DTYPE})...")
    tokenizer = AutoTokenizer.from_pretrained(small_model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        small_model_name,
        torch_dtype=DTYPE,
        device_map="cuda",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    print(f"  Loaded ({torch.cuda.memory_allocated()/1024**3:.1f} GB)")

    small_results = {}
    for prompt_name, prompt_text in ROUTING_PROMPTS.items():
        print(f"  small: {prompt_name}")
        res = _generate_with_trajectory(model, tokenizer, prompt_text, prompt_name, small_model_name)
        res["prompt_text"] = prompt_text
        small_results[prompt_name] = res

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("  Freed small model VRAM.")

    # ---- Large model pass -------------------------------------------------
    large_model_name = "microsoft/phi-2"
    print(f"\nLoading large model: {large_model_name} ({DTYPE})...")
    large_tokenizer = AutoTokenizer.from_pretrained(large_model_name, trust_remote_code=True)
    if large_tokenizer.pad_token is None:
        large_tokenizer.pad_token = large_tokenizer.eos_token

    large_model = AutoModelForCausalLM.from_pretrained(
        large_model_name,
        torch_dtype=DTYPE,
        device_map="cuda",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    large_model.eval()
    print(f"  Loaded ({torch.cuda.memory_allocated()/1024**3:.1f} GB)")

    large_results = {}
    for prompt_name, s_res in small_results.items():
        print(f"  large: {prompt_name}")
        large_results[prompt_name] = _generate_with_trajectory(
            large_model, large_tokenizer, s_res["prompt_text"], prompt_name, large_model_name
        )

    del large_model
    gc.collect()
    torch.cuda.empty_cache()
    print("  Freed large model VRAM.")

    # ---- Scoring ----------------------------------------------------------
    # Measured per-token cost ratio, used for the honest savings number.
    small_tok_s = sum(r["wall_time_s"] for r in small_results.values()) / max(
        1, sum(r["n_generated"] for r in small_results.values()))
    large_tok_s = sum(r["wall_time_s"] for r in large_results.values()) / max(
        1, sum(r["n_generated"] for r in large_results.values()))
    cost_ratio = small_tok_s / large_tok_s if large_tok_s > 0 else 1.0
    print(f"\nMeasured cost: small {small_tok_s*1000:.1f} ms/tok, "
          f"large {large_tok_s*1000:.1f} ms/tok, ratio {cost_ratio:.3f}")

    for prompt_name, s_res in small_results.items():
        l_res = large_results[prompt_name]

        # The measurement that was missing: how good is the small model here,
        # independent of any routing decision.
        small_quality = SequenceMatcher(None, s_res["text"], l_res["text"]).ratio()

        n_k = min(K, len(s_res["probs"]))
        avg_k_prob = sum(s_res["probs"][:n_k]) / n_k
        avg_k_entropy = sum(s_res["entropies"][:n_k]) / n_k

        # The original SEER rule, applied after the fact.
        would_abort = not (avg_k_prob >= TAU and avg_k_entropy <= ETA)

        # Routed quality is 1.0 on abort by definition, because escalating
        # runs the large model. Reported for completeness; small_quality and
        # the cost column are the informative pair.
        routed_quality = 1.0 if would_abort else small_quality

        # Cost relative to always calling the large model.
        n_small, n_large = s_res["n_generated"], l_res["n_generated"]
        cost_large_only = n_large
        if would_abort:
            cost_routed = n_k * cost_ratio + n_large
        else:
            cost_routed = n_small * cost_ratio
        savings = 1.0 - (cost_routed / cost_large_only) if cost_large_only else 0.0

        config = {
            "prompt": prompt_name,
            "expected_difficulty": _difficulty(prompt_name),
            "input_len": s_res["input_len"],
        }

        with runner.trial(config) as trial:
            trial.record("seer_aborted", float(would_abort))
            trial.record("small_quality", small_quality)
            trial.record("routed_quality", routed_quality)
            trial.record("cost_vs_large_only", cost_routed / cost_large_only if cost_large_only else 0.0)
            trial.record("savings_vs_large_only", savings)
            trial.record("avg_k_prob", avg_k_prob)
            trial.record("avg_k_entropy", avg_k_entropy)
            trial.record("avg_small_prob", sum(s_res["probs"]) / len(s_res["probs"]))
            trial.record("avg_small_entropy", sum(s_res["entropies"]) / len(s_res["entropies"]))
            trial.record("small_tokens", float(n_small))
            trial.record("large_tokens", float(n_large))
            trial.record("difficulty_label", ["easy", "medium", "hard"].index(_difficulty(prompt_name)))
            trial.record_meta("small_text", s_res["text"])
            trial.record_meta("large_text", l_res["text"])
            # Raw per-step signals so thresholds can be swept without a rerun.
            trial.record_meta("step_probs", s_res["probs"])
            trial.record_meta("step_entropies", s_res["entropies"])

    runner.save()
    runner.to_csv()
    print(runner.report.summary())
    return runner


if __name__ == "__main__":
    run_routing_experiment()
