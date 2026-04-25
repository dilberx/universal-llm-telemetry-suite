"""
Experiment 1A: Prompt Complexity Routing

THE commercially valuable experiment.

Question: Can you predict which prompts need a big model vs a small one?
If yes, you can route 60-80% of traffic to a cheap model and save money.

Method:
1. Run diverse prompts through Qwen2.5-0.5B
2. Measure per-prompt: entropy, confidence, generation speed
3. Classify: which prompts does the small model handle confidently?
4. Build features that predict "small model is good enough"

Commercial value: every company running LLMs pays per-token.
A router that sends easy prompts to a small model = direct cost savings.
"""

from __future__ import annotations

import gc
import math
import time

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


@torch.no_grad()
def run_routing_experiment():
    """Test which prompts a small model handles confidently."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    runner = ExperimentRunner(
        name="prompt_routing",
        description=(
            "Measures per-prompt complexity signals on Qwen2.5-0.5B to build "
            "a routing classifier. Can we predict which prompts need a bigger model?"
        ),
    )

    model_name = "Qwen/Qwen2.5-0.5B"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    print(f"  Loaded ({torch.cuda.memory_allocated()/1024**3:.1f} GB)")

    for prompt_name, prompt_text in ROUTING_PROMPTS.items():
        # Determine expected difficulty
        if prompt_name in ["greeting", "simple_math", "yes_no", "definition",
                           "format", "list", "complete", "emoji"]:
            expected_difficulty = "easy"
        elif prompt_name in ["explain_simple", "code_easy", "summarize",
                             "rewrite", "translate_common", "classify"]:
            expected_difficulty = "medium"
        else:
            expected_difficulty = "hard"

        inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")
        input_len = inputs["input_ids"].shape[1]

        # === Feature 1: Prompt-level signals ===
        # Token count, unique ratio, etc.
        input_ids = inputs["input_ids"][0]
        unique_tokens = len(set(input_ids.tolist()))
        unique_ratio = unique_tokens / len(input_ids)

        # === Feature 2: First-pass entropy (prefill) ===
        t0 = time.perf_counter()
        outputs = model(**inputs, output_attentions=True)
        prefill_time = (time.perf_counter() - t0) * 1000

        logits = outputs.logits[0, -1].float()
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)

        first_token_entropy = -(probs * log_probs).sum().item()
        first_token_top_prob = probs.max().item()
        first_token_top5_mass = probs.topk(5).values.sum().item()
        first_token_top10_mass = probs.topk(10).values.sum().item()

        # === Feature 3: Attention pattern complexity ===
        attentions = [a.cpu().float() for a in outputs.attentions]
        avg_entropy_all = 0
        avg_sparsity_all = 0
        max_sink_mass = 0

        for attn in attentions:
            a = attn[0]  # [heads, seq, seq]
            a_clamped = a.clamp(min=1e-10)
            entropy = -(a_clamped * torch.log(a_clamped)).sum(dim=-1).mean()
            avg_entropy_all += entropy.item()

            sparsity = (a > 0.01).float().mean().item()
            avg_sparsity_all += sparsity

            if a.shape[-1] >= 4:
                sink = a[:, :, :4].sum(dim=-1).mean().item()
                max_sink_mass = max(max_sink_mass, sink)

        num_layers = len(attentions)
        avg_entropy_all /= num_layers
        avg_sparsity_all /= num_layers

        del attentions
        gc.collect()

        # === Feature 4: Generation confidence (decode 20 tokens) ===
        t0 = time.perf_counter()
        gen_out = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
        )
        decode_time = (time.perf_counter() - t0) * 1000

        gen_entropies = []
        gen_top_probs = []
        gen_agreements = []  # does top-1 == top-1 at temp=0?

        if gen_out.scores:
            for score in gen_out.scores:
                s = score[0].float()
                p = F.softmax(s, dim=-1)
                lp = F.log_softmax(s, dim=-1)
                ent = -(p * lp).sum().item()
                gen_entropies.append(ent)
                gen_top_probs.append(p.max().item())

        avg_gen_entropy = sum(gen_entropies) / max(len(gen_entropies), 1)
        avg_gen_top_prob = sum(gen_top_probs) / max(len(gen_top_probs), 1)
        min_gen_top_prob = min(gen_top_probs) if gen_top_probs else 0
        tokens_above_90 = sum(1 for p in gen_top_probs if p > 0.9) / max(len(gen_top_probs), 1)
        tokens_above_95 = sum(1 for p in gen_top_probs if p > 0.95) / max(len(gen_top_probs), 1)

        # === Feature 5: Output length and repetition ===
        output_ids = gen_out.sequences[0][input_len:].tolist()
        output_len = len(output_ids)
        output_unique = len(set(output_ids))
        output_repetition = 1 - (output_unique / max(output_len, 1))

        # === Routing signal: composite confidence ===
        # High = small model is probably fine
        # Low = probably needs a bigger model
        routing_score = (
            avg_gen_top_prob * 0.4 +
            first_token_top_prob * 0.3 +
            (1 - min(avg_gen_entropy / 5.0, 1.0)) * 0.2 +
            tokens_above_90 * 0.1
        )

        config = {
            "prompt": prompt_name,
            "expected_difficulty": expected_difficulty,
            "input_len": input_len,
            "prompt_text_len": len(prompt_text),
        }

        with runner.trial(config) as trial:
            # Prompt features
            trial.record("unique_token_ratio", unique_ratio)
            trial.record("input_token_count", input_len)

            # Prefill signals
            trial.record("prefill_time_ms", prefill_time)
            trial.record("first_token_entropy", first_token_entropy)
            trial.record("first_token_top_prob", first_token_top_prob)
            trial.record("first_token_top5_mass", first_token_top5_mass)
            trial.record("first_token_top10_mass", first_token_top10_mass)

            # Attention signals
            trial.record("avg_attention_entropy", avg_entropy_all)
            trial.record("avg_attention_sparsity", avg_sparsity_all)
            trial.record("max_sink_mass", max_sink_mass)

            # Decode signals
            trial.record("decode_time_ms", decode_time)
            trial.record("avg_gen_entropy", avg_gen_entropy)
            trial.record("avg_gen_top_prob", avg_gen_top_prob)
            trial.record("min_gen_top_prob", min_gen_top_prob)
            trial.record("tokens_above_90pct", tokens_above_90)
            trial.record("tokens_above_95pct", tokens_above_95)

            # Output signals
            trial.record("output_len", output_len)
            trial.record("output_repetition", output_repetition)

            # Composite routing score
            trial.record("routing_score", routing_score)

            # Difficulty label (for validation)
            trial.record("difficulty_label", ["easy", "medium", "hard"].index(expected_difficulty))

    del model
    gc.collect()
    torch.cuda.empty_cache()

    runner.save()
    runner.to_csv()
    print(runner.report.summary())
    return runner


if __name__ == "__main__":
    run_routing_experiment()
