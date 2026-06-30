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
    """Test the Speculative Early-Exit Routing (SEER) framework."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    runner = ExperimentRunner(
        name="prompt_routing",
        description=(
            "Measures SEER (Speculative Early-Exit Routing) on Qwen2.5-0.5B vs Phi-2 2.7B. "
            "Evaluates live trajectory stability to decide whether to abort and route."
        ),
    )

    # 1. Run generation and SEER evaluation on the small model
    small_model_name = "Qwen/Qwen2.5-0.5B"
    print(f"Loading small model: {small_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(small_model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        small_model_name,
        torch_dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    print(f"  Loaded small model ({torch.cuda.memory_allocated()/1024**3:.1f} GB)")

    small_results = {}
    K = 3  # Early-exit evaluation window
    max_tokens = 20

    for prompt_name, prompt_text in ROUTING_PROMPTS.items():
        print(f"  Small model generation: {prompt_name}")
        inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")
        input_len = inputs["input_ids"].shape[1]

        # SEER step-by-step generation loop
        gen_tokens = []
        gen_probs = []
        gen_entropies = []
        aborted = False
        curr_inputs = inputs["input_ids"]
        curr_past_key_values = None

        for step in range(max_tokens):
            if curr_past_key_values is None:
                outputs = model(input_ids=curr_inputs, use_cache=True, output_attentions=True)
            else:
                outputs = model(input_ids=next_input_id, past_key_values=curr_past_key_values, use_cache=True, output_attentions=True)

            curr_past_key_values = outputs.past_key_values
            logits = outputs.logits[0, -1].float()
            probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)

            top_prob = probs.max().item()
            entropy = -(probs * log_probs).sum().item()

            gen_probs.append(top_prob)
            gen_entropies.append(entropy)

            next_token = logits.argmax().item()
            gen_tokens.append(next_token)
            next_input_id = torch.tensor([[next_token]], device="cuda")

            # Check stability at step K
            if step == K - 1:
                avg_k_prob = sum(gen_probs) / K
                avg_k_entropy = sum(gen_entropies) / K
                # Stable generation requires high probability and low entropy
                is_stable = (avg_k_prob >= 0.85) and (avg_k_entropy <= 2.0)
                if not is_stable:
                    aborted = True
                    break

        small_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        small_results[prompt_name] = {
            "aborted": aborted,
            "tokens": gen_tokens,
            "text": small_text,
            "probs": gen_probs,
            "entropies": gen_entropies,
            "input_len": input_len,
            "prompt_text": prompt_text,
        }

    # Delete small model to free up VRAM
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("  Freed small model VRAM.")

    # 2. Run routed/comparison generation on the large model
    large_model_name = "microsoft/phi-2"
    print(f"\nLoading large model: {large_model_name}...")
    large_tokenizer = AutoTokenizer.from_pretrained(large_model_name, trust_remote_code=True)
    # Phi-2 uses different tokenizer settings, make sure pad token is set
    if large_tokenizer.pad_token is None:
        large_tokenizer.pad_token = large_tokenizer.eos_token

    large_model = AutoModelForCausalLM.from_pretrained(
        large_model_name,
        torch_dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    large_model.eval()
    print(f"  Loaded large model ({torch.cuda.memory_allocated()/1024**3:.1f} GB)")

    for prompt_name, s_res in small_results.items():
        print(f"  Large model generation/routing: {prompt_name}")
        prompt_text = s_res["prompt_text"]
        inputs = large_tokenizer(prompt_text, return_tensors="pt").to("cuda")
        input_len_l = inputs["input_ids"].shape[1]

        # Standalone large model baseline
        large_gen = large_model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
        )
        large_tokens = large_gen[0][input_len_l:].tolist()
        large_text = large_tokenizer.decode(large_tokens, skip_special_tokens=True)

        # Routed path
        if s_res["aborted"]:
            # Route: we abort small and run large model to completion (cost = K small tokens + large model run)
            routed_text = large_text
            routed_tokens = large_tokens
            tokens_saved = (max_tokens - K) / max_tokens * 0.05  # Weighted: small model is 20x cheaper
        else:
            # Local: small model completed locally (cost = small model run)
            routed_text = s_res["text"]
            routed_tokens = s_res["tokens"]
            tokens_saved = 1.0  # Saved large model execution entirely

        # Evaluate quality (comparison vs standalone large model output)
        from difflib import SequenceMatcher
        quality_score = SequenceMatcher(None, routed_text, large_text).ratio()

        expected_difficulty = "easy"
        if prompt_name in ["explain_simple", "code_easy", "summarize", "rewrite", "translate_common", "classify"]:
            expected_difficulty = "medium"
        elif prompt_name not in ["greeting", "simple_math", "yes_no", "definition", "format", "list", "complete", "emoji"]:
            expected_difficulty = "hard"

        config = {
            "prompt": prompt_name,
            "expected_difficulty": expected_difficulty,
            "input_len": s_res["input_len"],
        }

        with runner.trial(config) as trial:
            trial.record("seer_aborted", float(s_res["aborted"]))
            trial.record("tokens_saved_ratio", tokens_saved)
            trial.record("quality_retained", quality_score)
            trial.record("avg_small_prob", sum(s_res["probs"]) / len(s_res["probs"]))
            trial.record("avg_small_entropy", sum(s_res["entropies"]) / len(s_res["entropies"]))
            trial.record("difficulty_label", ["easy", "medium", "hard"].index(expected_difficulty))
            trial.record_meta("routed_text", routed_text)
            trial.record_meta("large_baseline_text", large_text)

    del large_model
    gc.collect()
    torch.cuda.empty_cache()
    print("  Freed large model VRAM.")

    runner.save()
    runner.to_csv()
    print(runner.report.summary())
    return runner


if __name__ == "__main__":
    run_routing_experiment()
