"""
Experiment 6B: Attention Entropy as Quality Predictor

Can you tell if a model's answer is good just by looking at 
attention entropy — without running any evaluation?

If yes, this is free quality monitoring for production systems.
No need for separate eval pipeline.

Method:
1. Run prompts through Qwen2.5-0.5B
2. For each response, compute attention entropy during generation
3. Compare entropy patterns between "confident" and "uncertain" outputs
4. Check: does low entropy actually predict higher quality?
"""

from __future__ import annotations

import gc
import math
import time

import torch
import torch.nn.functional as F

from src.experiments.runner import ExperimentRunner


QUALITY_PROMPTS = {
    # Prompts with objectively verifiable answers (we can check quality)
    "math_1": {"text": "What is 7 * 8?", "answer": "56", "type": "factual"},
    "math_2": {"text": "What is 144 / 12?", "answer": "12", "type": "factual"},
    "math_3": {"text": "What is 23 + 89?", "answer": "112", "type": "factual"},
    "math_4": {"text": "What is 15 * 15?", "answer": "225", "type": "factual"},
    "math_5": {"text": "What is the square root of 64?", "answer": "8", "type": "factual"},
    "fact_1": {"text": "What is the chemical symbol for water?", "answer": "H2O", "type": "factual"},
    "fact_2": {"text": "How many continents are there?", "answer": "7", "type": "factual"},
    "fact_3": {"text": "What planet is closest to the Sun?", "answer": "Mercury", "type": "factual"},
    "fact_4": {"text": "What is the speed of light in m/s approximately?", "answer": "300000000", "type": "factual"},
    "fact_5": {"text": "What year did World War 2 end?", "answer": "1945", "type": "factual"},
    "code_1": {"text": "In Python, what does len([1,2,3]) return?", "answer": "3", "type": "factual"},
    "code_2": {"text": "In Python, what is type(3.14)?", "answer": "float", "type": "factual"},

    # Prompts where quality is harder to judge (open-ended)
    "explain_1": {"text": "Explain recursion in one sentence.", "answer": None, "type": "open"},
    "explain_2": {"text": "What is the difference between a stack and a queue?", "answer": None, "type": "open"},
    "creative_1": {"text": "Write a haiku about programming.", "answer": None, "type": "open"},
    "creative_2": {"text": "Describe the taste of coffee to someone who has never tried it.", "answer": None, "type": "open"},
    "reason_1": {"text": "If I have 3 apples and give away 1, then buy 4 more, how many do I have?", "answer": "6", "type": "factual"},
    "reason_2": {"text": "A bat and ball cost $1.10 total. The bat costs $1 more than the ball. What does the ball cost?", "answer": "0.05", "type": "factual"},
    "ambig_1": {"text": "Is Python better than JavaScript?", "answer": None, "type": "open"},
    "ambig_2": {"text": "What is consciousness?", "answer": None, "type": "open"},
}


@torch.no_grad()
def run_entropy_quality_experiment():
    """Test if attention entropy predicts output quality."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    runner = ExperimentRunner(
        name="entropy_quality_predictor",
        description=(
            "Tests whether attention entropy during generation predicts "
            "output quality. If it does, you get free quality monitoring "
            "without running evaluations."
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

    for prompt_name, prompt_info in QUALITY_PROMPTS.items():
        prompt_text = prompt_info["text"]
        expected_answer = prompt_info["answer"]
        prompt_type = prompt_info["type"]

        inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")
        input_len = inputs["input_ids"].shape[1]

        # Generate with attention outputs
        gen_out = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,
            output_scores=True,
            output_attentions=True,
            return_dict_in_generate=True,
        )

        generated_text = tokenizer.decode(
            gen_out.sequences[0][input_len:], skip_special_tokens=True
        ).strip()

        # === Measure output quality ===
        if expected_answer is not None:
            # Check if the expected answer appears in the output
            answer_correct = expected_answer.lower() in generated_text.lower()
            # Also check for numeric equivalence
            try:
                gen_nums = [float(w) for w in generated_text.replace(",", "").split()
                           if w.replace(".", "").replace("-", "").isdigit()]
                exp_num = float(expected_answer)
                if any(abs(n - exp_num) < 0.01 for n in gen_nums):
                    answer_correct = True
            except (ValueError, IndexError):
                pass
            quality_score = 1.0 if answer_correct else 0.0
        else:
            # Open-ended: use output length and coherence as proxy
            quality_score = -1.0  # mark as unscored

        # === Measure entropy signals per generated token ===
        token_entropies = []
        token_top_probs = []

        if gen_out.scores:
            for score in gen_out.scores:
                s = score[0].float()
                p = F.softmax(s, dim=-1)
                lp = F.log_softmax(s, dim=-1)
                ent = -(p * lp).sum().item()
                token_entropies.append(ent)
                token_top_probs.append(p.max().item())

        # === Attention entropy during prefill ===
        prefill_out = model(**inputs, output_attentions=True)
        prefill_attentions = [a.cpu().float() for a in prefill_out.attentions]

        layer_entropies = []
        layer_sparsities = []
        for attn in prefill_attentions:
            a = attn[0].clamp(min=1e-10)
            ent = -(a * torch.log(a)).sum(dim=-1).mean().item()
            layer_entropies.append(ent)
            sparse = (a > 0.01).float().mean().item()
            layer_sparsities.append(sparse)

        del prefill_attentions
        gc.collect()

        # === Entropy statistics ===
        avg_token_entropy = sum(token_entropies) / max(len(token_entropies), 1)
        max_token_entropy = max(token_entropies) if token_entropies else 0
        min_token_entropy = min(token_entropies) if token_entropies else 0
        entropy_variance = (
            sum((e - avg_token_entropy)**2 for e in token_entropies) /
            max(len(token_entropies) - 1, 1)
        ) if len(token_entropies) > 1 else 0

        avg_attn_entropy = sum(layer_entropies) / max(len(layer_entropies), 1)
        avg_attn_sparsity = sum(layer_sparsities) / max(len(layer_sparsities), 1)

        # === Entropy trajectory: does entropy increase or decrease during generation? ===
        if len(token_entropies) >= 4:
            first_half = sum(token_entropies[:len(token_entropies)//2]) / (len(token_entropies)//2)
            second_half = sum(token_entropies[len(token_entropies)//2:]) / (len(token_entropies) - len(token_entropies)//2)
            entropy_drift = second_half - first_half  # positive = getting less certain
        else:
            entropy_drift = 0

        config = {
            "prompt": prompt_name,
            "prompt_type": prompt_type,
            "input_len": input_len,
            "has_ground_truth": expected_answer is not None,
        }

        with runner.trial(config) as trial:
            # Quality
            trial.record("quality_score", quality_score)
            trial.record("output_len", len(generated_text.split()))

            # Token-level entropy
            trial.record("avg_token_entropy", avg_token_entropy)
            trial.record("max_token_entropy", max_token_entropy)
            trial.record("min_token_entropy", min_token_entropy)
            trial.record("entropy_variance", entropy_variance)
            trial.record("entropy_drift", entropy_drift)

            # Token confidence
            trial.record("avg_top_prob", sum(token_top_probs) / max(len(token_top_probs), 1))
            trial.record("min_top_prob", min(token_top_probs) if token_top_probs else 0)

            # Attention-level entropy
            trial.record("avg_attention_entropy", avg_attn_entropy)
            trial.record("avg_attention_sparsity", avg_attn_sparsity)

            # Composite "confidence signal"
            confidence_signal = (
                (sum(token_top_probs) / max(len(token_top_probs), 1)) * 0.5 +
                (1 - min(avg_token_entropy / 5.0, 1.0)) * 0.3 +
                avg_attn_sparsity * 0.2
            )
            trial.record("confidence_signal", confidence_signal)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    runner.save()
    runner.to_csv()
    print(runner.report.summary())
    return runner


if __name__ == "__main__":
    run_entropy_quality_experiment()
