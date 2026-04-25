"""
Real-Model Inference Optimization Experiments

This is where novelty lives. Instead of synthetic data, we:
1. Load a real model (Qwen2.5-0.5B)
2. Extract real attention patterns from diverse prompts
3. Measure real quantization sensitivity on actual weights
4. Stack optimizations and find unexpected interactions

The goal: find something surprising that nobody has published.
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from src.experiments.runner import ExperimentRunner


# ── Model Loading ────────────────────────────────────────────────────

def load_model(model_name: str = "Qwen/Qwen2.5-0.5B", device: str = "cuda"):
    """Load model with attention output enabled."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
        attn_implementation="eager",  # Need eager for attention weights
    )
    model.eval()
    print(f"  ✓ Loaded on {device} ({torch.cuda.memory_allocated()/1024**3:.1f} GB)")
    return model, tokenizer


# ── Diverse Test Prompts ─────────────────────────────────────────────

PROMPTS = {
    "code_simple": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n\n# Call the function\nresult =",
    "code_complex": "import torch\nimport torch.nn as nn\n\nclass MultiHeadAttention(nn.Module):\n    def __init__(self, d_model, num_heads):\n        super().__init__()\n        self.d_model = d_model\n        self.num_heads = num_heads\n        self.d_k = d_model // num_heads\n        self.W_q = nn.Linear(d_model, d_model)\n        self.W_k = nn.Linear(d_model, d_model)\n        self.W_v = nn.Linear(d_model, d_model)\n        self.W_o = nn.Linear(d_model, d_model)\n\n    def forward(self, x):",
    "reasoning": "Let me solve this step by step. If a train travels at 60 mph for 2 hours, then at 80 mph for 3 hours, what is the total distance? First, distance = speed × time. For the first part:",
    "creative": "The ancient library held secrets that no living soul had ever uncovered. As Professor Elena pushed open the heavy oak door, the smell of centuries-old parchment filled her nostrils. In the dim candlelight, she noticed",
    "factual": "The transformer architecture was introduced in the paper 'Attention Is All You Need' by Vaswani et al. in 2017. The key innovation was the self-attention mechanism, which allows the model to",
    "short": "What is 2+2?",
    "instruction": "You are a helpful assistant. Please explain the concept of gradient descent in machine learning, including the learning rate and how it affects convergence.",
    "multilingual": "Translate the following English text to French: 'The quick brown fox jumps over the lazy dog.' Translation:",
}


@torch.no_grad()
def extract_attention_patterns(model, tokenizer, prompt: str, max_new_tokens: int = 50):
    """
    Run a real forward pass and extract attention patterns.

    Returns dict with attention weights, logits, and generation info.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    # Forward pass with attention output
    outputs = model(
        **inputs,
        output_attentions=True,
        use_cache=True,
    )

    # Extract attention weights: list of [batch, num_heads, seq_len, seq_len]
    attentions = [a.cpu().float() for a in outputs.attentions]

    # Get logits for token confidence analysis
    logits = outputs.logits[0, -1].cpu().float()  # Last token logits

    # Generate tokens for confidence tracking
    gen_logits = []
    gen_tokens = []
    past_kv = outputs.past_key_values
    next_input = outputs.logits[:, -1:].argmax(dim=-1)

    for step in range(min(max_new_tokens, 30)):
        step_out = model(
            input_ids=next_input,
            past_key_values=past_kv,
            use_cache=True,
        )
        step_logits = step_out.logits[0, -1].cpu().float()
        gen_logits.append(step_logits)

        next_token = step_logits.argmax()
        gen_tokens.append(next_token.item())
        next_input = next_token.unsqueeze(0).unsqueeze(0).to(model.device)
        past_kv = step_out.past_key_values

        # Stop on EOS
        if next_token.item() == tokenizer.eos_token_id:
            break

    return {
        "attentions": attentions,  # list of [1, num_heads, seq, seq]
        "logits": logits,
        "gen_logits": gen_logits,
        "gen_tokens": gen_tokens,
        "input_len": input_len,
        "prompt_type": None,  # Set by caller
    }


# ── Real Attention Head Analysis ─────────────────────────────────────

def analyze_real_heads(attentions: list[torch.Tensor]) -> dict:
    """Analyze real attention head patterns — find which heads matter."""
    num_layers = len(attentions)
    num_heads = attentions[0].shape[1]

    head_entropy = torch.zeros(num_layers, num_heads)
    head_sparsity = torch.zeros(num_layers, num_heads)
    head_sink_mass = torch.zeros(num_layers, num_heads)  # Mass on first 4 tokens

    for layer_idx, attn in enumerate(attentions):
        attn = attn[0]  # Remove batch dim: [num_heads, seq, seq]
        for head_idx in range(num_heads):
            a = attn[head_idx]  # [seq, seq]

            # Entropy (how focused is this head?)
            a_clamped = a.clamp(min=1e-10)
            entropy = -(a_clamped * torch.log(a_clamped)).sum(dim=-1).mean()
            head_entropy[layer_idx, head_idx] = entropy

            # Sparsity (what fraction of positions get >1% attention?)
            sparse_frac = (a > 0.01).float().mean()
            head_sparsity[layer_idx, head_idx] = sparse_frac

            # Attention sink mass (first 4 positions)
            if a.shape[-1] >= 4:
                sink = a[:, :4].sum(dim=-1).mean()
                head_sink_mass[layer_idx, head_idx] = sink

    return {
        "head_entropy": head_entropy,
        "head_sparsity": head_sparsity,
        "head_sink_mass": head_sink_mass,
        "num_layers": num_layers,
        "num_heads": num_heads,
    }


# ── Real Token Confidence ────────────────────────────────────────────

def analyze_real_confidence(gen_logits: list[torch.Tensor]) -> dict:
    """Analyze token confidence from real generation."""
    if not gen_logits:
        return {}

    top_probs = []
    entropies = []
    top_k_masses = []  # Mass in top-5

    for logits in gen_logits:
        probs = F.softmax(logits, dim=-1)
        top_prob = probs.max().item()
        top_probs.append(top_prob)

        # Entropy
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum().item()
        entropies.append(entropy)

        # Top-5 mass
        top5 = probs.topk(5).values.sum().item()
        top_k_masses.append(top5)

    return {
        "avg_top_prob": sum(top_probs) / len(top_probs),
        "min_top_prob": min(top_probs),
        "max_top_prob": max(top_probs),
        "skip_rate_50": sum(1 for p in top_probs if p > 0.5) / len(top_probs),
        "skip_rate_80": sum(1 for p in top_probs if p > 0.8) / len(top_probs),
        "skip_rate_90": sum(1 for p in top_probs if p > 0.9) / len(top_probs),
        "skip_rate_95": sum(1 for p in top_probs if p > 0.95) / len(top_probs),
        "skip_rate_99": sum(1 for p in top_probs if p > 0.99) / len(top_probs),
        "avg_entropy": sum(entropies) / len(entropies),
        "avg_top5_mass": sum(top_k_masses) / len(top_k_masses),
        "num_tokens": len(top_probs),
    }


# ── Real KV Cache Analysis ───────────────────────────────────────────

def analyze_real_kv_cache(attentions: list[torch.Tensor], budgets: list[float]) -> list[dict]:
    """
    Apply KV cache eviction policies to REAL attention patterns.
    This is where we might find that real patterns differ from synthetic.
    """
    results = []
    num_layers = len(attentions)

    for budget_frac in budgets:
        for layer_idx in range(num_layers):
            attn = attentions[layer_idx][0]  # [heads, seq, seq]
            num_heads, seq_len, _ = attn.shape
            budget = max(1, int(seq_len * budget_frac))

            # Policy 1: Window (keep last `budget` tokens)
            window_mask = torch.zeros(seq_len, dtype=torch.bool)
            window_mask[-budget:] = True
            window_quality = attn[:, :, window_mask].sum().item() / attn.sum().item()

            # Policy 2: H2O (keep tokens with highest cumulative attention)
            cumulative = attn.sum(dim=(0, 1))  # [seq_len] — total attention received
            _, top_indices = cumulative.topk(budget)
            h2o_mask = torch.zeros(seq_len, dtype=torch.bool)
            h2o_mask[top_indices] = True
            h2o_quality = attn[:, :, h2o_mask].sum().item() / attn.sum().item()

            # Policy 3: Sink + Recent (keep first 4 + last N)
            sink_mask = torch.zeros(seq_len, dtype=torch.bool)
            sink_count = min(4, budget // 2)
            recent_count = budget - sink_count
            sink_mask[:sink_count] = True
            sink_mask[-recent_count:] = True
            sink_quality = attn[:, :, sink_mask].sum().item() / attn.sum().item()

            results.append({
                "layer": layer_idx,
                "budget_frac": budget_frac,
                "budget_tokens": budget,
                "seq_len": seq_len,
                "window_quality": window_quality,
                "h2o_quality": h2o_quality,
                "sink_quality": sink_quality,
                "h2o_vs_window": h2o_quality / max(window_quality, 1e-10),
                "memory_saved_pct": (1 - budget_frac) * 100,
            })

    return results


# ── Main Experiment ──────────────────────────────────────────────────

def run_real_model_experiment():
    """Run all experiments on a real model."""
    runner = ExperimentRunner(
        name="real_model_analysis",
        description=(
            "Comprehensive analysis of Qwen2.5-0.5B with REAL attention "
            "patterns, REAL logit distributions, and REAL KV cache behavior. "
            "Hunting for findings that differ from synthetic experiments."
        ),
    )

    model, tokenizer = load_model("Qwen/Qwen2.5-0.5B")

    budgets = [0.1, 0.2, 0.3, 0.5, 0.75]

    for prompt_name, prompt_text in PROMPTS.items():
        print(f"\n{'='*50}")
        print(f"Prompt: {prompt_name}")
        print(f"{'='*50}")

        # Extract real attention patterns
        data = extract_attention_patterns(model, tokenizer, prompt_text)
        data["prompt_type"] = prompt_name

        # 1. Real head analysis
        head_info = analyze_real_heads(data["attentions"])

        # 2. Real token confidence
        conf_info = analyze_real_confidence(data["gen_logits"])

        # 3. Real KV cache policy comparison
        kv_results = analyze_real_kv_cache(data["attentions"], budgets)

        # Record head analysis trial
        config = {
            "experiment": "head_analysis",
            "prompt": prompt_name,
            "input_len": data["input_len"],
            "num_layers": head_info["num_layers"],
            "num_heads": head_info["num_heads"],
        }
        with runner.trial(config) as trial:
            trial.record("entropy_mean", head_info["head_entropy"].mean().item())
            trial.record("entropy_std", head_info["head_entropy"].std().item())
            trial.record("entropy_min", head_info["head_entropy"].min().item())
            trial.record("entropy_max", head_info["head_entropy"].max().item())
            trial.record("sparsity_mean", head_info["head_sparsity"].mean().item())
            trial.record("sink_mass_mean", head_info["head_sink_mass"].mean().item())
            trial.record("sink_mass_max", head_info["head_sink_mass"].max().item())

            # Find most/least important heads
            flat_entropy = head_info["head_entropy"].reshape(-1)
            trial.record("most_focused_head_entropy", flat_entropy.min().item())
            trial.record("most_diffuse_head_entropy", flat_entropy.max().item())
            trial.record("focused_head_count", (flat_entropy < flat_entropy.median()).sum().item())

        # Record token confidence trial
        if conf_info:
            config = {
                "experiment": "token_confidence_real",
                "prompt": prompt_name,
                "input_len": data["input_len"],
            }
            with runner.trial(config) as trial:
                for k, v in conf_info.items():
                    trial.record(k, v)

        # Record KV cache trials
        for kv in kv_results:
            config = {
                "experiment": "kv_cache_real",
                "prompt": prompt_name,
                "layer": kv["layer"],
                "budget_frac": kv["budget_frac"],
            }
            with runner.trial(config) as trial:
                for k, v in kv.items():
                    if isinstance(v, (int, float)):
                        trial.record(k, v)

        # Free attention tensors
        del data["attentions"]
        gc.collect()

    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()

    runner.save()
    runner.to_csv()
    print(runner.report.summary())
    return runner


if __name__ == "__main__":
    run_real_model_experiment()
