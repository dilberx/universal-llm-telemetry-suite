"""
Experiment 2A: Attention Head Importance Profiling

Question: How many attention heads can you disable at inference
time without losing quality? Which heads are critical vs. redundant?

Produces a "head importance map" — a visual heatmap showing which
heads in which layers are essential.

Can run with synthetic attention or real model hooks.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.experiments.runner import ExperimentRunner


def compute_head_importance(
    attention_weights: torch.Tensor,
) -> torch.Tensor:
    """
    Compute importance score for each attention head.

    Multiple metrics combined:
    - Attention entropy (low = focused = important)
    - Attention mass concentration (high = important)
    - Head agreement (unique heads = important)

    Args:
        attention_weights: [num_layers, num_heads, seq_len, seq_len]

    Returns:
        importance: [num_layers, num_heads] — higher = more important
    """
    num_layers, num_heads, seq_len, _ = attention_weights.shape

    importance = torch.zeros(num_layers, num_heads)

    for layer in range(num_layers):
        for head in range(num_heads):
            attn = attention_weights[layer, head]  # [seq_len, seq_len]

            # 1. Entropy: low entropy = focused attention = important
            attn_clamped = attn.clamp(min=1e-10)
            entropy = -(attn_clamped * torch.log(attn_clamped)).sum(dim=-1).mean()
            max_entropy = torch.log(torch.tensor(float(seq_len)))
            normalized_entropy = entropy / max_entropy
            focus_score = 1.0 - normalized_entropy.item()

            # 2. Mass concentration: top-10% of positions capture how much mass?
            top_k = max(1, seq_len // 10)
            top_mass = attn.topk(top_k, dim=-1).values.sum(dim=-1).mean().item()

            # 3. Uniqueness: how different is this head from others in same layer?
            other_heads = [h for h in range(num_heads) if h != head]
            if other_heads:
                similarities = []
                for other in other_heads:
                    sim = F.cosine_similarity(
                        attn.reshape(-1).unsqueeze(0),
                        attention_weights[layer, other].reshape(-1).unsqueeze(0),
                    ).item()
                    similarities.append(sim)
                uniqueness = 1.0 - (sum(similarities) / len(similarities))
            else:
                uniqueness = 1.0

            # Combined importance
            importance[layer, head] = (
                0.4 * focus_score +
                0.3 * top_mass +
                0.3 * uniqueness
            )

    return importance


def simulate_head_pruning(
    attention_weights: torch.Tensor,
    importance: torch.Tensor,
    prune_fraction: float,
) -> dict:
    """
    Simulate pruning heads and measure quality impact.

    Returns quality metrics comparing full vs. pruned attention.
    """
    num_layers, num_heads, seq_len, _ = attention_weights.shape

    # Flatten importance to find global threshold
    flat_importance = importance.reshape(-1)
    total_heads = flat_importance.numel()
    num_to_prune = int(total_heads * prune_fraction)

    if num_to_prune == 0:
        return {
            "prune_fraction": prune_fraction,
            "heads_pruned": 0,
            "heads_remaining": total_heads,
            "quality_score": 1.0,
            "attention_mass_retained": 1.0,
        }

    # Find threshold
    threshold = flat_importance.topk(total_heads - num_to_prune, largest=False).values.max()

    # Prune: zero out low-importance heads
    pruned = attention_weights.clone()
    heads_pruned = 0
    for layer in range(num_layers):
        for head in range(num_heads):
            if importance[layer, head] <= threshold:
                pruned[layer, head] = 0
                heads_pruned += 1

    # Renormalize remaining heads
    head_sums = pruned.sum(dim=1, keepdim=True)
    head_sums = head_sums.clamp(min=1e-10)
    pruned_normalized = pruned / head_sums * num_heads

    # Quality: compare attention distributions
    full_attn = attention_weights.mean(dim=1)  # Average across heads
    pruned_attn = pruned_normalized.mean(dim=1)

    # KL divergence between full and pruned attention
    full_flat = full_attn.reshape(-1).clamp(min=1e-10)
    pruned_flat = pruned_attn.reshape(-1).clamp(min=1e-10)
    full_flat = full_flat / full_flat.sum()
    pruned_flat = pruned_flat / pruned_flat.sum()

    kl_div = F.kl_div(
        pruned_flat.log(), full_flat, reduction="sum"
    ).item()

    # Cosine similarity
    cos_sim = F.cosine_similarity(
        full_attn.reshape(1, -1),
        pruned_attn.reshape(1, -1),
    ).item()

    # Attention mass retained
    mass_retained = pruned.sum().item() / attention_weights.sum().item()

    return {
        "prune_fraction": prune_fraction,
        "heads_pruned": heads_pruned,
        "heads_remaining": total_heads - heads_pruned,
        "quality_score": cos_sim,
        "kl_divergence": kl_div,
        "attention_mass_retained": mass_retained,
    }


def generate_model_attention(
    num_layers: int = 24,
    num_heads: int = 16,
    seq_len: int = 128,
    pattern: str = "realistic",
    seed: int = 42,
) -> torch.Tensor:
    """Generate synthetic attention patterns for profiling."""
    torch.manual_seed(seed)

    attn = torch.zeros(num_layers, num_heads, seq_len, seq_len)

    for layer in range(num_layers):
        for head in range(num_heads):
            # Layer depth affects pattern
            layer_ratio = layer / num_layers

            # Create causal mask
            raw = torch.randn(seq_len, seq_len)

            # Add patterns based on layer depth
            if layer_ratio < 0.3:
                # Early layers: local attention
                for i in range(seq_len):
                    window = max(4, int(16 * (1 - layer_ratio)))
                    start = max(0, i - window)
                    raw[i, start:i+1] += 3.0
            elif layer_ratio < 0.7:
                # Middle layers: mixed local + global
                for i in range(seq_len):
                    raw[i, :4] += 1.5  # Attention sinks
                    window = 32
                    start = max(0, i - window)
                    raw[i, start:i+1] += 2.0
            else:
                # Late layers: global patterns
                raw[:, :4] += 2.0  # Strong sinks
                # Some heads attend to specific positions
                if head % 4 == 0:
                    raw += torch.eye(seq_len) * 3.0

            # Head-specific variation
            if head % 3 == 0:
                raw *= 1.5  # Some heads are "louder"

            # Apply causal mask + softmax
            causal = torch.triu(torch.ones(seq_len, seq_len) * -1e9, diagonal=1)
            raw = raw + causal
            attn[layer, head] = F.softmax(raw, dim=-1)

    return attn


def run_attention_head_experiment():
    """Run the attention head importance profiling experiment."""
    runner = ExperimentRunner(
        name="attention_head_importance",
        description=(
            "Profiling attention head importance across layers and "
            "measuring quality impact of progressive head pruning. "
            "Finds the Pareto frontier of heads-removed vs. quality-retained."
        ),
    )

    # Model configurations to test
    model_configs = [
        {"name": "small", "layers": 12, "heads": 12, "seq_len": 128},
        {"name": "medium", "layers": 24, "heads": 16, "seq_len": 128},
        {"name": "large", "layers": 32, "heads": 32, "seq_len": 128},
    ]

    prune_fractions = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8]
    patterns = ["realistic"]

    for model_cfg in model_configs:
        for pattern in patterns:
            attn = generate_model_attention(
                num_layers=model_cfg["layers"],
                num_heads=model_cfg["heads"],
                seq_len=model_cfg["seq_len"],
                pattern=pattern,
            )

            importance = compute_head_importance(attn)

            # Log the importance map
            for prune_frac in prune_fractions:
                config = {
                    "model": model_cfg["name"],
                    "layers": model_cfg["layers"],
                    "heads": model_cfg["heads"],
                    "seq_len": model_cfg["seq_len"],
                    "prune_fraction": prune_frac,
                    "pattern": pattern,
                }

                with runner.trial(config) as trial:
                    result = simulate_head_pruning(attn, importance, prune_frac)
                    for k, v in result.items():
                        if isinstance(v, (int, float)):
                            trial.record(k, v)

                    # Also record importance stats
                    trial.record("importance_mean", importance.mean().item())
                    trial.record("importance_std", importance.std().item())
                    trial.record("importance_min", importance.min().item())
                    trial.record("importance_max", importance.max().item())

    runner.save()
    runner.to_csv()
    print(runner.report.summary())
    return runner


if __name__ == "__main__":
    run_attention_head_experiment()
