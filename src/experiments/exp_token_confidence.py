"""
Experiment 3B: Token Confidence Thresholding

Question: During decode, if the model is >X% confident about the
next token, can we skip full sampling and just emit argmax?

Measures:
- skip_rate: % of tokens where confidence > threshold
- quality_delta: change in perplexity vs. full sampling
- speedup_pct: theoretical speedup from skipping sampling

This can run on CPU with synthetic logits first, then real model.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.experiments.runner import ExperimentRunner


def generate_realistic_logits(
    vocab_size: int = 32000,
    seq_len: int = 512,
    temperature: float = 0.7,
    confidence_profile: str = "natural",
    seed: int = 42,
) -> torch.Tensor:
    """
    Generate synthetic but realistic logit distributions.

    Profiles:
    - "natural": mix of confident and uncertain tokens (realistic)
    - "confident": mostly high-confidence tokens (simple prompt)
    - "uncertain": mostly low-confidence tokens (complex reasoning)
    """
    torch.manual_seed(seed)

    logits = torch.randn(seq_len, vocab_size)

    if confidence_profile == "confident":
        # Make ~70% of tokens very confident (one logit much higher)
        confident_mask = torch.rand(seq_len) < 0.7
        for i in range(seq_len):
            if confident_mask[i]:
                top_idx = torch.randint(0, vocab_size, (1,))
                logits[i, top_idx] += 8.0  # Very sharp peak
    elif confidence_profile == "uncertain":
        # Make logits more uniform (less confident)
        logits = logits * 0.3  # Flatten distribution
    else:  # natural
        # Mix: ~40% confident, ~40% medium, ~20% uncertain
        for i in range(seq_len):
            r = torch.rand(1).item()
            if r < 0.4:
                top_idx = torch.randint(0, vocab_size, (1,))
                logits[i, top_idx] += 5.0
            elif r < 0.8:
                top_idx = torch.randint(0, vocab_size, (1,))
                logits[i, top_idx] += 2.0

    return logits / temperature


def analyze_confidence(
    logits: torch.Tensor,
    threshold: float,
) -> dict:
    """Analyze a logit sequence at a given confidence threshold."""
    probs = F.softmax(logits, dim=-1)
    top_probs, top_indices = probs.max(dim=-1)

    # How many tokens exceed the threshold?
    confident = top_probs > threshold
    skip_rate = confident.float().mean().item()

    # What's the entropy of confident vs. non-confident tokens?
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)

    confident_entropy = entropy[confident].mean().item() if confident.any() else 0
    uncertain_entropy = entropy[~confident].mean().item() if (~confident).any() else 0

    # Quality: compare argmax selection vs. sampling
    # For confident tokens, argmax == sample (effectively)
    # For uncertain tokens, argmax might miss the right token
    argmax_tokens = logits.argmax(dim=-1)

    # Simulate top-p sampling for comparison
    torch.manual_seed(42)
    sampled_tokens = torch.multinomial(probs, 1).squeeze(-1)
    agreement_rate = (argmax_tokens == sampled_tokens).float().mean().item()

    return {
        "threshold": threshold,
        "skip_rate": skip_rate,
        "confident_entropy_avg": confident_entropy,
        "uncertain_entropy_avg": uncertain_entropy,
        "argmax_sample_agreement": agreement_rate,
        "avg_top_prob": top_probs.mean().item(),
        "median_top_prob": top_probs.median().item(),
        "p90_top_prob": top_probs.quantile(0.9).item(),
        "p10_top_prob": top_probs.quantile(0.1).item(),
    }


def run_token_confidence_experiment():
    """Run the full token confidence experiment."""
    runner = ExperimentRunner(
        name="token_confidence_threshold",
        description=(
            "Measuring when argmax is safe to skip full sampling. "
            "For each (threshold, profile, temperature) combo, compute "
            "skip rate, quality agreement, and entropy distribution."
        ),
    )

    thresholds = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99, 0.995, 0.999]
    profiles = ["natural", "confident", "uncertain"]
    temperatures = [0.3, 0.5, 0.7, 1.0, 1.5]
    vocab_sizes = [32000, 128256]  # Llama-style, GPT-4-style

    total = len(thresholds) * len(profiles) * len(temperatures) * len(vocab_sizes)
    print(f"Running {total} trials...")

    for vocab_size in vocab_sizes:
        for profile in profiles:
            for temp in temperatures:
                logits = generate_realistic_logits(
                    vocab_size=vocab_size,
                    seq_len=512,
                    temperature=temp,
                    confidence_profile=profile,
                )

                for threshold in thresholds:
                    config = {
                        "threshold": threshold,
                        "profile": profile,
                        "temperature": temp,
                        "vocab_size": vocab_size,
                        "seq_len": 512,
                    }

                    with runner.trial(config) as trial:
                        result = analyze_confidence(logits, threshold)
                        for k, v in result.items():
                            if isinstance(v, (int, float)):
                                trial.record(k, v)

    runner.save()
    runner.to_csv()
    print(runner.report.summary())
    return runner


if __name__ == "__main__":
    run_token_confidence_experiment()
