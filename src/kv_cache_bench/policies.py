"""
KV Cache eviction policies for benchmark comparison.

Each policy implements the same interface:
    select_tokens_to_keep(attention_scores, budget) -> mask

This allows apples-to-apples comparison across strategies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass
class EvictionResult:
    """Result of a cache eviction decision."""
    keep_mask: torch.Tensor       # Boolean mask: True = keep, False = evict
    tokens_kept: int              # Number of tokens retained
    tokens_evicted: int           # Number of tokens evicted
    budget_used_pct: float        # Percentage of budget used
    policy_name: str              # Name of the eviction policy


class EvictionPolicy(ABC):
    """Base class for KV cache eviction policies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable policy name."""
        ...

    @abstractmethod
    def select(
        self,
        attention_scores: torch.Tensor,
        budget: int,
        layer_idx: int = 0,
        total_layers: int = 1,
    ) -> EvictionResult:
        """
        Decide which KV cache entries to keep.

        Args:
            attention_scores: Attention weights [num_heads, seq_len, seq_len]
                              or [batch, num_heads, seq_len, seq_len]
            budget: Maximum number of tokens to keep in cache.
            layer_idx: Current transformer layer index (for layer-aware policies).
            total_layers: Total number of layers (for layer-aware policies).

        Returns:
            EvictionResult with keep mask and metadata.
        """
        ...


class FullCachePolicy(EvictionPolicy):
    """
    Baseline: keep everything. No eviction.
    This is the gold standard for quality — all other policies
    are measured against the quality loss from this baseline.
    """

    @property
    def name(self) -> str:
        return "full"

    def select(self, attention_scores, budget, layer_idx=0, total_layers=1):
        seq_len = attention_scores.shape[-1]
        keep_mask = torch.ones(seq_len, dtype=torch.bool, device=attention_scores.device)
        return EvictionResult(
            keep_mask=keep_mask,
            tokens_kept=seq_len,
            tokens_evicted=0,
            budget_used_pct=100.0,
            policy_name=self.name,
        )


class WindowPolicy(EvictionPolicy):
    """
    StreamingLLM-style: keep the first `sink` tokens (attention sinks)
    and the last `budget - sink` recent tokens.

    Reference: Xiao et al., "Efficient Streaming Language Models
    with Attention Sinks" (2023)
    """

    def __init__(self, sink_tokens: int = 4):
        self.sink_tokens = sink_tokens

    @property
    def name(self) -> str:
        return "window"

    def select(self, attention_scores, budget, layer_idx=0, total_layers=1):
        seq_len = attention_scores.shape[-1]
        if budget >= seq_len:
            return FullCachePolicy().select(attention_scores, budget)

        keep_mask = torch.zeros(seq_len, dtype=torch.bool, device=attention_scores.device)

        # Keep attention sinks (first N tokens)
        sink_count = min(self.sink_tokens, budget)
        keep_mask[:sink_count] = True

        # Keep recent tokens
        recent_count = budget - sink_count
        if recent_count > 0:
            keep_mask[-recent_count:] = True

        kept = keep_mask.sum().item()
        return EvictionResult(
            keep_mask=keep_mask,
            tokens_kept=kept,
            tokens_evicted=seq_len - kept,
            budget_used_pct=(kept / budget * 100) if budget > 0 else 0,
            policy_name=self.name,
        )


class H2OPolicy(EvictionPolicy):
    """
    Heavy Hitter Oracle: keep tokens that receive the most
    cumulative attention across all heads.

    Key insight: a small fraction of tokens ("heavy hitters")
    accumulate most of the attention mass. Keeping only these
    preserves quality while drastically reducing cache size.

    Reference: Zhang et al., "H2O: Heavy-Hitter Oracle for
    Efficient Generative Inference of Large Language Models" (2023)
    """

    @property
    def name(self) -> str:
        return "h2o"

    def select(self, attention_scores, budget, layer_idx=0, total_layers=1):
        seq_len = attention_scores.shape[-1]
        if budget >= seq_len:
            return FullCachePolicy().select(attention_scores, budget)

        # Sum attention across heads → [seq_len, seq_len] or [seq_len]
        if attention_scores.dim() == 4:
            # [batch, heads, seq_len, seq_len] → sum over batch and heads
            attn_sum = attention_scores.sum(dim=(0, 1))  # [seq_len, seq_len]
        elif attention_scores.dim() == 3:
            # [heads, seq_len, seq_len] → sum over heads
            attn_sum = attention_scores.sum(dim=0)  # [seq_len, seq_len]
        else:
            attn_sum = attention_scores

        # Cumulative attention received by each token (column-wise sum)
        # = how much attention all query positions pay to this key position
        token_importance = attn_sum.sum(dim=0)  # [seq_len]

        # Keep top-budget tokens by importance
        _, top_indices = token_importance.topk(min(budget, seq_len))
        keep_mask = torch.zeros(seq_len, dtype=torch.bool, device=attention_scores.device)
        keep_mask[top_indices] = True

        kept = keep_mask.sum().item()
        return EvictionResult(
            keep_mask=keep_mask,
            tokens_kept=kept,
            tokens_evicted=seq_len - kept,
            budget_used_pct=(kept / budget * 100) if budget > 0 else 0,
            policy_name=self.name,
        )


class SnapKVPolicy(EvictionPolicy):
    """
    Attention-variance-based selection: keep tokens whose attention
    scores have high variance across heads. High-variance tokens
    are "informative" — different heads disagree on them, so they
    carry more discriminative signal.

    Reference: Li et al., "SnapKV: LLM Knows What You are Looking for
    Before Generation" (2024)
    """

    def __init__(self, observation_window: int = 16):
        self.observation_window = observation_window

    @property
    def name(self) -> str:
        return "snapkv"

    def select(self, attention_scores, budget, layer_idx=0, total_layers=1):
        seq_len = attention_scores.shape[-1]
        if budget >= seq_len:
            return FullCachePolicy().select(attention_scores, budget)

        # Use last `observation_window` queries to observe attention patterns
        if attention_scores.dim() == 4:
            # [batch, heads, seq_len, seq_len]
            obs = attention_scores[:, :, -self.observation_window:, :]
            # Variance across heads for each key position
            var_across_heads = obs.var(dim=1).mean(dim=(0, 2))  # [seq_len]
        elif attention_scores.dim() == 3:
            # [heads, seq_len, seq_len]
            obs = attention_scores[:, -self.observation_window:, :]
            var_across_heads = obs.var(dim=0).mean(dim=0)  # [seq_len]
        else:
            var_across_heads = attention_scores.var(dim=0)

        # Keep tokens with highest attention variance
        _, top_indices = var_across_heads.topk(min(budget, seq_len))
        keep_mask = torch.zeros(seq_len, dtype=torch.bool, device=attention_scores.device)
        keep_mask[top_indices] = True

        kept = keep_mask.sum().item()
        return EvictionResult(
            keep_mask=keep_mask,
            tokens_kept=kept,
            tokens_evicted=seq_len - kept,
            budget_used_pct=(kept / budget * 100) if budget > 0 else 0,
            policy_name=self.name,
        )


class PyramidKVPolicy(EvictionPolicy):
    """
    Layer-wise budget allocation: early layers get smaller budgets
    (they capture low-level patterns with less context dependency),
    later layers get larger budgets (they need more context for
    complex reasoning).

    The budget follows a pyramid shape: layer i gets
    budget * (i / total_layers) tokens.

    Reference: Cai et al., "PyramidKV: Dynamic KV Cache Compression
    based on Pyramidal Information Funneling" (2024)
    """

    @property
    def name(self) -> str:
        return "pyramid"

    def select(self, attention_scores, budget, layer_idx=0, total_layers=1):
        seq_len = attention_scores.shape[-1]

        # Scale budget by layer position (pyramid: early=small, late=large)
        if total_layers > 1:
            layer_ratio = (layer_idx + 1) / total_layers
            layer_budget = max(4, int(budget * layer_ratio))
        else:
            layer_budget = budget

        if layer_budget >= seq_len:
            return FullCachePolicy().select(attention_scores, budget)

        # Use H2O-style importance within the layer-specific budget
        h2o = H2OPolicy()
        result = h2o.select(attention_scores, layer_budget, layer_idx, total_layers)
        result.policy_name = self.name
        return result


class AdaptivePolicy(EvictionPolicy):
    """
    NOVEL CONTRIBUTION: Dynamic per-request cache budget based on
    attention entropy.

    Key insight (analogous to LoRA's simplicity):
    Instead of using a fixed eviction budget for all requests,
    let the attention pattern itself decide how much to keep.

    High attention entropy → complex prompt → keep more cache
    Low attention entropy → simple retrieval → evict aggressively

    This is the novel contribution that could be paper-worthy.
    """

    def __init__(self, min_budget_ratio: float = 0.25, max_budget_ratio: float = 1.0):
        self.min_budget_ratio = min_budget_ratio
        self.max_budget_ratio = max_budget_ratio

    @property
    def name(self) -> str:
        return "adaptive"

    def _compute_entropy(self, attention_scores: torch.Tensor) -> float:
        """
        Compute attention entropy as a complexity measure.

        High entropy = attention is spread across many tokens (complex reasoning)
        Low entropy = attention is concentrated on few tokens (simple retrieval)
        """
        if attention_scores.dim() == 4:
            # [batch, heads, seq_len, seq_len] → average over batch
            attn = attention_scores.mean(dim=0)  # [heads, seq_len, seq_len]
        elif attention_scores.dim() == 3:
            attn = attention_scores  # [heads, seq_len, seq_len]
        else:
            attn = attention_scores.unsqueeze(0)

        # Clamp to avoid log(0)
        attn = attn.clamp(min=1e-10)

        # Shannon entropy per head, per query position
        entropy = -torch.sum(attn * torch.log(attn), dim=-1)  # [heads, seq_len]

        # Average entropy across heads and positions
        avg_entropy = entropy.mean().item()

        # Normalize to [0, 1] range using max possible entropy
        seq_len = attention_scores.shape[-1]
        max_entropy = torch.log(torch.tensor(float(seq_len))).item()
        normalized = avg_entropy / max_entropy if max_entropy > 0 else 0

        return normalized

    def select(self, attention_scores, budget, layer_idx=0, total_layers=1):
        seq_len = attention_scores.shape[-1]

        # Compute complexity
        complexity = self._compute_entropy(attention_scores)

        # Scale budget: simple prompts get min_ratio, complex get max_ratio
        budget_ratio = self.min_budget_ratio + complexity * (
            self.max_budget_ratio - self.min_budget_ratio
        )
        adaptive_budget = max(4, int(budget * budget_ratio))
        adaptive_budget = min(adaptive_budget, seq_len)

        if adaptive_budget >= seq_len:
            return FullCachePolicy().select(attention_scores, budget)

        # Use H2O selection within the adaptive budget
        h2o = H2OPolicy()
        result = h2o.select(attention_scores, adaptive_budget, layer_idx, total_layers)
        result.policy_name = self.name
        return result


# Registry for easy lookup
POLICIES: dict[str, type[EvictionPolicy]] = {
    "full": FullCachePolicy,
    "window": WindowPolicy,
    "h2o": H2OPolicy,
    "snapkv": SnapKVPolicy,
    "pyramid": PyramidKVPolicy,
    "adaptive": AdaptivePolicy,
}


def get_policy(name: str, **kwargs) -> EvictionPolicy:
    """Get an eviction policy by name."""
    if name not in POLICIES:
        raise ValueError(f"Unknown policy: {name}. Available: {list(POLICIES.keys())}")
    return POLICIES[name](**kwargs)
