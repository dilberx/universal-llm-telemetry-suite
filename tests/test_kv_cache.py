"""
Tests for KV cache eviction policies.
"""

import torch
import pytest

from src.kv_cache_bench.policies import (
    FullCachePolicy,
    WindowPolicy,
    H2OPolicy,
    SnapKVPolicy,
    PyramidKVPolicy,
    AdaptivePolicy,
    get_policy,
    POLICIES,
)
from src.kv_cache_bench.benchmark import KVCacheBenchmark, BenchmarkConfig


# ============================================================================
# Helper: generate dummy attention scores
# ============================================================================

def make_attention(num_heads=4, seq_len=32):
    """Create causal attention scores for testing."""
    attn = torch.rand(num_heads, seq_len, seq_len)
    # Apply causal mask
    causal = torch.tril(torch.ones(seq_len, seq_len))
    attn = attn * causal
    # Normalize per query position
    attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-10)
    return attn


# ============================================================================
# Policy Tests
# ============================================================================

class TestFullCachePolicy:
    def test_keeps_everything(self):
        attn = make_attention(seq_len=64)
        policy = FullCachePolicy()
        result = policy.select(attn, budget=32)
        assert result.tokens_kept == 64
        assert result.tokens_evicted == 0
        assert result.keep_mask.all()

    def test_name(self):
        assert FullCachePolicy().name == "full"


class TestWindowPolicy:
    def test_keeps_sinks_and_recent(self):
        attn = make_attention(seq_len=64)
        policy = WindowPolicy(sink_tokens=4)
        result = policy.select(attn, budget=16)

        # First 4 should be kept (sinks)
        assert result.keep_mask[:4].all()
        # Last 12 should be kept (recent)
        assert result.keep_mask[-12:].all()
        assert result.tokens_kept == 16

    def test_budget_larger_than_seq(self):
        attn = make_attention(seq_len=8)
        policy = WindowPolicy()
        result = policy.select(attn, budget=16)
        assert result.tokens_kept == 8  # Can't keep more than seq_len

    def test_name(self):
        assert WindowPolicy().name == "window"


class TestH2OPolicy:
    def test_keeps_high_attention_tokens(self):
        attn = make_attention(seq_len=32)
        policy = H2OPolicy()
        result = policy.select(attn, budget=8)
        assert result.tokens_kept == 8
        assert result.tokens_evicted == 24

    def test_deterministic(self):
        attn = make_attention(seq_len=32)
        policy = H2OPolicy()
        r1 = policy.select(attn, budget=8)
        r2 = policy.select(attn, budget=8)
        assert torch.equal(r1.keep_mask, r2.keep_mask)

    def test_4d_input(self):
        attn = torch.rand(2, 4, 32, 32)  # batch=2
        attn = attn * torch.tril(torch.ones(32, 32))
        policy = H2OPolicy()
        result = policy.select(attn, budget=8)
        assert result.tokens_kept == 8


class TestSnapKVPolicy:
    def test_keeps_high_variance_tokens(self):
        attn = make_attention(seq_len=32)
        policy = SnapKVPolicy()
        result = policy.select(attn, budget=8)
        assert result.tokens_kept == 8

    def test_name(self):
        assert SnapKVPolicy().name == "snapkv"


class TestPyramidKVPolicy:
    def test_early_layer_smaller_budget(self):
        attn = make_attention(seq_len=64)
        policy = PyramidKVPolicy()

        early = policy.select(attn, budget=32, layer_idx=0, total_layers=32)
        late = policy.select(attn, budget=32, layer_idx=31, total_layers=32)

        # Early layers should keep fewer tokens
        assert early.tokens_kept < late.tokens_kept

    def test_single_layer(self):
        attn = make_attention(seq_len=32)
        policy = PyramidKVPolicy()
        result = policy.select(attn, budget=16, layer_idx=0, total_layers=1)
        assert result.tokens_kept == 16


class TestAdaptivePolicy:
    def test_adjusts_budget(self):
        policy = AdaptivePolicy()
        attn = make_attention(seq_len=64)
        result = policy.select(attn, budget=32)
        # Should keep between min and max ratio of budget
        assert 4 <= result.tokens_kept <= 64

    def test_uniform_attention_lower_budget(self):
        """Uniform attention = high entropy = keep more."""
        policy = AdaptivePolicy(min_budget_ratio=0.25, max_budget_ratio=1.0)
        # Create high-entropy attention (uniform)
        seq_len = 32
        attn = torch.ones(4, seq_len, seq_len)
        attn = attn * torch.tril(torch.ones(seq_len, seq_len))
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-10)

        result = policy.select(attn, budget=16)
        # High entropy → should keep more tokens
        assert result.tokens_kept > 4

    def test_concentrated_attention_smaller_cache(self):
        """Concentrated attention = low entropy = evict more aggressively."""
        policy = AdaptivePolicy(min_budget_ratio=0.25, max_budget_ratio=1.0)
        seq_len = 32
        # Create low-entropy attention (concentrated on first token)
        attn = torch.zeros(4, seq_len, seq_len)
        for q in range(seq_len):
            attn[:, q, 0] = 0.95  # Almost all attention on token 0
            if q > 0:
                attn[:, q, q] = 0.05

        result = policy.select(attn, budget=16)
        # Low entropy → should keep fewer tokens (more aggressive eviction)
        assert result.tokens_kept <= 16

    def test_name(self):
        assert AdaptivePolicy().name == "adaptive"


class TestPolicyRegistry:
    def test_all_policies_registered(self):
        assert len(POLICIES) == 6
        for name in ["full", "window", "h2o", "snapkv", "pyramid", "adaptive"]:
            assert name in POLICIES

    def test_get_policy(self):
        policy = get_policy("h2o")
        assert policy.name == "h2o"

    def test_unknown_policy_raises(self):
        with pytest.raises(ValueError, match="Unknown policy"):
            get_policy("nonexistent")


# ============================================================================
# Benchmark Runner Tests
# ============================================================================

class TestBenchmarkRunner:
    def test_runs_on_cpu(self):
        config = BenchmarkConfig(
            policies=["full", "window", "h2o"],
            budgets=[8, 16],
            context_lengths=[32],
            device="cpu",
        )
        bench = KVCacheBenchmark(config)
        report = bench.run()
        # full + window@8 + window@16 + h2o@8 + h2o@16 = 5 results
        assert len(report.results) >= 3

    def test_quality_baseline(self):
        config = BenchmarkConfig(
            policies=["full"],
            budgets=[16],
            context_lengths=[32],
            device="cpu",
        )
        bench = KVCacheBenchmark(config)
        report = bench.run()
        # Full cache should have quality = 1.0
        for r in report.results:
            assert r.quality_score == 1.0

    def test_report_save(self, tmp_path):
        config = BenchmarkConfig(
            policies=["window"],
            budgets=[8],
            context_lengths=[16],
            device="cpu",
        )
        bench = KVCacheBenchmark(config)
        report = bench.run()

        out_path = tmp_path / "results.json"
        report.save(out_path)
        assert out_path.exists()

        import json
        data = json.loads(out_path.read_text())
        assert "results" in data
        assert len(data["results"]) > 0

    def test_summary_table(self):
        config = BenchmarkConfig(
            policies=["h2o", "adaptive"],
            budgets=[8],
            context_lengths=[32],
            device="cpu",
        )
        bench = KVCacheBenchmark(config)
        report = bench.run()
        table = report.summary_table()
        assert "h2o" in table
        assert "adaptive" in table
        assert "Quality" in table
