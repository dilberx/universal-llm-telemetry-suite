"""
KV Cache Eviction Policy Benchmark Suite.

Systematically benchmarks KV cache management strategies for LLM inference:
- Full (no eviction — gold standard baseline)
- Window (StreamingLLM — keep first + last N tokens)
- H2O (Heavy Hitter Oracle — keep high-attention tokens)
- SnapKV (attention-variance-based selection)
- Adaptive (NOVEL: dynamic per-request budget based on attention entropy)

Measures:
- Memory savings (MB reduction vs. full cache)
- Quality degradation (perplexity delta vs. full cache)
- Throughput impact (tokens/sec)
- Time to first token (TTFT)
- Cache hit rate (how often evicted tokens are re-needed)

This module lives inside llm-inference-benchmark because it extends
the existing benchmarking infrastructure with a new dimension:
memory management efficiency.
"""
