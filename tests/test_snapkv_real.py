"""SnapKV compression: per-head selection, observation window and equal-memory budget."""

import torch
from transformers import DynamicCache

from src.experiments.exp_e2e_kv_cache import _snapkv_compress, _snapkv_prefill_budget


def _cache(n_layers=2, n_kv=2, n=40, d=3):
    torch.manual_seed(0)
    layers = tuple((torch.randn(1, n_kv, n, d), torch.randn(1, n_kv, n, d)) for _ in range(n_layers))
    return DynamicCache.from_legacy_cache(layers), layers


def _attn(n_layers=2, n_q=4, n=40, window=8, hot=(5, 11)):
    """Query heads 0-1 (KV head 0) vote for key hot[0]; heads 2-3 (KV head 1) for hot[1]."""
    attn = torch.zeros(1, n_q, n, n)
    attn[0, 0:2, -window:, hot[0]] = 1.0
    attn[0, 2:4, -window:, hot[1]] = 1.0
    return [attn.clone() for _ in range(n_layers)]


def test_shapes_and_window_kept():
    cache, layers = _cache()
    out = _snapkv_compress(cache, _attn(), budget=20, window=8, kernel=1).to_legacy_cache()
    for k, v in out:
        assert k.shape == (1, 2, 20, 3) and v.shape == k.shape


def test_each_head_keeps_its_own_voted_key_and_the_window():
    cache, layers = _cache()
    out = _snapkv_compress(cache, _attn(), budget=20, window=8, kernel=1).to_legacy_cache()
    key0, _ = layers[0]
    k = out[0][0]
    # head 0 kept key 5, head 1 kept key 11: find them by value
    assert any(torch.equal(k[0, 0, i], key0[0, 0, 5]) for i in range(20))
    assert any(torch.equal(k[0, 1, i], key0[0, 1, 11]) for i in range(20))
    # the last eight keys (the observation window) are kept in every head
    for h in range(2):
        for pos in range(32, 40):
            assert any(torch.equal(k[0, h, i], key0[0, h, pos]) for i in range(20))


def test_kept_entries_are_original_rows_in_order():
    cache, layers = _cache()
    out = _snapkv_compress(cache, _attn(), budget=20, window=8, kernel=1).to_legacy_cache()
    key0 = layers[0][0]
    for h in range(2):
        rows = out[0][0][0, h]
        idx = [int((key0[0, h] == r).all(dim=-1).nonzero()[0]) for r in rows]
        assert idx == sorted(idx) and len(set(idx)) == 20


def test_budget_at_or_above_length_is_a_no_op():
    cache, layers = _cache()
    out = _snapkv_compress(cache, _attn(), budget=40).to_legacy_cache()
    assert torch.equal(out[0][0], layers[0][0])


def test_prefill_budget_matches_mean_cache_of_per_step_policies():
    n, n_dec = 1024, 31
    for frac in (0.1, 0.3, 0.5, 0.9):
        b0 = _snapkv_prefill_budget(n, frac, n_dec)
        mine = sum(b0 + j for j in range(n_dec)) / n_dec
        others = sum(int((n + j) * frac) for j in range(n_dec)) / n_dec
        assert abs(mine - others) <= 1.0
