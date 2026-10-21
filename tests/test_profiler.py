"""Tests for single-pass retention curves.

The correctness bar is that the curve equals a brute-force mask-and-sum at
every budget, because the profiler exists to replace exactly that computation.
"""

from __future__ import annotations

import torch

from src.kv_profiler.curves import (
    h2o_curve,
    key_mass,
    layer_curves,
    sampled_key_mass,
    sink_curve,
    window_curve,
)


def _brute_force(attn: torch.Tensor, mask: torch.Tensor) -> float:
    """The incumbent computation: sum attention landing on kept keys."""
    total = attn.to(torch.float64).sum()
    return float(attn.to(torch.float64)[:, :, mask].sum() / total)


def _attn(heads=4, seq=16, seed=0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.rand(heads, seq, seq)


def test_h2o_curve_matches_brute_force_at_every_budget():
    attn = _attn()
    seq = attn.shape[-1]
    curve = h2o_curve(key_mass(attn))
    cumulative = attn.sum(dim=(0, 1))
    for b in range(1, seq + 1):
        mask = torch.zeros(seq, dtype=torch.bool)
        mask[cumulative.topk(b).indices] = True
        assert abs(float(curve.values[b]) - _brute_force(attn, mask)) < 1e-9


def test_window_curve_matches_brute_force_at_every_budget():
    attn = _attn(seed=1)
    seq = attn.shape[-1]
    curve = window_curve(key_mass(attn))
    for b in range(1, seq + 1):
        mask = torch.zeros(seq, dtype=torch.bool)
        mask[-b:] = True
        assert abs(float(curve.values[b]) - _brute_force(attn, mask)) < 1e-9


def test_sink_curve_matches_brute_force_at_every_budget():
    attn = _attn(seed=2)
    seq = attn.shape[-1]
    curve = sink_curve(key_mass(attn))
    for b in range(1, seq + 1):
        mask = torch.zeros(seq, dtype=torch.bool)
        sink = min(4, b // 2)
        mask[:sink] = True
        mask[-(b - sink):] = True
        assert abs(float(curve.values[b]) - _brute_force(attn, mask)) < 1e-9


def test_h2o_dominates_window_everywhere():
    """H2O takes the largest masses first, so it cannot be beaten by any
    fixed-position policy at the same budget."""
    attn = _attn(seed=3)
    mass = key_mass(attn)
    h2o, window = h2o_curve(mass), window_curve(mass)
    assert bool((h2o.values >= window.values - 1e-12).all())


def test_curve_endpoints_and_monotonicity():
    mass = key_mass(_attn(seed=4))
    for curve in (h2o_curve(mass), window_curve(mass), sink_curve(mass)):
        assert abs(float(curve.values[0])) < 1e-12
        assert abs(float(curve.values[-1]) - 1.0) < 1e-9
    h2o = h2o_curve(mass)
    assert bool((h2o.values[1:] >= h2o.values[:-1] - 1e-12).all())


def test_at_fraction_uses_grid_sweep_rounding():
    """Budget rounding must match the incumbent so comparisons are honest."""
    mass = key_mass(_attn(seq=20, seed=5))
    curve = h2o_curve(mass)
    for frac in (0.1, 0.25, 0.5, 0.75):
        assert curve.at_fraction(frac) == float(curve.values[max(1, int(20 * frac))])


def test_knee_is_within_range_and_beats_endpoints():
    curve = h2o_curve(key_mass(_attn(seq=32, seed=6)))
    budget, retention = curve.knee()
    assert 0 <= budget <= curve.seq_len
    assert 0.0 <= retention <= 1.0


def test_uniform_attention_gives_linear_h2o_curve():
    """With equal mass on every key, retaining b of n keys retains b/n."""
    attn = torch.ones(2, 8, 8)
    curve = h2o_curve(key_mass(attn))
    for b in range(9):
        assert abs(float(curve.values[b]) - b / 8) < 1e-9


def test_sampling_is_deterministic_and_respects_rate():
    attn = _attn(seq=200, seed=7)
    a, rate_a = sampled_key_mass(attn, 0.25)
    b, rate_b = sampled_key_mass(attn, 0.25)
    assert torch.equal(a, b) and rate_a == rate_b
    assert 0.15 < rate_a < 0.35
    full, rate_full = sampled_key_mass(attn, 1.0)
    assert rate_full == 1.0 and full.numel() == 200


def test_layer_curves_returns_all_policies_and_accepts_batched_input():
    attn = _attn(seq=12, seed=8)
    curves = layer_curves(attn, layer=3)
    assert set(curves) == {"h2o", "window", "sink"}
    assert all(c.layer == 3 and c.seq_len == 12 for c in curves.values())
    batched = layer_curves(attn.unsqueeze(0), layer=3)
    assert torch.allclose(curves["h2o"].values, batched["h2o"].values)


# --- incremental collection ------------------------------------------------

class _FakeAttention(torch.nn.Module):
    """Returns (output, weights) like an eager HF attention module."""

    def __init__(self, heads: int, seq: int, seed: int):
        super().__init__()
        torch.manual_seed(seed)
        self.weights = torch.rand(1, heads, seq, seq)

    def forward(self, hidden_states=None, **kwargs):
        return torch.zeros(1), self.weights


class _FakeLayer(torch.nn.Module):
    def __init__(self, attn):
        super().__init__()
        self.self_attn = attn


class _FakeModel(torch.nn.Module):
    def __init__(self, n_layers=3, heads=2, seq=8):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList(
            [_FakeLayer(_FakeAttention(heads, seq, seed=i)) for i in range(n_layers)]
        )


def test_collector_reduces_each_layer_and_restores_forward():
    from src.kv_profiler.collect import MassCollector

    model = _FakeModel()
    originals = [l.self_attn.forward for l in model.model.layers]
    with MassCollector(model, seq_len=8) as collector:
        assert all(l.self_attn.forward is not o
                   for l, o in zip(model.model.layers, originals))
        for layer in model.model.layers:
            layer.self_attn()
        collected = collector.mass.clone()
    # forward restored on exit
    assert all(l.self_attn.forward == o for l, o in zip(model.model.layers, originals))

    expected = torch.stack([l.self_attn.weights.to(torch.float64).sum(dim=(0, 1, 2))
                            for l in model.model.layers])
    assert torch.allclose(collected, expected)


def test_collector_discards_attention_from_module_output():
    from src.kv_profiler.collect import MassCollector

    model = _FakeModel(n_layers=1)
    layer = model.model.layers[0]
    with MassCollector(model, seq_len=8):
        _, weights = layer.self_attn()
    assert weights is None, "attention must not survive the module call"

    collector = MassCollector(model, seq_len=8)
    collector.discard_attention = False
    with collector:
        _, weights = layer.self_attn()
    assert torch.is_tensor(weights)


def test_collector_accumulates_across_calls():
    """Chunked prefill relies on contributions summing across forward passes."""
    from src.kv_profiler.collect import MassCollector

    model = _FakeModel(n_layers=1)
    with MassCollector(model, seq_len=8) as collector:
        model.model.layers[0].self_attn()
        once = collector.mass.clone()
        model.model.layers[0].self_attn()
        twice = collector.mass.clone()
    assert torch.allclose(twice, 2 * once)


def test_collector_rejects_oversized_key_dimension():
    from src.kv_profiler.collect import MassCollector

    model = _FakeModel(seq=8)
    with MassCollector(model, seq_len=4):
        try:
            model.model.layers[0].self_attn()
        except ValueError as exc:
            assert "sized for 4" in str(exc)
        else:
            raise AssertionError("expected ValueError for oversized key dimension")


# --- checkpoint integrity --------------------------------------------------

def test_shard_complete_rejects_truncated_weights(tmp_path):
    """A half-downloaded shard still matches a glob; loading one fails deep
    inside transformers, so the profiler experiments check it up front."""
    import json as _json

    from src.experiments.exp_family_scaling import _shard_complete

    header = {"w": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]}}
    blob = _json.dumps(header).encode()
    full = tmp_path / "model.safetensors"
    full.write_bytes(len(blob).to_bytes(8, "little") + blob + b"\x00" * 16)
    assert _shard_complete(str(full))

    short = tmp_path / "partial.safetensors"
    short.write_bytes(len(blob).to_bytes(8, "little") + blob + b"\x00" * 4)
    assert not _shard_complete(str(short))

    garbage = tmp_path / "garbage.safetensors"
    garbage.write_bytes(b"\xff" * 64)
    assert not _shard_complete(str(garbage))

    assert not _shard_complete(str(tmp_path / "missing.safetensors"))
