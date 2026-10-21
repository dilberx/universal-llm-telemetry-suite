"""Tests for the profiling driver and adaptive loading.

These cover the parts that do not need a model on the GPU: curve serialisation,
weight estimation, placement choice, and the shape of the emitted artifact.
"""

from __future__ import annotations

import json

import torch

from src.kv_profiler.curves import h2o_curve, key_mass
from src.kv_profiler.loading import estimate_weight_gib, placement_for
from src.kv_profiler.profile import (
    BUDGET_FRACTIONS,
    CURVE_POINTS,
    _downsample,
    calibration_text,
    print_summary,
    write_profile,
)


def _curve(seq=64, seed=0):
    torch.manual_seed(seed)
    return h2o_curve(key_mass(torch.rand(2, seq, seq)))


def test_downsample_preserves_endpoints_and_monotonicity():
    curve = _curve()
    pts = _downsample(curve)
    assert len(pts) == CURVE_POINTS
    assert pts[0] == 0.0
    assert abs(pts[-1] - 1.0) < 1e-6
    assert all(a <= b + 1e-9 for a, b in zip(pts, pts[1:]))


def test_downsample_tracks_the_real_curve():
    """The stored curve must agree with the exact one at the points it keeps."""
    curve = _curve(seq=100, seed=1)
    pts = _downsample(curve, n_points=101)
    for i in (10, 25, 50, 75):
        assert abs(pts[i] - float(curve.values[i])) < 1e-5


def test_estimate_weight_measures_local_checkpoint(tmp_path):
    (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"\x00" * (3 * 1024 ** 2))
    (tmp_path / "model-00002-of-00002.safetensors").write_bytes(b"\x00" * (1 * 1024 ** 2))
    got = estimate_weight_gib(str(tmp_path))
    assert abs(got - 4 / 1024) < 1e-6


def test_estimate_weight_falls_back_to_config(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({
        "hidden_size": 896, "num_hidden_layers": 24,
        "intermediate_size": 4864, "vocab_size": 151936,
    }))
    got = estimate_weight_gib(str(tmp_path))
    assert 0.3 < got < 4.0, f"implausible estimate {got}"


def test_estimate_weight_unknown_source_assumes_offload(tmp_path):
    assert estimate_weight_gib(str(tmp_path / "nonexistent")) == 99.0


def test_placement_splits_when_weights_exceed_free_memory():
    if not torch.cuda.is_available():
        assert placement_for(1.0) is None
        return
    free_gib = torch.cuda.mem_get_info()[0] / 1024 ** 3
    assert placement_for(max(0.1, free_gib - 5.0), verbose=False) is None
    split = placement_for(free_gib + 50, verbose=False)
    assert split is not None and "cpu" in split


def test_calibration_text_is_long_enough():
    text = calibration_text(min_chars=5000)
    assert len(text) >= 5000


def _synthetic_profile():
    curve = _curve()
    layer = {
        "layer": 0,
        "knee_budget_frac": 0.12,
        "knee_retention": 0.55,
        "retention_at": {p: {str(f): 0.5 for f in BUDGET_FRACTIONS}
                         for p in ("h2o", "window", "sink")},
        "curve": {p: _downsample(curve) for p in ("h2o", "window", "sink")},
    }
    return {
        "schema": "kv-retention-profile/1", "model": "fake/model",
        "generated": "2026-09-16T00:00:00",
        "architecture": {"layers": 1, "attention_heads": 2, "kv_heads": 1, "hidden_size": 8},
        "calibration": {"tokens": 64, "windows": 1, "chunk_size": 32,
                        "dtype": "bfloat16", "text_source": "wikitext-2", "seconds": 1.0},
        "budget_fractions": BUDGET_FRACTIONS, "curve_points": CURVE_POINTS,
        "note": "measurements only", "layers": [layer],
    }


def test_write_profile_round_trips(tmp_path):
    profile = _synthetic_profile()
    path = write_profile(profile, str(tmp_path))
    assert path.endswith(".json")
    back = json.load(open(path))
    assert back["schema"] == "kv-retention-profile/1"
    assert len(back["layers"][0]["curve"]["h2o"]) == CURVE_POINTS


def test_print_summary_runs_on_a_single_layer_profile(capsys):
    print_summary(_synthetic_profile())
    out = capsys.readouterr().out
    assert "KV RETENTION PROFILE" in out
    assert "measurements only" in out, "the proxy caveat must survive into the summary"
