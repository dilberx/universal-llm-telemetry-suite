"""Tests for decode-time block mass and the expiry rule.

The controller tests use synthetic masses so the rule's arithmetic is checked in
isolation. The model tests build a tiny random Qwen2 on CPU, because the failure
that matters here is a mask that is passed but never takes effect, and only a
real forward pass can show that.
"""

from __future__ import annotations

import pytest
import torch

from src.kv_profiler.block_mass import (
    DecodeMassRecorder,
    ExpiryController,
    block_sum,
    dead_fractions,
    decode,
    prefill,
    theta_for_literal,
)


# --- aggregation -------------------------------------------------------------

def test_block_sum_groups_tokens_and_pads_the_tail():
    m = torch.arange(10, dtype=torch.float32)            # 0..9
    got = block_sum(m, block_size=4)
    assert got.tolist() == [0 + 1 + 2 + 3, 4 + 5 + 6 + 7, 8 + 9]


def test_block_sum_keeps_leading_dimensions():
    m = torch.ones(3, 8)
    assert block_sum(m, block_size=4).shape == (3, 2)
    assert torch.allclose(block_sum(m, 4), torch.full((3, 2), 4.0))


def test_theta_for_literal_scales_by_heads_and_layers():
    assert theta_for_literal(1e-3, n_heads=14, n_layers=24) == pytest.approx(1e-3 / 336)
    assert theta_for_literal(1e-3, n_heads=14, n_layers=24, sum_layers=False) == pytest.approx(1e-3 / 14)


# --- controller --------------------------------------------------------------

def _ctrl(**kw):
    kw.setdefault("capacity", 16)
    kw.setdefault("theta", 0.1)
    return ExpiryController(**kw)


def test_block_expires_only_after_window_consecutive_low_reads():
    c = _ctrl(window=3, alpha=1.0)                       # alpha=1 -> EMA is the raw mass
    mass = torch.tensor([0.9, 0.01, 0.09])               # block 1 is low, block 2 is open
    assert c.update(mass).numel() == 0
    assert c.update(mass).numel() == 0
    assert c.update(mass).tolist() == [1]
    assert bool(c.expired[1]) and not bool(c.expired[0])


def test_one_high_read_resets_the_window():
    c = _ctrl(window=3, alpha=1.0)
    low, high = torch.tensor([0.9, 0.01, 0.09]), torch.tensor([0.5, 0.4, 0.1])
    c.update(low); c.update(low)
    c.update(high)                                       # mass returns: counter resets
    assert c.update(low).numel() == 0
    assert c.update(low).numel() == 0
    assert c.update(low).tolist() == [1]


def test_the_open_block_is_never_expired():
    c = _ctrl(window=1, alpha=1.0)
    mass = torch.tensor([0.5, 0.5, 0.0])                 # last block has no mass yet
    assert c.update(mass).numel() == 0
    assert not bool(c.expired[2])
    # Once another block opens, the former open block becomes eligible.
    assert c.update(torch.tensor([0.5, 0.5, 0.0, 0.0])).tolist() == [2]


def test_expired_blocks_stay_expired_and_are_not_reported_twice():
    c = _ctrl(window=1, alpha=1.0)
    mass = torch.tensor([0.9, 0.01, 0.09])
    assert c.update(mass).tolist() == [1]
    assert c.update(mass).numel() == 0
    assert bool(c.expired[1])


def test_ema_smooths_a_single_dip():
    c = _ctrl(window=1, alpha=0.5, theta=0.1)
    c.update(torch.tensor([0.9, 0.5, 0.0]))              # seeds block 1 at 0.5
    c.update(torch.tensor([0.9, 0.0, 0.0]))              # raw 0.0, EMA 0.25: still above theta
    assert not bool(c.expired[1])
    assert c.ema[1].item() == pytest.approx(0.25)


def test_first_read_seeds_the_ema_instead_of_blending_with_zero():
    c = _ctrl(alpha=0.9)
    c.update(torch.tensor([0.8, 0.2, 0.0]))
    assert c.ema[0].item() == pytest.approx(0.8)


def test_new_blocks_arriving_mid_run_are_tracked_from_their_first_read():
    c = _ctrl(window=2, alpha=1.0)
    c.update(torch.tensor([0.6, 0.4]))
    c.update(torch.tensor([0.6, 0.4, 0.0]))
    assert bool(c.seen[2])
    assert c.low_reads[2].item() == 0                    # was the open block, never counted


def test_capacity_is_enforced():
    with pytest.raises(ValueError):
        _ctrl(capacity=2).update(torch.ones(3))


# --- dead fractions ----------------------------------------------------------

def test_dead_fractions_separates_pooled_from_per_layer():
    # Two layers, five blocks (the last is excluded as still open).
    # Block 1 is dead in layer 0 only, block 2 is dead in both.
    m = torch.tensor([[0.50, 0.00, 0.00, 0.50, 0.0],
                      [0.30, 0.30, 0.00, 0.40, 0.0]])
    r = dead_fractions(m, [0.05])[0.05]
    assert r["all_layers"] == pytest.approx(1 / 4)       # only block 2 is dead everywhere
    assert r["layer_min"] == pytest.approx(1 / 4)        # layer 1 alone: block 2
    assert r["layer_max"] == pytest.approx(2 / 4)        # layer 0 alone: blocks 1, 2
    assert r["pooled"] == pytest.approx(1 / 4)           # mean mass: block 2 only


def test_dead_fractions_rise_with_theta():
    torch.manual_seed(0)
    m = torch.softmax(torch.randn(4, 64) * 3, dim=-1)
    r = dead_fractions(m, [1e-4, 1e-3, 1e-2, 1e-1])
    pooled = [r[t]["pooled"] for t in sorted(r)]
    assert pooled == sorted(pooled)


# --- the real masking path, on a tiny model -----------------------------------

@pytest.fixture(scope="module")
def tiny():
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    cfg = transformers.Qwen2Config(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512)
    model = transformers.Qwen2ForCausalLM._from_config(cfg, attn_implementation="eager")
    return model.eval()


def _run(tiny, mode, theta=0.0, **kw):
    n_ctx, n_steps = 48, 24
    tokens = torch.randint(0, 64, (n_ctx + n_steps + 2,), generator=torch.Generator().manual_seed(1))
    with DecodeMassRecorder(tiny, max_len=n_ctx + n_steps + 8) as rec:
        base = prefill(tiny, tokens[:n_ctx].unsqueeze(0), chunk=16)
        full = decode(tiny, base, tokens, n_ctx, n_steps, mode="full", recorder=rec,
                      block_size=4, interval=4)
        out = decode(tiny, base, tokens, n_ctx, n_steps, mode=mode, theta=theta, recorder=rec,
                     reference_logprobs=full["logprobs"], block_size=4, interval=4,
                     window=1, **kw)
    return full, out


def test_all_ones_mask_is_identical_to_no_mask(tiny):
    """theta=0 expires nothing, so this is the noise-floor control."""
    _, out = _run(tiny, "mass", theta=0.0)
    assert max(out["kl"]) < 1e-6
    assert min(out["top1"]) == 1.0
    assert out["expired_count"][-1] == 0


def test_expiry_changes_the_output_and_zeroes_the_expired_blocks(tiny):
    """With a huge theta everything but the open block expires. The mask must
    both change the logits and leave the expired positions with exactly zero
    attention mass, which is what shows the mask took effect inside attention."""
    _, out = _run(tiny, "mass", theta=10.0)
    assert out["expired_count"][-1] > 0
    assert out["first_expiry_step"] is not None
    assert max(out["kl"]) > 1e-4
    assert out["max_expired_mass"] == 0.0


def test_recorder_ignores_prefill_and_restores_forward(tiny):
    originals = [m.forward for m in __import__("src.kv_profiler.collect", fromlist=["x"])._attention_modules(tiny)]
    with DecodeMassRecorder(tiny, max_len=64) as rec:
        prefill(tiny, torch.randint(0, 64, (1, 32)), chunk=16)
        assert rec.last == {}, "prefill must not be recorded"
    after = [m.forward for m in __import__("src.kv_profiler.collect", fromlist=["x"])._attention_modules(tiny)]
    assert all(a == b for a, b in zip(originals, after))


def test_decode_mass_sums_to_head_count_per_layer(tiny):
    """Each head's softmax sums to one over keys, so head-summed mass over all
    tokens equals n_heads. This pins the normalisation the thresholds rely on."""
    n_ctx = 24
    tokens = torch.randint(0, 64, (n_ctx + 3,))
    with DecodeMassRecorder(tiny, max_len=64) as rec:
        base = prefill(tiny, tokens[:n_ctx].unsqueeze(0), chunk=8)
        decode(tiny, base, tokens, n_ctx, 1, mode="full", recorder=rec, block_size=4, interval=100)
        last = rec.last_step_mass()
    assert torch.allclose(last.sum(-1), torch.full((2,), 4.0), atol=1e-4)


def test_random_arm_expires_the_requested_number_of_blocks(tiny):
    _, mass = _run(tiny, "mass", theta=10.0)
    _, rand = _run(tiny, "random", random_counts=mass["new_per_read"])
    assert rand["expired_count"][-1] == mass["expired_count"][-1]
    assert rand["max_expired_mass"] == 0.0


# --- analysis helpers ---------------------------------------------------------

def test_interp_at_interpolates_between_bracketing_points():
    from src.experiments.analyze_block_expiry import interp_at
    pts = [(0.1, 0.0), (0.5, 0.4), (0.9, 1.2)]
    assert interp_at(pts, 0.3) == pytest.approx(0.2)
    assert interp_at(pts, 0.7) == pytest.approx(0.8)


def test_interp_at_refuses_to_extrapolate():
    from src.experiments.analyze_block_expiry import interp_at
    pts = [(0.2, 0.1), (0.5, 0.3)]
    assert interp_at(pts, 0.1) is None
    assert interp_at(pts, 0.9) is None


def test_interp_at_handles_an_exact_hit_and_unsorted_input():
    from src.experiments.analyze_block_expiry import interp_at
    assert interp_at([(0.9, 2.0), (0.3, 1.0), (0.6, 1.5)], 0.6) == pytest.approx(1.5)


# --- controls that ignore attention --------------------------------------------

def _controls(tiny, mode):
    _, mass = _run(tiny, "mass", theta=10.0, snapshot_at=24)
    _, ctl = _run(tiny, mode, random_counts=mass["new_per_read"], snapshot_at=24)
    return mass, ctl


@pytest.mark.parametrize("mode", ["random_nosink", "oldest"])
def test_attention_blind_controls_match_the_expiry_count_and_spare_the_sink(tiny, mode):
    mass, ctl = _controls(tiny, mode)
    # theta=10 is above every mass, so the mass arm expires even block 0. In real
    # runs the sink holds a large share of the mass and is never below threshold.
    # The controls spare block 0 by design, so they come up short by exactly that one.
    sink_spent = int(bool(mass["snapshot"][0]))
    assert ctl["expired_count"][-1] == mass["expired_count"][-1] - sink_spent
    assert not bool(ctl["snapshot"][0]), "block 0 holds the attention sink and must survive"
    assert ctl["max_expired_mass"] == 0.0


def test_oldest_expires_the_lowest_indexed_blocks_after_the_sink(tiny):
    _, ctl = _controls(tiny, "oldest")
    gone = torch.nonzero(ctl["snapshot"]).flatten().tolist()
    assert gone == list(range(1, 1 + len(gone))), f"expected a contiguous run from block 1, got {gone}"


def test_uniform_random_can_remove_the_sink_block(tiny):
    """That is why it is a weak control: across seeds it eventually hits block 0."""
    _, mass = _run(tiny, "mass", theta=10.0)
    hit = False
    for seed in range(40):
        _, ctl = _run(tiny, "random", random_counts=mass["new_per_read"], snapshot_at=24, seed=seed)
        hit |= bool(ctl["snapshot"][0])
    assert hit


def test_decode_reports_whether_each_prediction_was_correct(tiny):
    full, out = _run(tiny, "mass", theta=0.0)
    assert len(out["hits"]) == 24 and set(out["hits"]) <= {0.0, 1.0}
    assert out["hits"] == full["hits"], "with nothing expired both arms must predict identically"


# --- memory cap ----------------------------------------------------------------

def test_limit_gpu_memory_caps_below_free_memory_and_can_be_undone():
    from src.kv_profiler.loading import limit_gpu_memory
    if not torch.cuda.is_available():
        assert limit_gpu_memory(verbose=False) is None
        return
    free_gib = torch.cuda.mem_get_info()[0] / 1024 ** 3
    try:
        cap = limit_gpu_memory(headroom_gib=1.0, verbose=False)
        assert cap == pytest.approx(max(1.0, free_gib - 1.0), abs=0.5)
        assert cap < free_gib
    finally:
        torch.cuda.set_per_process_memory_fraction(1.0)


# --- per-window result files ----------------------------------------------------

def _fake_result(tmp_path, name, n_trials):
    import json as _json
    (tmp_path / name).write_text(_json.dumps({"trials": [{"config": {"part": "B"}} for _ in range(n_trials)]}))


def test_load_runs_merges_windows_and_prefers_the_newest_file(tmp_path):
    from src.experiments.analyze_block_expiry import load_runs
    _fake_result(tmp_path, "block_expiry_q05b_n2048_w0_20261021_100000.json", 3)
    _fake_result(tmp_path, "block_expiry_q05b_n2048_w1_20261021_100500.json", 4)
    _fake_result(tmp_path, "block_expiry_q05b_n2048_w1_20261021_110000.json", 5)   # rerun of w1
    _fake_result(tmp_path, "block_expiry_q05b_n8192_w0_20261021_101000.json", 2)
    _fake_result(tmp_path, "block_expiry_q15b_n2048_w0_20261021_101000.json", 9)   # other tag
    runs = load_runs("q05b", str(tmp_path))
    assert list(runs) == [2048, 8192]
    assert len(runs[2048]["trials"]) == 3 + 5, "w1 must come from the newer file only"
    assert len(runs[8192]["trials"]) == 2


def test_load_runs_skips_empty_result_files(tmp_path):
    from src.experiments.analyze_block_expiry import load_runs
    _fake_result(tmp_path, "block_expiry_q05b_n4096_w0_20261021_100000.json", 0)
    assert load_runs("q05b", str(tmp_path)) == {}
