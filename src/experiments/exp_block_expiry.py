"""Experiment 3A: how much of a long context is dead, and what does dropping it cost?

vLLM RFC #39076 proposes freeing KV blocks from a running request once their
attention mass stays low during decode. The RFC projects a 30 to 60 percent
smaller active KV footprint at 32K to 128K contexts and reports no measurements.
This runs the RFC's rule on real attention at contexts up to what the hardware
holds and reports three things.

Part A, concentration. With no masking, what fraction of a request's blocks fall
below a mass threshold, decided on the layer-pooled mass (what a block-level
rule can act on) and judged layer by layer (what a per-layer rule could free if
blocks were per-layer). Thresholds are given relative to the uniform share
1/blocks as well as absolutely, because a fixed absolute threshold means very
different things at 2K and 32K tokens.

Part B, cost on ordinary text. The same rule run closed-loop, with expired blocks
masked out of attention, scored against the full-cache run on identical
teacher-forced inputs: per-token KL divergence of the next-token distribution
(the metric the RFC plans to use), NLL change on the true next token, top-1
agreement. Controls: theta = 0 (nothing expires, the noise floor), and two rules
that expire the same number of blocks without looking at attention, so the mass
signal has something to beat: sink-protected random, and sink plus a recent
window. Uniform random expiry is deliberately not used: it removes the
attention-sink block about half the time, which wrecks any model and measures
nothing about mass.

Part C, dormant then needed. Part B's text is wikitext articles concatenated,
which is the best case for expiry, since earlier articles really are irrelevant
to later ones. The risk the RFC has to survive is a block that goes quiet and is
needed later. So a passage is placed in the context, the model decodes 160
unrelated tokens (long enough for its blocks to be judged dead), then copies the
passage back. If the mass rule expired the passage's blocks, the copy fails.

What this does not test: contexts past what fits here, models with multi-head
attention at long context (the available ones top out at 2K to 4K positions), or
an actual serving kernel. Masking is applied, so the quality effect is real, but
no memory is freed and the mass comes from exact softmax weights rather than a
kernel's reconstruction. Part B's text is not one long coherent document.
"""

from __future__ import annotations

import argparse
import gc
import time
from statistics import mean

import torch

from src.experiments.exp_family_scaling import _wikitext
from src.experiments.runner import ExperimentRunner
from src.kv_profiler.block_mass import (
    BLOCK_SIZE,
    DecodeMassRecorder,
    dead_fractions,
    decode,
    prefill,
    theta_for_literal,
)
from src.kv_profiler.loading import limit_gpu_memory, load_for_profiling

REL_THETAS = [0.03, 0.1, 0.3, 1.0]   # multiples of the uniform share 1/blocks
CONTROLS = ("random_nosink", "oldest")
N_STEPS = 256            # first expiry is possible at step 96 with K=32, W=3
INTERVAL = 32

# Part C
P_LEN = 128              # the passage that goes dormant, 8 blocks
FRESH = 160              # unrelated tokens decoded before the passage is copied back
COPY = 100               # tokens of the passage that are copied
PART_C_THETAS = ("rel_0.03", "rel_0.1", "rel_0.3", "literal_sum_heads")


def _concentration(pooled: torch.Tensor) -> dict[str, float]:
    """Fraction of blocks that hold 90 and 99 percent of the pooled mass."""
    srt = torch.sort(pooled[:-1].to(torch.float64), descending=True).values
    cum = torch.cumsum(srt, 0) / srt.sum()
    n = srt.numel()
    return {f"blocks_for_{int(p * 100)}pct": float((cum < p).sum() + 1) / n for p in (0.9, 0.99)}


def _arm_metrics(arm: dict, full: dict) -> dict[str, float]:
    """Summarise a masked arm over the steps where a mask was actually in force."""
    first = arm["first_expiry_step"]
    live = [max(1, b - 1) for b in arm["live_blocks"]]          # exclude the open block
    frac = [e / b for e, b in zip(arm["expired_count"], live)]
    out = {
        "expired_frac_end": frac[-1],
        "any_expiry": float(first is not None),
        "max_expired_mass": arm["max_expired_mass"],
    }
    if first is None:
        out.update(kl_mean=0.0, kl_max=0.0, nll_delta=0.0, top1_agree=1.0, acc_delta=0.0,
                   expired_frac_mean=0.0, masked_steps=0.0)
        return out
    idx = range(first, len(arm["kl"]))
    out.update(
        kl_mean=mean(arm["kl"][i] for i in idx),
        kl_max=max(arm["kl"][i] for i in idx),
        nll_delta=mean(arm["nll"][i] - full["nll"][i] for i in idx),
        top1_agree=mean(arm["top1"][i] for i in idx),
        acc_delta=mean(arm["hits"][i] - full["hits"][i] for i in idx),
        expired_frac_mean=mean(frac[i] for i in idx),
        masked_steps=float(len(idx)),
    )
    return out


def _thetas(n_ctx: int, n_heads: int, n_layers: int) -> dict[str, float]:
    nb0 = n_ctx // BLOCK_SIZE
    lit = {
        "literal_sum_heads_layers": theta_for_literal(1e-3, n_heads, n_layers, True),
        "literal_sum_heads": theta_for_literal(1e-3, n_heads, n_layers, False),
        "literal_as_normalised": 1e-3,
    }
    return {"none": 0.0} | {f"rel_{r:g}": r / nb0 for r in REL_THETAS} | lit


def _tick(label: str, t0: float) -> None:
    print(f"    . {label:<44}{time.perf_counter() - t0:>7.0f}s", flush=True)


def _part_ab(model, seg, n_ctx, n_steps, w, runner, model_name, thetas, short):
    t0 = time.perf_counter()
    nb0 = n_ctx // BLOCK_SIZE
    with DecodeMassRecorder(model, max_len=n_ctx + n_steps + 8) as rec:
        base = prefill(model, seg[:n_ctx].unsqueeze(0), chunk=256 if n_ctx > 8192 else 512)
        full = decode(model, base, seg, n_ctx, n_steps, mode="full", recorder=rec, interval=INTERVAL)
        ref = full["logprobs"]
        _tick(f"N={n_ctx} w{w} A/B: prefill + full arm", t0)

        # ---- Part A: concentration from the unmasked arm ---------------------------
        layer_blocks = full["reads"][-1]                          # [layers, blocks], last read
        nb = layer_blocks.shape[1]
        dead = dead_fractions(layer_blocks, sorted(set(thetas.values()) - {0.0}))
        conc = _concentration(layer_blocks.mean(0))
        with runner.trial({"model": model_name, "n_ctx": n_ctx, "window": w, "part": "A"}) as trial:
            trial.record("n_blocks_end", float(nb))
            trial.record("uniform_share", 1.0 / nb)
            trial.record("blocks_for_90pct", conc["blocks_for_90pct"])
            trial.record("blocks_for_99pct", conc["blocks_for_99pct"])
            trial.record("nll_full_mean", mean(full["nll"]))
            trial.record_meta("dead_by_theta", {
                name: {"theta": th, "theta_rel_uniform": th * nb, **dead[th]}
                for name, th in thetas.items() if th > 0})
            trial.record_meta("pooled_block_mass_sorted",
                              [round(float(x), 9) for x in
                               torch.sort(layer_blocks.mean(0)[:-1], descending=True).values[:64]])

        # ---- Part B: closed loop, with controls -------------------------------------
        summary = []
        for name, th in thetas.items():
            arm = decode(model, base, seg, n_ctx, n_steps, mode="mass", theta=th, recorder=rec,
                         reference_logprobs=ref, interval=INTERVAL)
            results = {"mass": _arm_metrics(arm, full)}
            if arm["first_expiry_step"] is not None:
                for kind in CONTROLS:
                    c = decode(model, base, seg, n_ctx, n_steps, mode=kind, recorder=rec,
                               reference_logprobs=ref, random_counts=arm["new_per_read"],
                               seed=w, interval=INTERVAL)
                    results[kind] = _arm_metrics(c, full)
            for kind, mm in results.items():
                cfgb = {"model": model_name, "n_ctx": n_ctx, "window": w, "part": "B",
                        "theta_name": name, "theta": th, "theta_rel_uniform": th * nb0, "arm": kind}
                with runner.trial(cfgb) as trial:
                    for k, v in mm.items():
                        trial.record(k, float(v))
                    if kind == "mass":
                        trial.record_meta("new_per_read", arm["new_per_read"])
            summary.append((name, th, results))
            _tick(f"N={n_ctx} w{w} A/B: {name} (+{len(results) - 1} controls)", t0)
            del arm
            gc.collect()

    print(f"\n[{short} N={n_ctx} window {w}] {time.perf_counter() - t0:.0f}s, {nb} blocks at end, "
          f"90% of mass in {conc['blocks_for_90pct']:.1%} of blocks, "
          f"99% in {conc['blocks_for_99pct']:.1%}", flush=True)
    print(f"  {'theta':<26}{'x unif':>8}{'freed':>8} | {'KL':>8}{'dNLL':>8}{'top1':>6}"
          f" | nosink-rand: {'KL':>7}{'dNLL':>7} | oldest: {'KL':>7}{'dNLL':>7}", flush=True)
    for name, th, r in summary:
        m = r["mass"]
        cells = []
        for kind in CONTROLS:
            c = r.get(kind)
            cells.append(f"{c['kl_mean']:>7.4f}{c['nll_delta']:>7.3f}" if c else f"{'-':>7}{'-':>7}")
        print(f"  {name:<26}{th * nb0:>8.3g}{m['expired_frac_end']:>8.1%} | {m['kl_mean']:>8.4f}"
              f"{m['nll_delta']:>8.3f}{m['top1_agree']:>6.2f} | nosink-rand: {cells[0]} | oldest: {cells[1]}",
              flush=True)
    del base, full, ref
    gc.collect()
    torch.cuda.empty_cache()


def _part_c(model, ids_all, cursor, n_ctx, w, runner, model_name, thetas, short):
    """Dormant then needed: a passage sits in the context, goes quiet, is copied back."""
    n_steps = FRESH + COPY
    need = n_ctx + FRESH + 2 + P_LEN
    ctx = ids_all[cursor:cursor + n_ctx].clone()
    fresh = ids_all[cursor + n_ctx:cursor + n_ctx + FRESH + 2]
    passage = ids_all[cursor + n_ctx + FRESH + 2:cursor + need]
    depth = (0.25, 0.5, 0.75)[w % 3]                          # how far into the context it sits
    p0 = (int(n_ctx * depth) // BLOCK_SIZE) * BLOCK_SIZE
    ctx[p0:p0 + P_LEN] = passage
    dev = next(model.parameters()).device
    seg = torch.cat([ctx, fresh[:FRESH], passage[:COPY + 2]]).to(dev)
    assert seg.numel() == n_ctx + n_steps + 2
    p_blocks = list(range(p0 // BLOCK_SIZE, (p0 + P_LEN - 1) // BLOCK_SIZE + 1))
    t0 = time.perf_counter()

    with DecodeMassRecorder(model, max_len=n_ctx + n_steps + 8) as rec:
        base = prefill(model, seg[:n_ctx].unsqueeze(0), chunk=256 if n_ctx > 8192 else 512)
        full = decode(model, base, seg, n_ctx, n_steps, mode="full", recorder=rec, interval=INTERVAL)
        ref = full["logprobs"]
        _tick(f"N={n_ctx} w{w} C: prefill + full arm", t0)
        copy_nll_full = mean(full["nll"][FRESH:])
        copy_acc_full = mean(full["hits"][FRESH:])
        rows = []
        for name in PART_C_THETAS:
            th = thetas[name]
            arm = decode(model, base, seg, n_ctx, n_steps, mode="mass", theta=th, recorder=rec,
                         reference_logprobs=ref, interval=INTERVAL, snapshot_at=FRESH)
            kinds = {"mass": arm}
            if arm["first_expiry_step"] is not None:
                kinds["oldest"] = decode(model, base, seg, n_ctx, n_steps, mode="oldest", recorder=rec,
                                         reference_logprobs=ref, random_counts=arm["new_per_read"],
                                         interval=INTERVAL, snapshot_at=FRESH)
            for kind, a in kinds.items():
                snap = a["snapshot"]
                p_gone = float(snap[p_blocks].float().mean()) if snap is not None else 0.0
                m = {
                    "copy_nll": mean(a["nll"][FRESH:]),
                    "copy_nll_delta": mean(a["nll"][FRESH:]) - copy_nll_full,
                    "copy_acc": mean(a["hits"][FRESH:]),
                    "copy_acc_delta": mean(a["hits"][FRESH:]) - copy_acc_full,
                    "passage_blocks_expired": p_gone,
                    "all_blocks_expired_at_copy_start": (
                        float(snap.sum()) / max(1, a["live_blocks"][FRESH - 1] - 1)
                        if snap is not None else 0.0),
                    "kl_before_copy": mean(a["kl"][:FRESH]) if a["kl"] else 0.0,
                }
                cfgc = {"model": model_name, "n_ctx": n_ctx, "window": w, "part": "C",
                        "theta_name": name, "theta": th, "arm": kind, "passage_depth": depth}
                with runner.trial(cfgc) as trial:
                    for k, v in m.items():
                        trial.record(k, float(v))
                    trial.record("copy_nll_full", copy_nll_full)
                    trial.record("copy_acc_full", copy_acc_full)
                rows.append((name, kind, m))
            _tick(f"N={n_ctx} w{w} C: {name}", t0)
            del arm
            gc.collect()

    print(f"\n[{short} N={n_ctx} dormant passage at {depth:.0%} depth] {time.perf_counter() - t0:.0f}s   "
          f"full-cache copy NLL {copy_nll_full:.3f}, copy accuracy {copy_acc_full:.2f}", flush=True)
    print(f"  {'theta':<26}{'arm':<9}{'passage blocks':>15}{'copy NLL':>10}{'dNLL':>8}{'copy acc':>10}", flush=True)
    for name, kind, m in rows:
        print(f"  {name:<26}{kind:<9}{m['passage_blocks_expired']:>15.0%}{m['copy_nll']:>10.3f}"
              f"{m['copy_nll_delta']:>8.3f}{m['copy_acc']:>10.2f}", flush=True)
    del base, full, ref
    gc.collect()
    torch.cuda.empty_cache()
    return need


def run_block_expiry(model_name: str, dtype: torch.dtype, lengths: list[int],
                     windows: dict[int, int], n_steps: int = N_STEPS,
                     tag: str | None = None, part_c: bool = True,
                     only_window: int | None = None) -> ExperimentRunner:
    from transformers import AutoTokenizer

    short = tag or model_name.rstrip("/").split("/")[-1].lower().replace(".", "_").replace("-", "_")
    runner = ExperimentRunner(
        name=f"block_expiry_{short}",
        description=(
            f"Block-level attention mass and the RFC 39076 expiry rule on {model_name}, "
            f"contexts {lengths}, {n_steps} teacher-forced decode steps, block size "
            f"{BLOCK_SIZE}, K={INTERVAL}, alpha=0.9, W=3. Dead-block fractions, closed-loop "
            "KL / NLL cost against the full cache with noise-floor, sink-protected random and "
            "sink-plus-recent controls, and a dormant-then-needed copy test."
        ),
    )

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    limit_gpu_memory()      # without this, 32K runs spill into system RAM under WSL
    model = load_for_profiling(model_name, dtype=dtype)
    cfg = model.config
    n_layers, n_heads = cfg.num_hidden_layers, cfg.num_attention_heads
    print(f"{model_name}: {n_layers} layers, {n_heads} heads, {cfg.num_key_value_heads} KV, "
          f"max_pos {cfg.max_position_embeddings}", flush=True)

    per_window = lambda n: (n + n_steps + 2) + ((n + FRESH + 2 + P_LEN) if part_c else 0)
    if only_window is None:
        need = sum(per_window(n) * windows.get(n, 1) for n in lengths)
    else:
        need = sum(per_window(n) * (only_window + 1) for n in lengths)
    ids_all = tok(_wikitext(min_chars=need * 6), return_tensors="pt").input_ids[0]
    if ids_all.numel() < need:
        raise ValueError(f"need {need} tokens of text, have {ids_all.numel()}")
    device = next(model.parameters()).device

    cursor = 0
    for n_ctx in lengths:
        if n_ctx + max(n_steps, FRESH + COPY) + 2 > cfg.max_position_embeddings:
            print(f"skip {n_ctx}: beyond max_position_embeddings", flush=True)
            continue
        thetas = _thetas(n_ctx, n_heads, n_layers)
        for w in range(windows.get(n_ctx, 1)):
            if only_window is not None and w != only_window:
                cursor += per_window(n_ctx)      # same text offsets as a full run
                continue
            seg = ids_all[cursor:cursor + n_ctx + n_steps + 2].to(device)
            cursor += n_ctx + n_steps + 2
            _part_ab(model, seg, n_ctx, n_steps, w, runner, model_name, thetas, short)
            if part_c:
                cursor += _part_c(model, ids_all, cursor, n_ctx, w, runner, model_name, thetas, short)

    runner.save()
    runner.to_csv()
    print(runner.report.summary())
    return runner


def _parse_lengths(s: str) -> dict[int, int]:
    """'2048:3,8192:2' -> {2048: 3, 8192: 2}; a bare '4096' means one window."""
    out = {}
    for part in s.split(","):
        n, _, w = part.partition(":")
        out[int(n)] = int(w) if w else 1
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    ap.add_argument("--lengths", default="2048:1", help="comma list of N[:windows]")
    ap.add_argument("--steps", type=int, default=N_STEPS)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--no-part-c", action="store_true")
    ap.add_argument("--window", type=int, default=None,
                    help="run only this window index; each window is its own result file")
    a = ap.parse_args()
    lens = _parse_lengths(a.lengths)
    run_block_expiry(a.model, getattr(torch, a.dtype), list(lens), lens, a.steps, a.tag,
                     part_c=not a.no_part_c, only_window=a.window)
