"""Decode-time block attention mass and a block-expiry rule, for measurement.

vLLM RFC #39076 proposed freeing KV blocks from a running request when their
attention mass stays low during decode. That RFC reports no measurements, so
the premise (that a large share of a long context is "dead") is untested. This
module makes the premise testable on a Hugging Face model, with the exact
softmax weights standing in for whatever a serving kernel would report.

The rule follows the RFC's text. Per decode step, a block's mass is the sum of
softmax weights over the block's tokens and over heads. Every K decode steps the
mass is read, smoothed with an EMA (alpha), and a block is marked expired when
the EMA stays below theta for W consecutive reads. Expired blocks are masked out
of attention from then on. Defaults are the RFC's: K=32, alpha=0.9, W=3, and
blocks of 16 tokens.

Three things the RFC does not pin down, and how this module handles each:

* Scale. "Sum over heads" leaves open whether layers are also summed, which is a
  factor of up to n_heads * n_layers on the threshold. Masses here are
  normalised to a head-and-layer mean, so they sum to 1 over a request's blocks,
  and the caller chooses theta in those units. `theta_for_literal` converts.
* Layers. The RFC's buffer has no layer dimension, so one decision applies to a
  whole block across layers. That is also what vLLM can physically do, since a
  block holds all layers of a KV group. Per-layer masses are kept so the cost of
  that pooling can be measured.
* The open block. The block still being filled is never expired.

Nothing here frees memory. Masking is applied, so the quality effect is real,
and the memory saving is read off the fraction of blocks expired.
"""

from __future__ import annotations

import copy
import functools
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from src.kv_profiler.collect import _attention_modules

BLOCK_SIZE = 16


def block_sum(token_mass: torch.Tensor, block_size: int = BLOCK_SIZE) -> torch.Tensor:
    """[..., S] -> [..., ceil(S / block_size)], summing inside each block."""
    pad = (-token_mass.shape[-1]) % block_size
    if pad:
        token_mass = F.pad(token_mass, (0, pad))
    return token_mass.reshape(*token_mass.shape[:-1], -1, block_size).sum(-1)


def theta_for_literal(theta_literal: float, n_heads: int, n_layers: int,
                      sum_layers: bool = True) -> float:
    """Convert a threshold on a head-summed (and optionally layer-summed) mass
    into the head-and-layer-mean units used here."""
    return theta_literal / (n_heads * (n_layers if sum_layers else 1))


class DecodeMassRecorder:
    """Records per-token attention mass from decode steps, one layer at a time.

    Only forwards with a single query token are recorded, so prefill is ignored.
    The module wrapper drops the weights tensor after reading it, as the profiler
    collector does, so nothing accumulates inside the model.
    """

    def __init__(self, model, max_len: int):
        self._modules = _attention_modules(model)
        self.n_layers = len(self._modules)
        self.max_len = max_len
        self.n_heads: int | None = None
        self.active = False
        self.last: dict[int, torch.Tensor] = {}   # head-summed mass, latest step
        self.acc: dict[int, torch.Tensor] = {}    # head-summed mass, since reset
        self.steps = 0
        self._originals: list = []

    def _wrap(self, layer: int, original):
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            out = original(*args, **kwargs)
            if not self.active or not isinstance(out, (tuple, list)) or len(out) < 2:
                return out
            w = out[1]
            if w is None or not torch.is_tensor(w) or w.dim() != 4 or w.shape[-2] != 1:
                return out
            S = w.shape[-1]
            if S > self.max_len:
                raise ValueError(f"layer {layer}: {S} keys exceed recorder size {self.max_len}")
            if layer not in self.last:
                self.last[layer] = torch.zeros(self.max_len, dtype=torch.float32, device=w.device)
                self.acc[layer] = torch.zeros(self.max_len, dtype=torch.float32, device=w.device)
            self.n_heads = w.shape[1]
            step = w[0, :, 0, :].to(torch.float32).sum(0)      # [S], summed over heads
            self.last[layer].zero_()
            self.last[layer][:S] = step
            self.acc[layer][:S] += step
            return (out[0], None) + tuple(out[2:])
        return wrapped

    def __enter__(self):
        for i, mod in enumerate(self._modules):
            self._originals.append((mod, mod.forward))
            mod.forward = self._wrap(i, mod.forward)
        return self

    def __exit__(self, *exc):
        for mod, original in self._originals:
            mod.forward = original
        self._originals.clear()
        return False

    def count_step(self) -> None:
        self.steps += 1

    def interval_mass(self, n_steps: int) -> torch.Tensor:
        """[layers, max_len] mean token mass over the interval, as a head mean."""
        if self.n_heads is None or not self.acc:
            raise RuntimeError("no decode step has been recorded")
        rows = [self.acc[l] for l in range(self.n_layers)]
        return torch.stack(rows) / (n_steps * self.n_heads)

    def reset_interval(self) -> None:
        for t in self.acc.values():
            t.zero_()
        self.steps = 0

    def last_step_mass(self) -> torch.Tensor:
        """[layers, max_len] head-summed mass of the most recent step."""
        return torch.stack([self.last[l] for l in range(self.n_layers)])


@dataclass
class ExpiryController:
    """The RFC's EMA-and-window rule over block masses.

    `update` takes the pooled, normalised mass of every live block at a read and
    returns the indices that expired at this read. Expired blocks stay expired.
    """
    capacity: int
    theta: float
    alpha: float = 0.9
    window: int = 3
    ema: torch.Tensor = field(init=False)
    seen: torch.Tensor = field(init=False)
    low_reads: torch.Tensor = field(init=False)
    expired: torch.Tensor = field(init=False)

    def __post_init__(self):
        self.ema = torch.zeros(self.capacity, dtype=torch.float64)
        self.seen = torch.zeros(self.capacity, dtype=torch.bool)
        self.low_reads = torch.zeros(self.capacity, dtype=torch.long)
        self.expired = torch.zeros(self.capacity, dtype=torch.bool)

    def update(self, pooled_mass: torch.Tensor) -> torch.Tensor:
        n = pooled_mass.shape[0]
        if n > self.capacity:
            raise ValueError(f"{n} live blocks exceed capacity {self.capacity}")
        m = pooled_mass.detach().to("cpu", torch.float64)
        seen = self.seen[:n]
        prev = self.ema[:n]
        # A block's first read seeds its EMA rather than being blended with zero.
        self.ema[:n] = torch.where(seen, self.alpha * m + (1 - self.alpha) * prev, m)
        self.seen[:n] = True

        eligible = ~self.expired[:n]
        eligible[n - 1] = False                      # the open block is never expired
        low = (self.ema[:n] < self.theta) & eligible
        self.low_reads[:n] = torch.where(low, self.low_reads[:n] + 1,
                                         torch.zeros_like(self.low_reads[:n]))
        newly = low & (self.low_reads[:n] >= self.window)
        self.expired[:n] |= newly
        return torch.nonzero(newly).flatten()


def dead_fractions(layer_block_mass: torch.Tensor, thetas: list[float]) -> dict:
    """How much of a request's blocks fall below each threshold.

    Args:
        layer_block_mass: [layers, blocks] head-mean mass, each layer summing to 1.
        thetas: thresholds in the same units.

    Returns per theta: the fraction of blocks below it when decided on the
    layer-pooled mass (what a block-level rule can act on), the fraction dead in
    every layer at once, and the min/median/max across layers when each layer is
    judged alone (what a per-layer rule could free if blocks were per-layer).
    The last block is excluded because it is still being filled.
    """
    m = layer_block_mass[:, :-1].to(torch.float64)
    pooled = m.mean(0)
    out = {}
    for th in thetas:
        per_layer = (m < th).to(torch.float64).mean(1)
        out[th] = {
            "pooled": float((pooled < th).to(torch.float64).mean()),
            "all_layers": float((m < th).all(0).to(torch.float64).mean()),
            "layer_min": float(per_layer.min()),
            "layer_median": float(per_layer.median()),
            "layer_max": float(per_layer.max()),
        }
    return out


@torch.no_grad()
def prefill(model, ids: torch.Tensor, chunk: int = 256):
    """Chunked prefill. Returns the KV cache after the whole context."""
    past = None
    for s in range(0, ids.shape[1], chunk):
        out = model(input_ids=ids[:, s:s + chunk], past_key_values=past, use_cache=True)
        past = out.past_key_values
    return past


@torch.no_grad()
def decode(model, base_cache, tokens: torch.Tensor, n_ctx: int, n_steps: int, *,
           mode: str, recorder: DecodeMassRecorder, theta: float = 0.0,
           reference_logprobs: torch.Tensor | None = None,
           random_counts: list[int] | None = None, seed: int = 0,
           block_size: int = BLOCK_SIZE, interval: int = 32,
           alpha: float = 0.9, window: int = 3,
           snapshot_at: int | None = None) -> dict:
    """Teacher-forced decode of `tokens[n_ctx : n_ctx + n_steps + 1]`.

    Step j feeds token n_ctx + j and scores token n_ctx + j + 1, so every step
    has a target and the two arms being compared see identical inputs.

    mode "full"          no mask; stores log-probs for use as a reference.
    mode "mass"          the RFC's rule at `theta`; the mask is always passed, so
                         theta=0 (nothing expires) is the noise-floor control.
    The remaining modes expire the same number of blocks at each read that a
    "mass" run did (`random_counts`, one entry per read) but choose them by a
    rule that does not look at attention, so they show what the mass signal is
    worth:
    mode "random"        uniformly random live blocks. Weak control: it removes
                         the attention-sink block half the time, which is
                         catastrophic for any model and says nothing about mass.
    mode "random_nosink" random live blocks, never the first block.
    mode "oldest"        the oldest live blocks after the first, i.e. sink plus
                         a recent window of the same footprint.

    `snapshot_at` returns a copy of the expired set after that many steps.
    """
    cache = copy.deepcopy(base_cache)
    device = tokens.device
    total = n_ctx + n_steps
    capacity = (total + block_size) // block_size + 1
    ctrl = ExpiryController(capacity, theta, alpha, window)
    keep = torch.ones(total + 1, dtype=torch.bool, device=device)
    gen = torch.Generator().manual_seed(seed)

    logprobs, nll, kl, top1, hits = [], [], [], [], []
    snapshot = None
    expired_count, live_blocks = [], []
    reads = []                      # per read: layer x block mass, for the "full" arm
    new_per_read, max_expired_mass, first_expiry = [], 0.0, None
    read_idx = 0

    recorder.reset_interval()
    recorder.active = True
    try:
        for j in range(n_steps):
            seq_len = n_ctx + j + 1
            mask = None if mode == "full" else keep[:seq_len].to(torch.long).unsqueeze(0)
            out = model(input_ids=tokens[n_ctx + j].view(1, 1), past_key_values=cache,
                        attention_mask=mask, use_cache=True)
            recorder.count_step()
            lp = F.log_softmax(out.logits[0, -1].float(), dim=-1)
            target = tokens[n_ctx + j + 1]
            nll.append(float(-lp[target]))
            hits.append(float(lp.argmax() == target))
            if mode == "full":
                logprobs.append(lp)
            elif reference_logprobs is not None:
                ref = reference_logprobs[j]
                kl.append(float((ref.exp() * (ref - lp)).sum()))
                top1.append(float(ref.argmax() == lp.argmax()))

            if mode != "full" and ctrl.expired.any():
                dead_tok = ~keep[:seq_len]
                if bool(dead_tok.any()):
                    last = recorder.last_step_mass()[:, :seq_len].to(device)
                    max_expired_mass = max(max_expired_mass,
                                           float(last[:, dead_tok].max()))

            expired_count.append(int(ctrl.expired.sum()))
            live_blocks.append((seq_len + block_size - 1) // block_size)

            if (j + 1) % interval == 0:
                n_live = (seq_len + block_size - 1) // block_size
                tok_mass = recorder.interval_mass(interval)[:, :seq_len]
                layer_blocks = block_sum(tok_mass, block_size)[:, :n_live].cpu()
                if mode == "full":
                    reads.append(layer_blocks)
                elif mode == "mass":
                    newly = ctrl.update(layer_blocks.mean(0))
                    new_per_read.append(int(newly.numel()))
                    for b in newly.tolist():
                        keep[b * block_size:(b + 1) * block_size] = False
                elif mode in ("random", "random_nosink", "oldest"):
                    want = random_counts[read_idx] if random_counts else 0
                    pick = torch.empty(0, dtype=torch.long)
                    if want:
                        pool = torch.nonzero(~ctrl.expired[:n_live - 1]).flatten()
                        if mode != "random":
                            pool = pool[pool != 0]               # never the sink block
                        if mode == "oldest":
                            pick = pool[:want]
                        else:
                            pick = pool[torch.randperm(pool.numel(), generator=gen)[:want]]
                        ctrl.expired[pick] = True
                        for b in pick.tolist():
                            keep[b * block_size:(b + 1) * block_size] = False
                    new_per_read.append(int(pick.numel()))
                if first_expiry is None and ctrl.expired.any():
                    first_expiry = j + 1
                read_idx += 1
                recorder.reset_interval()
            if snapshot_at is not None and j + 1 == snapshot_at:
                snapshot = ctrl.expired.clone()
    finally:
        recorder.active = False

    return {
        "logprobs": torch.stack(logprobs) if logprobs else None,
        "nll": nll, "kl": kl, "top1": top1, "hits": hits, "snapshot": snapshot,
        "expired_count": expired_count, "live_blocks": live_blocks,
        "reads": reads, "new_per_read": new_per_read,
        "first_expiry_step": first_expiry,
        "max_expired_mass": max_expired_mass,
    }
