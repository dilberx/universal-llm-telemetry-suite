"""Single-pass retention curves for KV cache budgets.

Background
----------
The grid sweep in `exp_real_model.analyze_real_kv_cache` evaluates a policy by
building a keep-mask at one budget and summing the attention that lands on the
kept keys, then repeating for every budget in a grid. That is one pass over the
attention tensor per budget point, and it returns retention only at the points
someone chose in advance.

Storage and computer architecture solved the same shape of problem in the 1970s.
Mattson's stack algorithm computes a full miss-ratio curve across every cache
size in a single pass, because the quantity you need per reference (its stack
distance) is independent of the cache size you are asking about. SHARDS later
made that constant-space by hashing which references to track.

The same reduction applies here. For a layer's attention tensor A with shape
[heads, queries, keys], define the per-key mass

    mass[k] = A[:, :, k].sum()

Every policy in the grid sweep scores as a sum of mass over its keep set, so:

  * H2O keeps the top-b keys by cumulative attention, which is exactly the top-b
    by mass. Its retention curve over all budgets is the normalized cumulative
    sum of mass sorted descending. This is a Lorenz curve of the mass
    distribution.
  * Window keeps the last b keys, so its curve is the normalized suffix sum.
  * Sink+recent keeps a prefix and a suffix, so its curve comes from prefix and
    suffix sums in constant time per budget.

So one reduction from [heads, queries, keys] to [keys] yields the exact curve at
every integer budget, for every one of these policies, with no grid. The output
is finer than the grid (every budget, not five) and cheaper to produce.

This module computes those curves and the knee of each one. It deliberately
stops at the measurement. Turning curves into a deployed configuration is a
separate concern and lives outside this file.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RetentionCurve:
    """Retention as a function of budget, for one policy on one layer.

    `values[b]` is the fraction of total attention mass retained when the policy
    is given a budget of `b` keys. Index 0 is the empty cache, index seq_len is
    the full cache and is always 1.0.
    """
    policy: str
    layer: int
    values: torch.Tensor   # [seq_len + 1], non-decreasing for H2O, in [0, 1]
    seq_len: int

    def at_fraction(self, frac: float) -> float:
        """Retention at a budget expressed as a fraction of sequence length.

        Uses the same budget rounding as the grid sweep, so the two are
        directly comparable.
        """
        budget = max(1, int(self.seq_len * frac))
        return float(self.values[budget])

    def knee(self) -> tuple[int, float]:
        """Budget at the point of maximum curvature, and its retention.

        Uses maximum perpendicular distance from the chord joining the two
        endpoints of the curve. This is the standard elbow criterion and needs
        no threshold parameter.
        """
        n = self.seq_len
        if n < 3:
            return n, float(self.values[n])
        x = torch.arange(n + 1, dtype=torch.float64) / n
        y = self.values.to(torch.float64)
        # Chord from (0, y0) to (1, y1); distance is proportional to the
        # signed area of the triangle, so the normalising constant drops out.
        x0, y0, x1, y1 = x[0], y[0], x[-1], y[-1]
        dist = ((y1 - y0) * x - (x1 - x0) * y + x1 * y0 - y1 * x0).abs()
        idx = int(dist.argmax())
        return idx, float(self.values[idx])


def key_mass(attn: torch.Tensor) -> torch.Tensor:
    """Reduce an attention tensor to per-key attention mass.

    This is the only pass over the attention data. Everything downstream works
    on the [seq_len] vector this returns.

    Args:
        attn: [heads, queries, keys] or [batch, heads, queries, keys].

    Returns:
        [keys] float64 tensor of total attention received by each key.
    """
    if attn.dim() == 4:
        attn = attn[0]
    if attn.dim() != 3:
        raise ValueError(f"expected [heads, queries, keys], got shape {tuple(attn.shape)}")
    return attn.to(torch.float64).sum(dim=(0, 1))


def h2o_curve(mass: torch.Tensor, layer: int = 0) -> RetentionCurve:
    """Exact H2O retention at every budget: the Lorenz curve of the mass."""
    total = mass.sum()
    ordered = torch.sort(mass, descending=True).values
    cum = torch.cat([torch.zeros(1, dtype=mass.dtype), torch.cumsum(ordered, dim=0)])
    return RetentionCurve("h2o", layer, cum / total if total > 0 else cum, mass.numel())


def window_curve(mass: torch.Tensor, layer: int = 0) -> RetentionCurve:
    """Exact window retention at every budget: the suffix sum of the mass."""
    total = mass.sum()
    suffix = torch.cat([torch.zeros(1, dtype=mass.dtype),
                        torch.cumsum(mass.flip(0), dim=0)])
    return RetentionCurve("window", layer, suffix / total if total > 0 else suffix, mass.numel())


def sink_curve(mass: torch.Tensor, layer: int = 0, max_sink: int = 4) -> RetentionCurve:
    """Exact sink-plus-recent retention at every budget.

    Mirrors the grid sweep's split: `sink = min(max_sink, budget // 2)` tokens
    from the front and the remainder from the back, with the union taken when
    the two halves meet.
    """
    n = mass.numel()
    total = mass.sum()
    prefix = torch.cat([torch.zeros(1, dtype=mass.dtype), torch.cumsum(mass, dim=0)])
    suffix = torch.cat([torch.zeros(1, dtype=mass.dtype), torch.cumsum(mass.flip(0), dim=0)])

    out = torch.zeros(n + 1, dtype=mass.dtype)
    for b in range(n + 1):
        sink = min(max_sink, b // 2)
        recent = b - sink
        if sink + recent >= n:
            out[b] = total
        else:
            out[b] = prefix[sink] + suffix[recent]
    return RetentionCurve("sink", layer, out / total if total > 0 else out, n)


POLICY_BUILDERS = {"h2o": h2o_curve, "window": window_curve, "sink": sink_curve}


def layer_curves(attn: torch.Tensor, layer: int = 0,
                 policies: tuple[str, ...] = ("h2o", "window", "sink")) -> dict[str, RetentionCurve]:
    """All retention curves for one layer from a single reduction of its attention."""
    mass = key_mass(attn)
    return {p: POLICY_BUILDERS[p](mass, layer) for p in policies}


def sampled_key_mass(attn: torch.Tensor, rate: float, seed: int = 0) -> tuple[torch.Tensor, float]:
    """SHARDS-style spatial sampling of keys.

    Keeps each key with probability `rate` using a hash of its index, so the
    same key is included or excluded consistently across layers and calls. The
    curve built from the returned mass is an estimate of the full curve; budgets
    on it are in units of sampled keys and scale back by the realised rate.

    Returns the sampled mass vector and the realised sampling rate.
    """
    if not 0 < rate <= 1:
        raise ValueError("rate must be in (0, 1]")
    mass = key_mass(attn)
    n = mass.numel()
    if rate == 1.0:
        return mass, 1.0
    idx = torch.arange(n, dtype=torch.int64)
    # Cheap deterministic hash, mixed with the seed so runs are reproducible.
    h = (idx * 2654435761 + seed * 40503) % 2147483647
    keep = (h.to(torch.float64) / 2147483647.0) < rate
    if not bool(keep.any()):
        keep[torch.randint(n, (1,))] = True
    return mass[keep], float(keep.sum()) / n
