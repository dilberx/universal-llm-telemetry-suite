"""Incremental per-key mass accumulation.

`curves.py` reduces a layer's attention tensor to a `[keys]` mass vector and
derives every retention curve from it. That reduction is cheap, but the caller
still has to produce the attention tensor first, and the usual way to do that
holds far more memory than the reduction needs.

Two separate costs are involved:

1. `output_attentions=True` makes the model keep one attention tensor per
   layer, so peak memory is `L * H * S^2`. Nothing needs all layers at once.
   Reducing each layer to `[S]` the moment it is produced takes peak to
   `H * S^2 + L * S`.

2. A single layer's `H * S^2` still dominates at long context. Causal attention
   lets the prompt be fed in chunks with a KV cache, which is what chunked
   prefill does in a real serving stack. At chunk size `C` the largest
   attention tensor is `H * C * S`, and the mass contributions sum across
   chunks because each query row is computed exactly once either way.

Together these take peak attention memory from `L * H * S^2` to `H * C * S`.

Implementation note. This wraps `attn.forward` rather than registering a
forward hook. Transformers 4.57 captures attention for `output_attentions` by
monkey-patching layer forwards, which runs inside the module call and therefore
before any PyTorch forward hook, so a hook cannot stop the tensor being
retained. Wrapping forward puts the reduction at the same level. Under eager
attention the module returns its weights whether or not `output_attentions` is
set, so the collector never asks for the flag and the model never accumulates
anything.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field

import torch


def _attention_modules(model) -> list:
    """Attention submodules in layer order, for the common HF decoder layout."""
    # "layers" covers a bare decoder stack (e.g. LlamaModel), which is what
    # unwrapping a vision-language checkpoint hands back; the others cover the
    # usual ForCausalLM wrappers.
    for path in ("model.layers", "layers", "transformer.h", "model.decoder.layers",
                 "model.model.layers"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
        except AttributeError:
            continue
        mods = []
        for layer in obj:
            for name in ("self_attn", "attn", "attention"):
                if hasattr(layer, name):
                    mods.append(getattr(layer, name))
                    break
        if len(mods) == len(obj):
            return mods
    raise ValueError(
        f"could not locate attention modules on {type(model).__name__}; "
        "pass them explicitly"
    )


@dataclass
class MassCollector:
    """Accumulates per-key attention mass across layers without keeping attention.

    Use as a context manager around one or more forward passes. Mass from every
    pass inside the context accumulates, which is what makes chunked prefill
    work: each chunk contributes the query rows it computed.

    Entering replaces `forward` on each attention module and exiting restores
    it, so the model is unchanged outside the context.
    """
    model: object
    seq_len: int
    dtype: torch.dtype = torch.float64
    discard_attention: bool = True
    query_rate: float = 1.0
    query_seed: int = 0
    mass: torch.Tensor = field(init=False)
    _originals: list = field(default_factory=list, init=False)
    _peak_attn_bytes: int = field(default=0, init=False)
    _queries_seen: int = field(default=0, init=False)
    _queries_used: int = field(default=0, init=False)

    def __post_init__(self):
        self._modules = _attention_modules(self.model)
        self.mass = torch.zeros(len(self._modules), self.seq_len, dtype=self.dtype)

    @property
    def n_layers(self) -> int:
        return len(self._modules)

    @property
    def scaled_mass(self) -> torch.Tensor:
        """Mass corrected for query sampling.

        With `query_rate < 1` the accumulated sums cover only the sampled
        queries, so they are scaled by the realised inverse rate. Sampling is
        uniform over positions, so one global factor is correct for every key.
        """
        if self.query_rate >= 1.0 or self._queries_used == 0:
            return self.mass
        return self.mass * (self._queries_seen / self._queries_used)

    @property
    def realised_query_rate(self) -> float:
        return self._queries_used / self._queries_seen if self._queries_seen else 1.0

    @property
    def peak_attention_bytes(self) -> int:
        """Largest single attention tensor seen. The quantity chunking reduces."""
        return self._peak_attn_bytes

    def _make_wrapper(self, layer_idx: int, original):
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            output = original(*args, **kwargs)
            weights = None
            if isinstance(output, (tuple, list)) and len(output) >= 2:
                cand = output[1]
                if torch.is_tensor(cand) and cand.dim() == 4:
                    weights = cand
            if weights is None:
                return output

            self._peak_attn_bytes = max(
                self._peak_attn_bytes, weights.numel() * weights.element_size()
            )
            n_keys = weights.shape[-1]
            n_queries = weights.shape[-2]

            # [batch, heads, queries, keys] -> [keys]
            if self.query_rate >= 1.0:
                contrib = weights.detach().to(self.dtype).sum(dim=(0, 1, 2)).cpu()
                used = n_queries
            else:
                # Sample query rows. Every key's mass is a sum over the queries
                # that attend to it, so dropping a uniform random share of
                # queries and scaling by the inverse rate is unbiased. This is
                # the axis that reduces work: attention costs H * S^2 in the
                # queries, while the mass vector it produces is only S long.
                # Positions are absolute, so a key is sampled consistently no
                # matter which chunk its queries land in.
                q_start = n_keys - n_queries
                pos = torch.arange(q_start, q_start + n_queries, dtype=torch.int64)
                h = (pos * 2654435761 + self.query_seed * 40503) % 2147483647
                keep = (h.to(torch.float64) / 2147483647.0) < self.query_rate
                used = int(keep.sum())
                if used == 0:
                    self._queries_seen += n_queries
                    return (output[0], None) + tuple(output[2:]) if self.discard_attention else output
                sel = weights.detach()[:, :, keep.to(weights.device), :]
                contrib = sel.to(self.dtype).sum(dim=(0, 1, 2)).cpu()
            if layer_idx == 0:
                self._queries_seen += n_queries
                self._queries_used += used
            if n_keys > self.seq_len:
                raise ValueError(
                    f"layer {layer_idx} produced {n_keys} keys but collector was "
                    f"sized for {self.seq_len}"
                )
            self.mass[layer_idx, :n_keys] += contrib

            # Drop the tensor here so it is freed as this layer returns instead
            # of being held to the end of the forward pass.
            if self.discard_attention:
                return (output[0], None) + tuple(output[2:])
            return output
        return wrapped

    def __enter__(self) -> "MassCollector":
        for i, mod in enumerate(self._modules):
            self._originals.append((mod, mod.forward))
            mod.forward = self._make_wrapper(i, mod.forward)
        return self

    def __exit__(self, *exc):
        for mod, original in self._originals:
            mod.forward = original
        self._originals.clear()
        return False


@torch.no_grad()
def collect_key_mass(model, input_ids: torch.Tensor, chunk_size: int | None = None,
                     query_rate: float = 1.0, query_seed: int = 0) -> torch.Tensor:
    """Per-key attention mass for every layer, without holding attention matrices.

    Args:
        model: a causal HF decoder with eager attention.
        input_ids: [1, seq_len] prompt.
        chunk_size: feed the prompt in chunks of this many tokens using a KV
            cache. None runs a single forward pass. Smaller chunks lower peak
            memory and do not change the result.
        query_rate: fraction of query rows to use. 1.0 is exact. Lower values
            estimate the mass from a sample, which is what makes the attention
            cost fall, at the price of variance in the resulting curve.
        query_seed: seed for the query sample, for repeatability.

    Returns:
        [n_layers, seq_len] float64 mass, identical to summing each layer's
        full attention tensor over batch, heads and queries.
    """
    if input_ids.dim() != 2 or input_ids.shape[0] != 1:
        raise ValueError(f"expected [1, seq_len] input_ids, got {tuple(input_ids.shape)}")
    seq_len = input_ids.shape[1]

    with MassCollector(model, seq_len, query_rate=query_rate, query_seed=query_seed) as collector:
        if chunk_size is None or chunk_size >= seq_len:
            model(input_ids=input_ids, use_cache=False)
        else:
            past = None
            for start in range(0, seq_len, chunk_size):
                chunk = input_ids[:, start:start + chunk_size]
                out = model(
                    input_ids=chunk,
                    past_key_values=past,
                    use_cache=True,
                )
                past = out.past_key_values
        return collector.scaled_mass.clone()


@torch.no_grad()
def reference_key_mass(model, input_ids: torch.Tensor) -> torch.Tensor:
    """The incumbent path: materialize every layer's attention, then reduce.

    Kept as the correctness reference for `collect_key_mass`. Peak memory is
    `L * H * S^2`, which is the cost the collector exists to avoid.
    """
    out = model(input_ids=input_ids, output_attentions=True, use_cache=False)
    return torch.stack([a.detach().to(torch.float64).sum(dim=(0, 1, 2)).cpu()
                        for a in out.attentions])
