"""Chunked eager attention that never materialises the full [heads, query, key] weights.

The cache-policy experiments need two things from attention: the key importance summed over every query (H2O and
hybrid) and the weights of the last few queries (SnapKV and the variance scorer). Prefill with
`output_attentions=True` keeps all of it, which caps the context at about 2K tokens on a 0.5B and 1.3K on a 7B.
This computes the same softmax in query chunks, adds each chunk's column sums to a running total, returns only the
last `WINDOW` rows as the layer's attention weights, and gives the same attention output as eager attention.
"""

from __future__ import annotations

import torch
from torch import nn
from transformers import AttentionInterface
from transformers.masking_utils import AttentionMaskInterface, eager_mask

NAME = "stream_stats"
WINDOW = 32
CHUNK = 256

_state: dict = {"colsum": None, "fresh": False}


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    b, h, t, d = x.shape
    return x[:, :, None, :, :].expand(b, h, n_rep, t, d).reshape(b, h * n_rep, t, d)


def stream_attention_forward(module: nn.Module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    key_states = _repeat_kv(key, module.num_key_value_groups)
    value_states = _repeat_kv(value, module.num_key_value_groups)
    q_len, k_len = query.shape[2], key_states.shape[2]
    colsum = torch.zeros(k_len, dtype=torch.float32, device=query.device)
    outputs, tail = [], []
    for q0 in range(0, q_len, CHUNK):
        q1 = min(q0 + CHUNK, q_len)
        weights = torch.matmul(query[:, :, q0:q1], key_states.transpose(2, 3)) * scaling
        if attention_mask is not None:
            weights = weights + attention_mask[:, :, q0:q1, :k_len]
        weights = nn.functional.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)
        outputs.append(torch.matmul(weights, value_states))
        colsum += weights.float().sum(dim=(0, 1, 2))
        if q1 > q_len - WINDOW:
            tail.append(weights[:, :, max(q0, q_len - WINDOW) - q0:])
    if module.layer_idx == 0 or _state["colsum"] is None:
        _state["colsum"] = colsum
    else:
        _state["colsum"] = _state["colsum"] + colsum.to(_state["colsum"].device)
    _state["fresh"] = True
    return torch.cat(outputs, dim=2).transpose(1, 2).contiguous(), torch.cat(tail, dim=2)


def enable(model) -> None:
    """Switch `model` to chunked attention. Idempotent."""
    AttentionInterface.register(NAME, stream_attention_forward)
    AttentionMaskInterface.register(NAME, eager_mask)
    model.set_attn_implementation(NAME)


def consume_colsum() -> torch.Tensor | None:
    """Key importance summed over layers, heads and queries of the latest forward, or None if not streaming."""
    if not _state["fresh"]:
        return None
    _state["fresh"] = False
    return _state["colsum"].clone()
