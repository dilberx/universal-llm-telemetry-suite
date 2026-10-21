"""Model loading that adapts to whatever VRAM is free right now.

A fixed `device_map` is wrong on a shared GPU. Free memory moves by several
GiB depending on what else is running, so a placement chosen when a script was
written may not fit when it runs. These helpers decide placement at load time
from `torch.cuda.mem_get_info`, and fall back to more CPU offload if the
allocation fails anyway.
"""

from __future__ import annotations

import gc
import glob
import json
import os

import torch

# Activations, the KV cache and one layer's attention tensor. At 1024 tokens
# the largest of those is tens of MB, so this is deliberately generous.
HEADROOM_GIB = 1.5


def limit_gpu_memory(headroom_gib: float = 1.0, verbose: bool = True) -> float | None:
    """Cap this process's CUDA memory just below what is physically free.

    Without a cap, PyTorch's caching allocator can reserve more than the card
    holds. Under WSL the driver then spills the overflow into system memory
    instead of failing, and every step pages data across PCIe. Measured on a
    10 GiB card running a 0.5B model at 32K tokens: real peak allocation 3.8 GiB,
    reserved pool 10.6 GiB, prefill 350 s, decode 146 ms/step; after releasing the
    cache, decode was 46 ms/step. A cap makes the allocator free cached blocks and
    retry when it reaches the limit, which keeps the pool inside real VRAM.

    Returns the cap in GiB, or None when there is no GPU.
    """
    if not torch.cuda.is_available():
        return None
    free, total = torch.cuda.mem_get_info()
    cap = max(1.0, free / 1024 ** 3 - headroom_gib)
    torch.cuda.set_per_process_memory_fraction(min(1.0, cap * 1024 ** 3 / total))
    if verbose:
        print(f"  memory cap: {cap:.1f} GiB ({free / 1024 ** 3:.1f} GiB free, "
              f"{headroom_gib:.1f} GiB left for other processes)", flush=True)
    return cap


def estimate_weight_gib(src: str, dtype_bytes: int = 2) -> float:
    """Approximate weight footprint in GiB.

    Measures the checkpoint on disk when it is a local directory, which is
    exact. Otherwise estimates from the config's parameter shape, which is
    rough but only needs to be good enough to choose a placement.
    """
    shards = glob.glob(os.path.join(src, "*.safetensors")) if os.path.isdir(src) else []
    if shards:
        return sum(os.path.getsize(f) for f in shards) / 1024 ** 3

    cfg_path = os.path.join(src, "config.json") if os.path.isdir(src) else None
    if cfg_path and os.path.exists(cfg_path):
        cfg = json.load(open(cfg_path))
    else:
        try:
            from transformers import AutoConfig
            cfg = AutoConfig.from_pretrained(src, trust_remote_code=True).to_dict()
        except Exception:
            return 99.0  # unknown: assume it needs offload
    h = cfg.get("hidden_size", 0)
    L = cfg.get("num_hidden_layers", 0)
    inter = cfg.get("intermediate_size", 4 * h)
    vocab = cfg.get("vocab_size", 0)
    params = L * (4 * h * h + 3 * h * inter) + 2 * vocab * h
    return params * dtype_bytes / 1024 ** 3


def placement_for(weight_gib: float, headroom_gib: float = HEADROOM_GIB,
                  verbose: bool = True) -> dict | None:
    """None to load wholly on GPU, or a max_memory dict to split with CPU."""
    if not torch.cuda.is_available():
        return None
    free_gib = torch.cuda.mem_get_info()[0] / 1024 ** 3
    if weight_gib + headroom_gib <= free_gib:
        if verbose:
            print(f"  placement: all on GPU ({weight_gib:.1f} GiB weights, "
                  f"{free_gib:.1f} GiB free)")
        return None
    budget = max(0.5, free_gib - headroom_gib)
    if verbose:
        print(f"  placement: split, {budget:.1f} GiB GPU + CPU "
              f"({weight_gib:.1f} GiB weights, {free_gib:.1f} GiB free)")
    return {0: f"{budget:.1f}GiB", "cpu": "11GiB"}


def unwrap_causal_lm(model):
    """Return the text decoder to profile, unwrapping a vision-language wrapper.

    Multimodal checkpoints nest the text model (`LlavaForConditionalGeneration`
    holds a `language_model`, and newer layouts nest it under `model`). The
    profiler reads attention from decoder layers, so it needs the text tower
    rather than the wrapper. Text-only input through the tower is exactly the
    base language model, which is what makes these checkpoints usable as a way
    to reach an architecture that is otherwise unavailable.
    """
    for path in ("language_model", "model.language_model", "text_model"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
        except AttributeError:
            continue
        if any(hasattr(obj, a) for a in ("layers", "model", "transformer")):
            return obj
    return model


def load_for_profiling(src: str, dtype=torch.bfloat16, attempts: int = 4):
    """Load a causal LM with eager attention, retrying with more offload on OOM.

    Eager attention is required: the profiler reads the attention weights the
    module returns, and the fused kernels never materialise them.
    """
    from transformers import AutoModel, AutoModelForCausalLM

    kwargs = dict(torch_dtype=dtype, trust_remote_code=True, attn_implementation="eager")
    placement = placement_for(estimate_weight_gib(src))

    def _from_pretrained(**extra):
        # Multimodal checkpoints are not causal-LM classes, so fall back to
        # AutoModel and let the caller unwrap the text tower.
        try:
            return AutoModelForCausalLM.from_pretrained(src, **kwargs, **extra)
        except (ValueError, KeyError):
            return AutoModel.from_pretrained(src, **kwargs, **extra)

    for attempt in range(attempts):
        try:
            if placement:
                model = _from_pretrained(device_map="auto", max_memory=placement)
            else:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model = _from_pretrained(device_map=device)
            return model.eval()
        except torch.cuda.OutOfMemoryError:
            gc.collect()
            torch.cuda.empty_cache()
            free_gib = torch.cuda.mem_get_info()[0] / 1024 ** 3
            budget = max(0.5, (free_gib - HEADROOM_GIB) / (2 ** attempt))
            placement = {0: f"{budget:.1f}GiB", "cpu": "11GiB"}
            print(f"  OOM on attempt {attempt + 1}, retrying with {budget:.1f} GiB on GPU")
    raise RuntimeError(f"could not fit {src} after {attempts} attempts")
