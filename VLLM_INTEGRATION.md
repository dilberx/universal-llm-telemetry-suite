# Integration Blueprint: vLLM & TensorRT-LLM

This document provides a technical integration blueprint showing how to port **Speculative Early-Exit Routing (SEER)** and **Step-by-Step Entropy-guided KV Pruning (SDE-KV)** into high-performance serving frameworks like **vLLM** and **TensorRT-LLM**.

---

## 1. Integrating SEER in vLLM

vLLM processes requests using a central `LLMEngine` that coordinates the `Scheduler` and the execution workers. To implement SEER cascades, the engine must dynamically inspect token generation confidence and trigger a schedule transition.

```
                  vLLM Engine Pipeline with SEER
                  ──────────────────────────────

 [Client Request] ──> [LLMEngine] ──> [Worker: Run Small Model]
                           │
                           ├──> [Evaluate Step K Confidence]
                           │          │
                           │          ├──> Stable? ──> [Generate locally]
                           │          │
                           │          └──> Unstable? ──> [Abort & Route]
                           │                                  │
                           └──────────────────────────────────v
                                                   [Worker: Run Large Model]
```

### Steps for Implementation

#### 1. Configure the Multi-Model Executor
Configure vLLM to run two model instances (e.g., Qwen2.5-0.5B and Phi-2 2.7B) in parallel workers or sequentially on the same GPU.

#### 2. Implement the Trajectory Monitor in `LLMEngine`
In `vllm/engine/llm_engine.py`, modify the step loop to intercept sampler outputs for early exit evaluation:

```python
# vllm/engine/llm_engine.py

class LLMEngine:
    def _process_model_outputs(self, model_outputs, scheduler_outputs):
        for seq_group in scheduler_outputs.active_seq_groups:
            # We target the speculative early exit check window (K=3 tokens)
            if seq_group.is_running_on_small_model and seq_group.num_generated_tokens == 3:
                # Retrieve logits / probabilities from the sampler output
                probs = seq_group.get_recent_token_probabilities(limit=3)
                entropies = seq_group.get_recent_token_entropies(limit=3)
                
                avg_prob = sum(probs) / len(probs)
                avg_entropy = sum(entropies) / len(entropies)
                
                # SEER transition threshold
                if avg_prob < 0.85 or avg_entropy > 2.0:
                    # Trigger early exit abort
                    self._abort_small_model_request(seq_group)
                    self._route_to_large_model_scheduler(seq_group)
```

#### 3. Update the Scheduler to Transition State
In `vllm/core/scheduler.py`, handle the transition:
- Release the physical blocks allocated for the small model's KV cache.
- Re-queue the prompt into the large model's prefill batch queue.

---

## 2. Integrating SDE-KV in vLLM PagedAttention

vLLM manages KV caches using physical pages via the `BlockSpaceManager`. SDE-KV can be integrated by dynamically freeing physical blocks when the step-wise attention entropy indicates a low-complexity generation phase.

### Steps for Implementation

#### 1. Implement Entropy-guided Page Pruning in the Sampler
At each step of generation, capture the attention scores returned by the model executor. Compute the Shannon entropy of the attention distribution across the layers:

```python
# vllm/model_executor/layers/attention/ops.py

def compute_step_entropy(attention_scores):
    # Shape: [num_seqs, num_heads, 1, seq_len]
    probs = torch.softmax(attention_scores, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
    return entropy.mean().item()
```

#### 2. Re-allocate Cache Blocks in the Block Manager
When entropy is low, communicate the target page budget to the `BlockSpaceManager`:

```python
# vllm/core/block_manager.py

class BlockSpaceManager:
    def adjust_kv_budget(self, seq_id, entropy_ratio):
        # Scale block count based on the current step's entropy ratio
        base_budget = self.get_allocated_blocks(seq_id)
        target_budget = max(4, int(base_budget * (0.25 + 0.75 * entropy_ratio)))
        
        # Free the least-important historical blocks if target budget is smaller
        if target_budget < base_budget:
            blocks_to_free = base_budget - target_budget
            self.free_historical_blocks(seq_id, count=blocks_to_free)
```

---

## 3. Integrating SDE-KV in TensorRT-LLM

TensorRT-LLM manages memory via the **KVCacheManager** in its C++ runtime. SDE-KV can be integrated directly into the custom plugins for FlashAttention or PagedAttention.

### Steps for Implementation

#### 1. Modify the Attention Kernel
In the C++ CUDA kernel (`tensorrt_llm/kernels/decoderMaskedMultiheadAttention.cu`), intercept the attention weights before computing the softmax output.

#### 2. Update the Cache Manager
Implement block-wise deallocation in `tensorrt_llm/runtime/kvCacheManager.cpp`:
- Set a step-wise budget based on the calculated entropy float passed from the CUDA kernel to the host runner.
- Modify the page allocation table to unmap historical block pointers when execution complexity drops.
