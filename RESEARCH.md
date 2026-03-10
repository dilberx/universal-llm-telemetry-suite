# Deep-Dive Thermal and Perplexity Analysis

## Abstract
The rapid scaling of Large Language Model (LLM) parameter counts has fundamentally outpaced the memory bandwidth, VRAM capacity, and thermal dissipation envelopes of consumer-grade hardware. Deploying inference systems in edge or cost-constrained environments necessitates aggressive optimization heuristics, primarily low-bit integer quantization. However, a critical systems-level question remains: does the energy efficiency gained via aggressive quantization (such as the $Q4\_K\_M$ paradigm) empirically justify the corresponding degradation in logic retention and generative accuracy within sustained production workloads?

This study presents a multi-dimensional analysis to map this "Efficiency-Logic Frontier." We evaluate a comprehensive matrix of modern architectures (Qwen, Mistral, Llama) and context windows across more than 12 discrete configurations. Utilizing an asynchronous orchestration framework bound to `pynvml`, we capture high-fidelity hardware telemetry—including granular VRAM allocation, power draw, and thermal states—during continuous inference cycles. Concurrent generative accuracy is objectively quantified using automated WikiText-2 Perplexity evaluations.

Furthermore, this research investigates the physical constraints of the NVIDIA Ampere architecture (RTX 3080) under sustained load. By analyzing Dynamic Frequency Scaling (DFS) behavior, we cross-reference GPU temperature against Streaming Multiprocessor (SM) clock velocities to isolate thermal throttling artifacts from fundamental architectural limits during prolonged prefill and decoding phases.

Our empirical data establishes a deterministic correlation between bit-depth, power consumption, and accuracy decay. We identify the Pareto optimal operating envelope for consumer-grade deployment, demonstrating a peak energy efficiency of [INSERT OPTIMAL T/J VALUE] Tokens per Joule (T/J) prior to the onset of catastrophic perplexity degradation.

## Objective
To quantify the impact of aggressive quantization on both hardware efficiency and generative accuracy during Large Language Model (LLM) inference on an NVIDIA RTX 3080.

## Perplexity vs. Execution Latency: The Accuracy Cliff
Aggressive quantization reduces VRAM footprint and memory bandwidth constraints, but introduces an "Accuracy Cliff" where the model's logic retention significantly degrades. 

To map this degradation, we correlate the **WikiText-2 Perplexity** score against Throughput (Tokens per Second). 

1. **Q8_0 and Q5_K_M:** These bit-depths generally exhibit minimal perplexity degradation compared to unquantized baselines. The logic retention remains robust.
2. **Q4_K_M:** As highlighted in our strategic analysis, `Q4_K_M` represents the Pareto Optimal frontier. It offers substantial latency improvements while maintaining acceptable perplexity levels for most general-purpose generative tasks.
3. **Sub-Q4 (The Cliff):** While we did not benchmark sub-Q4 models in this suite, empirical data suggests that dropping to 3-bit or 2-bit quantization leads to exponential increases in perplexity. The model effectively begins outputting noise, rendering the latency gains irrelevant.

## Thermal Throttling: The 3080 Heat Wall
Continuous, compute-heavy inference places immense stress on consumer-grade GPUs. To verify if our benchmarked throughput is sustainable, we logged the RTX 3080's thermal and clock data over extended sessions.

### Prefill vs. Decoding Thermal Profiles
* **Prefill (Prompt Evaluation):** This phase is highly compute-bound. The 8704 CUDA cores are fully saturated computing the attention matrix for the input context. We observe immediate, massive power spikes (approaching the GPU's TDP limit) and rapid temperature escalation.
* **Decoding (Token Generation):** This phase is memory-bandwidth bound. The power draw settles significantly as the ALU saturation drops, waiting for weights to be shuttled from VRAM.

### Throttling Analysis
By continuously polling the GPU Temperature (°C) and SM Clock Speed (MHz), we can identify thermal saturation. If the GPU reaches its thermal limit (typically around 83°C - 87°C for the 3080), it will dynamically down-clock the SMs to shed heat.
* **Impact on TPS:** If thermal throttling occurs during an extended inference session, the Throughput (TPS) will steadily decline as the clock speed drops, invalidating short-burst benchmark metrics. 
* **Conclusion for Production:** For true "Production AI Systems", cooling infrastructure and thermal limits are just as critical as VRAM capacity when deploying to bare-metal edge devices. Sustained heavy-batch inference will inevitably hit the thermal wall on consumer hardware without adequate cooling solutions.