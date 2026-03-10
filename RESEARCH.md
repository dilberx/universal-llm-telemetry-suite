# Deep-Dive Thermal and Perplexity Analysis

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