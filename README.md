# LLM-Inference-Benchmarker-3080: Evaluating Local GPU LLM Performance

## 1. Abstract & Key Insights

This engineering whitepaper and accompanying repository present a systematic evaluation of Large Language Model (LLM) inference performance on consumer-grade hardware, specifically targeting the NVIDIA RTX 3080 (10GB VRAM) running within a WSL2 environment. The objective is to quantify the performance trade-offs inherent in different model architectures (Qwen 3B vs. Mistral 7B) and quantization paradigms (Q4, Q5, Q8) using the GGUF format and `llama.cpp`.

By analyzing throughput (tokens/second) and VRAM efficiency (TPS/GB), this study provides empirical data on the boundaries of local inference. Key insights reveal a non-linear scaling of throughput relative to bit-depth reduction, highlighting a definitive transition from compute-bound operations at lower parameter counts (3B) to memory-bandwidth-bound constraints at higher parameter counts (7B).

## 2. System Architecture: The Telemetry-Aware Orchestrator

The core of this benchmarking suite is a bespoke, telemetry-aware orchestrator written in Python. It is designed to run asynchronously, capturing real-time hardware metrics without interfering with the inference process.

*   **Asynchronous VRAM Polling:** Utilizing `pynvml` (NVIDIA Management Library), a dedicated daemon thread polls VRAM allocation at 100ms intervals during inference. This ensures high-fidelity peak memory capture without introducing synchronization overhead to the main execution thread.
*   **Deterministic Metric Extraction:** The orchestrator interfaces directly with the `llama-cli` binary via subprocesses. To accurately capture performance metrics from the standard output stream, it employs a robust regular expression engine specifically tuned for the modern `llama.cpp` telemetry format (`[ Prompt: XX.X t/s | Generation: XX.X t/s ]`). This ensures precise throughput calculations regardless of terminal ANSI escapes or varied logging verbosity.

## 3. Experimental Design

The benchmark matrix was carefully constructed to isolate the effects of model scale and quantization depth.

*   **The Benchmark Matrix:**
    *   **Qwen2.5 3B:** Evaluated at Q4_K_M, Q5_K_M, and Q8_0 quantization levels to observe intra-architecture scaling.
    *   **Mistral 7B Instruct v0.3:** Evaluated at Q5_K_M to establish a comparative baseline for larger, memory-intensive architectures.
*   **Variables:** The primary independent variables are parameter count and quantization bit-depth. The dependent variables are generation throughput (tokens/sec) and peak VRAM allocation (MB). This setup explicitly tests the boundaries of compute-bound workloads (where ALU saturation is the limit) versus memory-bound workloads (where memory bandwidth dictates performance).

## 4. Hardware Bottleneck Analysis: The RTX 3080 Envelope

The RTX 3080, equipped with 10GB of GDDR6X memory and 8704 CUDA cores, presents a fascinating constraint profile for LLM inference.

Empirical data from this benchmark demonstrates a distinct "throughput ceiling." For the 3B parameter models, throughput scales inversely with bit-depth, as expected. However, the performance delta between Q4 and Q5 is less pronounced than the delta between Q5 and Q8, suggesting that at lower bit-depths, the compute units (ALUs) are fully saturated, and further memory bandwidth savings yield diminishing returns in generation speed.

Conversely, the Mistral 7B (Q5) model exhibits a classic memory-bandwidth bottleneck. The sheer volume of weights that must be transferred from VRAM to the streaming multiprocessors (SMs) for each token generated significantly limits the maximum tokens per second, regardless of available compute power. Furthermore, aggressive quantization (Q4/Q5) is absolutely critical for the 7B class on a 10GB GPU; without it, the KV cache footprint required for meaningful context windows would immediately trigger Out-Of-Memory (OOM) faults.

## 5. Sustainability & Cost-Efficiency Analysis

As language models scale, operational expenditure (OpEx) driven by power consumption becomes a critical constraint. This benchmark suite integrates real-time energy telemetry to evaluate the sustainability of local and cloud-based deployments.

*   **The Energy Metric:** We introduce **Tokens per Joule (T/J)** as the primary metric for measuring inference sustainability. This metric is calculated by dividing the generation throughput (tokens/second) by the average power draw (Watts) during the inference cycle.
*   **The Efficiency Gap:** Our latest empirical data highlights a significant efficiency gap between model architectures. The Qwen 2.5 3B (Q4) model provides the highest energy efficiency, generating nearly 1.0 tokens per joule. In contrast, the heavier Mistral 7B yields roughly 0.5 tokens per joule. This demonstrates that highly optimized, smaller models can offer double the energy efficiency of larger counterparts.
*   **The '3080' Thermal Profile:** The telemetry module accurately captures the power footprint across the inference lifecycle. We observe distinct thermal profiles on the RTX 3080: power consumption spikes dramatically during the compute-intensive 'Prefill' (Prompt Evaluation) phase as the ALUs are saturated, before settling into a lower, steady-state power draw during the memory-bound 'Decoding' (Token Generation) phase.
*   **Strategic Value:** For businesses like Nodal AI deploying AI infrastructure at scale, this framework provides immediate strategic value. By quantifying Tokens per Joule alongside throughput, organizations can make data-driven decisions to choose model architectures and quantization levels that minimize cloud infrastructure and electricity costs without sacrificing critical throughput.

## 6. DevOps & Systems Engineering Challenges

Executing these benchmarks reliably required mitigating several complex systems engineering challenges, particularly within the WSL2 environment.

*   **Mitigating the WSL2 'Interactive Hang':** Standard subprocess execution of `llama-cli` often resulted in interactive hangs, as the binary expects a TTY interface for standard input/output. This was resolved by implementing a POSIX `script` wrapper (`script -e -q -c <cmd> /dev/null`) to simulate a pseudo-terminal (pty), ensuring non-blocking execution and consistent metric flushing to standard output. Additionally, the `--single-turn` flag was utilized to force termination after initial generation.
*   **Optimized Model Acquisition:** To handle multi-gigabyte GGUF artifacts efficiently, the environment leverages `hf_transfer` for accelerated downloads from the Hugging Face Hub, maximizing network throughput and reducing CI/CD pipeline latency.
*   **Environment-Driven Automation:** The orchestration suite is fully parameterized via environment variables, allowing for headless, automated execution across different hardware profiles without code modifications.

## 7. Reproducibility Guide: Local Inference Lab Setup

To independently verify these findings or extend the benchmark matrix, follow this setup procedure:

1.  **Environment Initialization:**
    Ensure an NVIDIA driver and CUDA toolkit compatible with your hardware are installed. Within a WSL2 or native Linux environment, instantiate a Python virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install pynvml pandas seaborn matplotlib
    ```

2.  **Binary Compilation:**
    Clone the `llama.cpp` repository and compile it with CUDA support:
    ```bash
    git clone https://github.com/ggerganov/llama.cpp
    cd llama.cpp
    make GGML_CUDA=1
    ```

3.  **Model Acquisition:**
    Download the target GGUF models into respective directories (e.g., `~/dev/llm_models/qwen3b_gguf/`, `~/dev/llm_models/mistral7b/`).

4.  **Execute Benchmarks:**
    Run the orchestrator to generate the telemetry data:
    ```bash
    python src/orchestrator.py
    ```

5.  **Visualize Telemetry:**
    Generate the performance dashboards using Seaborn:
    ```bash
    python src/visualizer.py
    ```