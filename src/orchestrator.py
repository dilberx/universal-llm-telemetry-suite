from __future__ import annotations
"""
orchestrator.py — LLM Inference Telemetry Suite
Asynchronous benchmarking harness supporting NVIDIA RTX, Apple Silicon (M-series),
and CPU-only hosts via the pluggable TelemetryProvider architecture.

Usage:
  python src/orchestrator.py [--path <model-dir-or-file>] [--gpu-index <n>] [--dry-run]
"""

import os
import sys

# Ensure the 'src' directory is in sys.path so local modules are found
src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import glob
import time
import shutil
import threading
import subprocess
import csv
import re
import statistics
import math
import urllib.request
import zipfile
import argparse
import tempfile
import traceback
from typing import Any
from datetime import datetime, timezone

# Telemetry providers (NvidiaProvider, AppleSiliconProvider, NullProvider)
from providers import detect_provider, TelemetryProvider # type: ignore


# ---------------------------------------------------------------------------
# Model family lookup (extensible FAMILY_MAP — no more multi-branch regex)
# ---------------------------------------------------------------------------

FAMILY_MAP: dict[str, str] = {
    "deepseek": "DeepSeek",
    "phi":      "Phi",
    "qwen":     "Qwen",
    "mistral":  "Mistral",
    "llama":    "Llama",
    "gemma":    "Gemma",
    "falcon":   "Falcon",
}


def detect_model_family(model_name: str) -> str:
    """
    Map a GGUF filename to its model family using FAMILY_MAP prefix lookup.

    Lowercases the filename, then checks if any known prefix appears as a
    word-start in the name. Falls back to a best-effort single-token parse.

    Examples:
      DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf → DeepSeek
      Phi-4-mini-instruct-Q8_0.gguf            → Phi
      qwen2.5-3b-instruct-q4_k_m.gguf          → Qwen
      Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf  → Llama
    """
    lower = model_name.lower()
    for prefix, family in FAMILY_MAP.items():
        if re.search(r'(?:^|[-_./])' + re.escape(prefix), lower):
            return family
    # Fallback: capitalise first alphabetic token
    m = re.match(r'^([a-zA-Z]+)', model_name)
    return m.group(1).capitalize() if m else "Generic"


# ---------------------------------------------------------------------------
# WikiText-2 dataset bootstrap
# ---------------------------------------------------------------------------

def ensure_wikitext(dataset_path: str) -> None:
    if not os.path.exists(dataset_path):
        print("Downloading WikiText-2 dataset...")
        os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
        zip_path = dataset_path + ".zip"
        urllib.request.urlretrieve(
            "https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip",
            zip_path,
        )
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(os.path.dirname(dataset_path))
        extracted = os.path.join(
            os.path.dirname(dataset_path), "wikitext-2-raw", "wiki.test.raw"
        )
        os.rename(extracted, dataset_path)
        import shutil as _shutil
        _shutil.rmtree(os.path.join(os.path.dirname(dataset_path), "wikitext-2-raw"))
        os.remove(zip_path)


# ---------------------------------------------------------------------------
# Perplexity evaluation
# ---------------------------------------------------------------------------

def run_perplexity(
    model_path: str,
    llama_perplexity_path: str,
    dataset_path: str,
    context_length: int = 512,
) -> float:
    print(f"  Running Perplexity on {os.path.basename(model_path)} (ctx={context_length})...")
    command = [
        llama_perplexity_path,
        "-m", model_path,
        "-f", dataset_path,
        "-c", str(context_length),
        "--chunks", "4",
    ]
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        output, _ = process.communicate(timeout=300)
        m = re.search(r"Final estimate: PPL = ([\d.]+)", output)
        if m:
            return float(m.group(1))
    except Exception as e:
        print(f"  ⚠️  Perplexity test failed: {e}")
    return 0.0


# ---------------------------------------------------------------------------
# Core benchmark — one inference run
# ---------------------------------------------------------------------------
def run_benchmark(
    model_path: str,
    llama_cli_path: str,
    provider: TelemetryProvider,
    thermal_log_csv: str,
    context_length: int = 2048,
    prompt: str = "Explain transformers in simple terms.",
) -> dict:
    model_name = os.path.basename(model_path)
    provider.start(model_name, thermal_log_csv)
    start_time = time.time()

    backend_flags = provider.get_cli_flags()

    command = [
        llama_cli_path,
        "-m", model_path,
        "-p", prompt,
        "-n", "200",
        "-ngl", "99",
        "-c", str(context_length),
        "--flash-attn", "auto",
    ] + backend_flags

    try:
        # NSUnbufferedIO forces macOS to flush the pipe immediately
        env = os.environ.copy()
        env["LLAMA_LOG_COLORS"] = "0"
        env["NSUnbufferedIO"] = "YES" 

        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env
        )
        # Link the process PID to the telemetry provider for accurate RSS tracking
        provider.set_target_pid(proc.pid)
        
        output, _ = proc.communicate(timeout=180)
        
        # DEBUG: Let's actually see what we captured
        if not output or "Prompt:" not in output:
            print(f"  ⚠️  Warning: Capture length {len(output)}. Check CLI flags.")
            
    except Exception as e:
        output = str(e)
        print(f"  ❌ Runtime Error: {e}")

    end_time = time.time()
    hw: dict[str, Any] = provider.stop()
    latency = end_time - start_time

    # Clean ANSI codes
    ansi_esc = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    clean = ansi_esc.sub("", output)
    
    tokens_per_sec = _parse_tps(clean)
    # Pass the context_length to get an accurate TTFT calculation
    ttft_ms = _parse_ttft_ms(clean, context_length)

    avg_power = hw["avg_power_watts"]
    tokens_per_joule = (tokens_per_sec / avg_power) if avg_power > 0 else 0.0
    family = detect_model_family(model_name)
    
    # Accurate Cold Start: latancy minus the actual inference time
    cold_start_sec = max(0.0, latency - (200 / tokens_per_sec)) if tokens_per_sec > 0 else 0.0

    return {
        "model":           model_name,
        "family":          family,
        "latency_sec":     round(latency, 2),
        "tokens_per_sec":  round(tokens_per_sec, 2),
        "ttft_ms":         round(ttft_ms, 2),
        "max_vram_mb":     hw["max_vram_mb"],
        "avg_power_watts": hw["avg_power_watts"],
        "tokens_per_joule": round(tokens_per_joule, 4),
        "avg_temp_c":      hw["avg_temp_c"],
        "avg_clock_mhz":   hw["avg_clock_mhz"],
        "cold_start_sec":  round(cold_start_sec, 3),
    }

def _parse_tps(clean_output: str) -> float:
    # 1. Look for the formal 'eval time' table (Highest Precision)
    # Example: eval time = 1234.56 ms / 200 runs ( 6.17 ms per token, 161.99 t/s)
    m = re.search(r"eval time\s*=\s*[\d.]+\s*ms\s*/\s*\d+\s*runs\s*.*?\s*([\d.]+)\s*t/s", clean_output)
    if m:
        return float(m.group(1))

    # 2. Fallback to the 'Generation' label if the table is suppressed
    scrubbed = clean_output.replace('\r', '\n')
    matches = re.findall(r"Generation:[^\d]*([\d.]+)", scrubbed, re.IGNORECASE)
    return float(matches[-1]) if matches else 0.0

def _parse_ttft_ms(clean_output: str, ctx_len: int) -> float:
    # 1. Look for 'prompt eval time' in the formal table
    m = re.search(r"prompt eval time\s*=\s*([\d.]+)\s*ms", clean_output, re.IGNORECASE)
    if m:
        return float(m.group(1))

    # 2. Fallback to status bar math
    scrubbed = clean_output.replace('\r', '\n')
    matches = re.findall(r"Prompt:[^\d]*([\d.]+)", scrubbed, re.IGNORECASE)
    if matches:
        tps = float(matches[-1])
        return (ctx_len / tps) * 1000 if tps > 0 else 0.0
        
    return 0.0


def _measure_cold_start(tmp_file, proc: subprocess.Popen, timeout: float = 30.0) -> float:
    """
    Poll the temp file for the first non-empty output line while the process runs.
    Returns seconds from process spawn to first token line appearance.
    Returns 0.0 if not detectable within timeout.
    """
    t0 = time.time()
    deadline = t0 + timeout
    while time.time() < deadline and proc.poll() is None:
        try:
            tmp_file.seek(0)
            content = tmp_file.read()
            if content.strip():
                return float(f"{(time.time() - t0):.3f}")
        except Exception:
            pass
        time.sleep(0.05)
    return 0.0


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def _compute_ci(values: list[float]) -> tuple[float, float]:
    """Return (mean, 95% CI half-width)."""
    if not values:
        return 0.0, 0.0
    avg = statistics.mean(values)
    if len(values) < 2:
        return avg, 0.0
    std = statistics.stdev(values)
    ci = 1.96 * std / math.sqrt(len(values))
    return avg, ci


def _log_error(
    log_path: str,
    model: str,
    ctx_len: int,
    label: str,
    exc: Exception,
) -> None:
    """Append a timestamped error entry to errors.log."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(log_path, "a") as f:
        f.write(f"[{ts}] {label} | model={model} ctx={ctx_len}\n")
        f.write(f"  {type(exc).__name__}: {exc}\n")
        f.write(traceback.format_exc())
        f.write("\n")


# ---------------------------------------------------------------------------
# Binary resolution
# ---------------------------------------------------------------------------

_LLAMA_BUILD_HINT = (
    "\n  Build instructions:\n"
    "    git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp\n"
    "    cmake -B build && cmake --build build --config Release -j$(nproc)\n"
    "  Or add the directory containing llama-cli to your PATH."
)


def _resolve_binary(name: str, fallback_relative: str) -> str:
    """Resolve a binary: PATH first, then relative fallback."""
    on_path = shutil.which(name)
    if on_path:
        return on_path
    if os.path.isfile(fallback_relative):
        return fallback_relative
    return ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM Inference Telemetry Suite — cross-platform benchmarking harness"
    )
    parser.add_argument(
        "--path",
        type=str,
        default=os.path.expanduser("~/dev/llm_models"),
        help="Path to a .gguf file or directory of models.",
    )
    parser.add_argument(
        "--gpu-index",
        type=int,
        default=0,
        help="GPU device index for NVIDIA multi-GPU hosts (default: 0).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Detect hardware, validate paths, then exit without running inference.",
    )
    args = parser.parse_args()

    # --- Provider detection -------------------------------------------
    provider = detect_provider(gpu_index=args.gpu_index)
    hw_info: dict[str, Any] = provider.get_hardware_info()

    print(f"Hardware Detected : {hw_info['gpu_name']}")
    print(f"Memory            : {hw_info['total_vram_gb']} GB ({hw_info['memory_type']})")
    print(f"Provider          : {type(provider).__name__}")
    print(f"Isolation Level   : {hw_info['isolation_level']}")
    if hw_info["base_clock_mhz"] > 0:
        print(f"Base Clock        : {hw_info['base_clock_mhz']:.0f} MHz")

    if args.dry_run:
        print("\n[--dry-run] Skipping inference. Exiting cleanly.")
        sys.exit(0)

    # --- Binary resolution -------------------------------------------
    base_dir = os.path.abspath(os.path.dirname(__file__))
    llama_cli = _resolve_binary(
        "llama-cli",
        os.path.normpath(os.path.join(base_dir, "../../llama.cpp/build/bin/llama-cli")),
    )
    llama_perplexity = _resolve_binary(
        "llama-perplexity",
        os.path.normpath(os.path.join(base_dir, "../../llama.cpp/build/bin/llama-perplexity")),
    )

    if not llama_cli:
        print(f"\n❌ llama-cli binary not found.{_LLAMA_BUILD_HINT}")
        sys.exit(1)

    # --- Results directory (GPU-slug bucketed) ------------------------
    gpu_slug = (
        hw_info["gpu_name"]
        .lower()
        .replace(" ", "_")
        .replace("nvidia_", "")
        .replace("geforce_", "")
        .replace("apple_", "")
    )
    results_root = os.path.join(base_dir, "../results")
    results_dir  = os.path.join(results_root, gpu_slug)
    os.makedirs(results_dir, exist_ok=True)

    output_csv      = os.path.join(results_dir, "production_benchmarks.csv")
    thermal_log_csv = os.path.join(results_dir, "thermal_log.csv")
    dataset_path    = os.path.join(base_dir, "wikitext-2.txt")

    # Clear thermal log for this run
    if os.path.exists(thermal_log_csv):
        os.remove(thermal_log_csv)

    ensure_wikitext(dataset_path)

    # --- Model discovery ---------------------------------------------
    gguf_files: list[str] = []
    if os.path.isfile(args.path) and args.path.lower().endswith(".gguf"):
        gguf_files = [str(args.path)]
    elif os.path.isdir(args.path):
        pattern = os.path.join(args.path, "**", "*.gguf")
        found = glob.glob(pattern, recursive=True)
        gguf_files = [str(f) for f in sorted(found)]

    if not gguf_files:
        print(f"No GGUF models found in: {args.path}")
        sys.exit(0)

    print(f"\nFound {len(gguf_files)} GGUF model(s). Starting Scientific Rigor Benchmarks...\n")

    # --- Error log -------------------------------------------------------
    errors_log = os.path.join(results_dir, "errors.log")
    run_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(errors_log, "a") as ef:
        ef.write(f"\n{'='*60}\nRun started: {run_ts}\nHardware: {hw_info['gpu_name']}\n{'='*60}\n")

    # --- Benchmark loop --------------------------------------------------
    context_lengths = [512, 2048, 8192]
    num_runs        = 10
    all_results: list[dict] = []
    skipped: list[str] = []

    for model_path in gguf_files:
        model_name = os.path.basename(model_path)

        for ctx_len in context_lengths:
            print(f"Benchmarking {model_name} | Context: {ctx_len}")

            try:
                ppl = run_perplexity(model_path, llama_perplexity, dataset_path, ctx_len) \
                      if llama_perplexity else 0.0
                print(f"  --> Perplexity (WikiText-2): {ppl:.4f}")
            except Exception as e:
                _log_error(errors_log, model_name, ctx_len, "perplexity", e)
                ppl = 0.0

            run_metrics: list[dict] = []
            config_skipped = False
            for run_num in range(1, num_runs + 1):
                if config_skipped:
                    break
                print(f"  Run {run_num}/{num_runs}...", end=" ", flush=True)
                try:
                    m = run_benchmark(
                        model_path, llama_cli, provider, thermal_log_csv,
                        context_length=ctx_len,
                    )
                except MemoryError as e:
                    print("[OOM — skipping config]")
                    _log_error(errors_log, model_name, ctx_len, f"OOM run {run_num}", e)
                    skipped.append(f"{model_name} ctx={ctx_len}")
                    config_skipped = True
                    continue
                except subprocess.TimeoutExpired as e:
                    print("[TIMEOUT — skipping run]")
                    _log_error(errors_log, model_name, ctx_len, f"Timeout run {run_num}", e)
                    continue
                except Exception as e:
                    err_str = str(e).lower()
                    is_oom = any(kw in err_str for kw in [
                        "out of memory", "oom", "alloc failed", "cannot allocate",
                        "mlock failed", "vram", "metal",
                    ])
                    if is_oom:
                        print("[OOM — skipping config]")
                        _log_error(errors_log, model_name, ctx_len, f"OOM run {run_num}", e)
                        skipped.append(f"{model_name} ctx={ctx_len}")
                        config_skipped = True
                    else:
                        print("[ERROR — skipping run]")
                        _log_error(errors_log, model_name, ctx_len, f"Error run {run_num}", e)
                    continue

                m.update({
                    "gpu_name":        hw_info["gpu_name"],
                    "total_vram_gb":   hw_info["total_vram_gb"],
                    "base_clock_mhz":  hw_info["base_clock_mhz"],
                    "memory_type":     hw_info["memory_type"],
                    "isolation_level": hw_info["isolation_level"],
                    "run_number":      run_num,
                    "context_length":  ctx_len,
                    "perplexity":      ppl,
                })
                all_results.append(m)
                run_metrics.append(m)
                print(f"{m['tokens_per_sec']:.1f} t/s | "
                      f"TTFT {m.get('ttft_ms', 0):.0f} ms | "
                      f"{m['tokens_per_joule']:.4f} T/J")

            if run_metrics:
                tps_avg, tps_ci   = _compute_ci([r["tokens_per_sec"]     for r in run_metrics])
                vram_avg, vram_ci = _compute_ci([r["max_vram_mb"]        for r in run_metrics])
                cs_avg, cs_ci     = _compute_ci([r["cold_start_sec"]     for r in run_metrics])
                ttft_avg, ttft_ci = _compute_ci([r.get("ttft_ms", 0.0)   for r in run_metrics])
                temp_avg, _       = _compute_ci([r["avg_temp_c"]         for r in run_metrics])
                clock_avg, _      = _compute_ci([r["avg_clock_mhz"]      for r in run_metrics])

                print(f"  --> Avg TPS     : {tps_avg:.2f} ± {tps_ci:.2f} t/s (95% CI)")
                print(f"  --> Avg TTFT    : {ttft_avg:.1f} ± {ttft_ci:.1f} ms (95% CI)")
                print(f"  --> Avg Memory  : {vram_avg:.2f} ± {vram_ci:.2f} MB (95% CI)")
                print(f"  --> Cold Start  : {cs_avg:.3f} ± {cs_ci:.3f} s (95% CI)")
                if clock_avg > 0:
                    print(f"  --> SM Clock    : {clock_avg:.2f} MHz | Temp: {temp_avg:.1f} °C")
                print()

    # --- Write results CSV -------------------------------------------
    fieldnames = [
        "gpu_name", "total_vram_gb", "base_clock_mhz", "memory_type", "isolation_level",
        "model", "family", "context_length", "run_number",
        "latency_sec", "tokens_per_sec", "ttft_ms", "max_vram_mb",
        "avg_power_watts", "tokens_per_joule",
        "avg_temp_c", "avg_clock_mhz",
        "cold_start_sec", "perplexity",
    ]

    print(f"Saving results → {output_csv}")

    # --- Append-safe write: merge with existing data, deduplicate --------
    existing_rows: list[dict] = []
    if os.path.isfile(output_csv):
        with open(output_csv, mode="r", newline="") as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)
        print(f"  ℹ️  Found {len(existing_rows)} existing row(s) — merging without duplicates.")

    # Dedup key: (model, context_length, run_number)
    seen: set[tuple] = set()
    for row in existing_rows:
        key = (row.get("model", ""), row.get("context_length", ""), row.get("run_number", ""))
        seen.add(key)

    new_count = 0
    for row in all_results:
        key = (row.get("model", ""), str(row.get("context_length", "")), str(row.get("run_number", "")))
        if key not in seen:
            existing_rows.append({k: row.get(k, "") for k in fieldnames})
            seen.add(key)
            new_count += 1

    with open(output_csv, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in existing_rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    print(f"  ✅ Wrote {len(existing_rows)} total rows ({new_count} new, {len(existing_rows) - new_count} existing).")

    if skipped:
        print(f"\n⚠️  Skipped {len(skipped)} config(s) due to OOM/errors:")
        for s in skipped:
            print(f"   • {s}")
        print(f"   Full details → {errors_log}")

    print(f"Thermal log    → {thermal_log_csv}")
    print("Benchmarking complete! ✅")


if __name__ == "__main__":
    main()
