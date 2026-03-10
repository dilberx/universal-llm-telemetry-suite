import os
import sys
import glob
import time
import threading
import subprocess
import csv
import re
import statistics
import pynvml

def monitor_vram(stop_event, vram_usage_list):
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        if device_count == 0:
            return
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        
        while not stop_event.is_set():
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            vram_usage_list.append(info.used / (1024 ** 2)) # in MB
            time.sleep(0.1)
            
        pynvml.nvmlShutdown()
    except pynvml.NVMLError:
        pass

def run_benchmark(model_path, llama_cli_path, context_length=2048, prompt="Explain transformers in simple terms."):
    stop_event = threading.Event()
    vram_usage = []
    
    # Start VRAM monitor in background
    monitor_thread = threading.Thread(target=monitor_vram, args=(stop_event, vram_usage))
    monitor_thread.start()
    
    start_time = time.time()
    
    command = [
        llama_cli_path,
        "-m", model_path,
        "-p", prompt,
        "-n", "200",          # Limit generation
        "-ngl", "99",         # GPU Offload
        "-c", str(context_length), # Dynamic Context Length
        "--flash-attn", "auto",
        "--no-conversation",  # Disable chat mode
        "--single-turn"       # The 2026 "Secret Weapon" flag to force exit
    ]
    
    # Use 'script' to trick llama-cli into thinking it's in a TTY
    # Syntax: script -e -q -c "command" /dev/null
    cmd_str = " ".join([f'"{arg}"' if " " in arg else arg for arg in command])
    tty_command = ["script", "-e", "-q", "-c", cmd_str, "/dev/null"]
    
    import tempfile
    
    with tempfile.NamedTemporaryFile(mode='w+', delete=True) as tmp:
        try:
            process = subprocess.Popen(
                tty_command,
                stdin=subprocess.DEVNULL,
                stdout=tmp,
                stderr=subprocess.STDOUT,
                text=True
            )
            process.wait(timeout=120)
            tmp.seek(0)
            output = tmp.read()
        except subprocess.TimeoutExpired:
            process.kill()
            tmp.seek(0)
            output = tmp.read()
            print(f"⚠️ Benchmark for {model_path} timed out and was killed.")
    
    end_time = time.time()
    
    # Stop VRAM monitoring
    stop_event.set()
    monitor_thread.join()
    
    max_vram = max(vram_usage) if vram_usage else 0.0
    latency = end_time - start_time
    
    # Remove ANSI escape codes for cleaner regex matching
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    clean_output = ansi_escape.sub('', output)
    
    # Priority Match: Newest format [ Prompt: XX.X t/s | Generation: XX.X t/s ]
    tokens_per_sec = 0.0
    # Use a more flexible regex that allows for brackets and pipe, with MULTILINE
    generation_match = re.search(r"Generation:\s*([\d.]+)\s*t/s", clean_output, re.MULTILINE)
    
    if generation_match:
        tokens_per_sec = float(generation_match.group(1))
    else:
        # Fallback 1: "eval time = ... ( ... tokens per second)"
        match_eval = re.search(r"eval time\s*=\s*.*?\(\s*[\d.]+\s*ms per token,\s*([\d.]+)\s*tokens per second\)", clean_output, re.MULTILINE)
        if match_eval:
            tokens_per_sec = float(match_eval.group(1))
        else:
            # Fallback 2: Generic "XX.XX tokens per second"
            match_generic = re.search(r"([\d.]+)\s*tokens per second", clean_output, re.MULTILINE)
            if match_generic:
                tokens_per_sec = float(match_generic.group(1))
                
    model_name = os.path.basename(model_path)
    family = "Qwen" if "qwen" in model_name.lower() else "Mistral" if "mistral" in model_name.lower() else "Unknown"
        
    return {
        "model": model_name,
        "family": family,
        "latency_sec": round(latency, 2),
        "tokens_per_sec": round(tokens_per_sec, 2),
        "max_vram_mb": round(max_vram, 2)
    }


def main():
    dirs_to_scan = [
        os.path.expanduser("~/dev/llm_models/qwen3b_gguf"),
        os.path.expanduser("~/dev/llm_models/mistral7b")
    ]
    
    # Base directory is now /src
    base_dir = os.path.abspath(os.path.dirname(__file__))
    llama_cli = os.path.normpath(os.path.join(base_dir, "../../llama.cpp/build/bin/llama-cli"))
    results_dir = os.path.join(base_dir, "../results")
    output_csv = os.path.join(results_dir, "production_benchmarks.csv")
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    if not os.path.exists(llama_cli):
        print(f"Error: llama-cli not found at {llama_cli}")
        sys.exit(1)
        
    gguf_files = []
    for d in dirs_to_scan:
        if os.path.exists(d):
            gguf_files.extend(glob.glob(os.path.join(d, "*.gguf")))
        else:
            print(f"Warning: Models directory not found at {d}")
            
    gguf_files = sorted(gguf_files)
    
    if not gguf_files:
        print("No GGUF models found in any directory.")
        sys.exit(0)
        
    print(f"Found {len(gguf_files)} GGUF models:")
    for f in gguf_files:
        print(f"  - {os.path.basename(f)}")
    print("\nStarting Expansion Phase Benchmarks...\n")
    
    results = []
    for model_path in gguf_files:
        model_name = os.path.basename(model_path)
        
        # Context Scaling Test: Varying Context Windows for Q5 Qwen
        context_lengths = [2048]
        if "q5_k_m" in model_name.lower() and "qwen" in model_name.lower():
            context_lengths = [512, 2048, 8192]
            
        for ctx_len in context_lengths:
            print(f"Benchmarking {model_name} (Context: {ctx_len})...")
            
            run_metrics = []
            # Multi-Run Averaging: 3 runs per model/context setting
            for run_num in range(1, 4):
                print(f"  Run {run_num}/3...")
                metrics = run_benchmark(model_path, llama_cli, context_length=ctx_len)
                metrics['run_number'] = run_num
                metrics['context_length'] = ctx_len
                results.append(metrics)
                run_metrics.append(metrics)
            
            # Calculate metrics average and std deviation
            tps_list = [m['tokens_per_sec'] for m in run_metrics]
            vram_list = [m['max_vram_mb'] for m in run_metrics]
            
            avg_tps = statistics.mean(tps_list)
            std_tps = statistics.stdev(tps_list) if len(tps_list) > 1 else 0.0
            avg_vram = statistics.mean(vram_list)
            std_vram = statistics.stdev(vram_list) if len(vram_list) > 1 else 0.0
            
            print(f"  --> Avg Throughput: {avg_tps:.2f} ± {std_tps:.2f} tokens/s")
            print(f"  --> Avg Max VRAM: {avg_vram:.2f} ± {std_vram:.2f} MB\n")
        
    print(f"Saving results to {output_csv}")
    with open(output_csv, mode="w", newline="") as f:
        # Added new columns for dataset expansion
        writer = csv.DictWriter(f, fieldnames=["model", "family", "context_length", "run_number", "latency_sec", "tokens_per_sec", "max_vram_mb"])
        writer.writeheader()
        for r in results:
            writer.writerow(r)
            
    print("Benchmarking complete!")

if __name__ == "__main__":
    main()