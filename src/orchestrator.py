import os
import sys
import glob
import time
import threading
import subprocess
import csv
import re
import statistics
import math
import pynvml
import urllib.request
import zipfile

def ensure_wikitext(dataset_path):
    if not os.path.exists(dataset_path):
        print("Downloading WikiText-2 dataset...")
        os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
        zip_path = dataset_path + ".zip"
        urllib.request.urlretrieve("https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip", zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(os.path.dirname(dataset_path))
        extracted_file = os.path.join(os.path.dirname(dataset_path), "wikitext-2-raw", "wiki.test.raw")
        os.rename(extracted_file, dataset_path)
        import shutil
        shutil.rmtree(os.path.join(os.path.dirname(dataset_path), "wikitext-2-raw"))
        os.remove(zip_path)

def monitor_hardware(stop_event, vram_usage_list, power_usage_list, temp_list, clock_list, thermal_log_csv, model_name):
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        if device_count == 0:
            return
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        
        with open(thermal_log_csv, mode='a', newline='') as f:
            writer = csv.writer(f)
            if os.path.getsize(thermal_log_csv) == 0:
                writer.writerow(["timestamp", "model", "vram_mb", "power_w", "temp_c", "clock_mhz"])
                
            while not stop_event.is_set():
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                vram_mb = info.used / (1024 ** 2)
                vram_usage_list.append(vram_mb)
                
                power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                power_w = power_mw / 1000.0
                power_usage_list.append(power_w)
                
                temp_c = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                temp_list.append(temp_c)
                
                clock_mhz = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
                clock_list.append(clock_mhz)
                
                writer.writerow([time.time(), model_name, vram_mb, power_w, temp_c, clock_mhz])
                f.flush()
                
                time.sleep(0.5) 
                
        pynvml.nvmlShutdown()
    except pynvml.NVMLError:
        pass

def run_perplexity(model_path, llama_perplexity_path, dataset_path, context_length=512):
    print(f"  Running Perplexity test on {os.path.basename(model_path)} (Context: {context_length})...")
    command = [
        llama_perplexity_path,
        "-m", model_path,
        "-f", dataset_path,
        "-c", str(context_length),
        "--chunks", "4" 
    ]
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        output, _ = process.communicate(timeout=300)
        match = re.search(r"Final estimate: PPL = ([\d.]+)", output)
        if match:
            return float(match.group(1))
    except Exception as e:
        print(f"  ⚠️ Perplexity test failed: {e}")
    return 0.0

def run_benchmark(model_path, llama_cli_path, thermal_log_csv, context_length=2048, prompt="Explain transformers in simple terms."):
    stop_event = threading.Event()
    vram_usage = []
    power_usage = []
    temp_usage = []
    clock_usage = []
    
    model_name = os.path.basename(model_path)
    
    monitor_thread = threading.Thread(target=monitor_hardware, args=(stop_event, vram_usage, power_usage, temp_usage, clock_usage, thermal_log_csv, model_name))
    monitor_thread.start()
    
    start_time = time.time()
    
    command = [
        llama_cli_path,
        "-m", model_path,
        "-p", prompt,
        "-n", "200",          
        "-ngl", "99",         
        "-c", str(context_length), 
        "--flash-attn", "auto",
        "--no-conversation",  
        "--single-turn"       
    ]
    
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
    
    stop_event.set()
    monitor_thread.join()
    
    max_vram = max(vram_usage) if vram_usage else 0.0
    avg_power = statistics.mean(power_usage) if power_usage else 0.0
    avg_temp = statistics.mean(temp_usage) if temp_usage else 0.0
    avg_clock = statistics.mean(clock_usage) if clock_usage else 0.0
    latency = end_time - start_time
    
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    clean_output = ansi_escape.sub('', output)
    
    tokens_per_sec = 0.0
    generation_match = re.search(r"Generation:\s*([\d.]+)\s*t/s", clean_output, re.MULTILINE)
    
    if generation_match:
        tokens_per_sec = float(generation_match.group(1))
    else:
        match_eval = re.search(r"eval time\s*=\s*.*?\(\s*[\d.]+\s*ms per token,\s*([\d.]+)\s*tokens per second\)", clean_output, re.MULTILINE)
        if match_eval:
            tokens_per_sec = float(match_eval.group(1))
        else:
            match_generic = re.search(r"([\d.]+)\s*tokens per second", clean_output, re.MULTILINE)
            if match_generic:
                tokens_per_sec = float(match_generic.group(1))
                
    family = "Qwen" if "qwen" in model_name.lower() else "Mistral" if "mistral" in model_name.lower() else "Unknown"
    tokens_per_joule = (tokens_per_sec / avg_power) if avg_power > 0 else 0.0
        
    return {
        "model": model_name,
        "family": family,
        "latency_sec": round(latency, 2),
        "tokens_per_sec": round(tokens_per_sec, 2),
        "max_vram_mb": round(max_vram, 2),
        "avg_power_watts": round(avg_power, 2),
        "tokens_per_joule": round(tokens_per_joule, 4),
        "avg_temp_c": round(avg_temp, 2),
        "avg_clock_mhz": round(avg_clock, 2)
    }

def main():
    dirs_to_scan = [
        os.path.expanduser("~/dev/llm_models/qwen3b_gguf"),
        os.path.expanduser("~/dev/llm_models/mistral7b")
    ]
    
    base_dir = os.path.abspath(os.path.dirname(__file__))
    llama_cli = os.path.normpath(os.path.join(base_dir, "../../llama.cpp/build/bin/llama-cli"))
    llama_perplexity = os.path.normpath(os.path.join(base_dir, "../../llama.cpp/build/bin/llama-perplexity"))
    
    results_dir = os.path.join(base_dir, "../results")
    output_csv = os.path.join(results_dir, "production_benchmarks.csv")
    thermal_log_csv = os.path.join(results_dir, "thermal_log.csv")
    dataset_path = os.path.join(base_dir, "wikitext-2.txt")
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    if not os.path.exists(llama_cli):
        print(f"Error: llama-cli not found at {llama_cli}")
        sys.exit(1)
        
    ensure_wikitext(dataset_path)
        
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
        
    print(f"Found {len(gguf_files)} GGUF models.")
    print("\nStarting Scientific Rigor Benchmarks...\n")
    
    if os.path.exists(thermal_log_csv):
        os.remove(thermal_log_csv)
    
    results = []
    for model_path in gguf_files:
        model_name = os.path.basename(model_path)
        
        context_lengths = [2048]
        if "q5_k_m" in model_name.lower() and "qwen" in model_name.lower():
            context_lengths = [512, 2048, 8192]
            
        for ctx_len in context_lengths:
            print(f"Benchmarking {model_name} (Context: {ctx_len})...")
            
            ppl = run_perplexity(model_path, llama_perplexity, dataset_path, ctx_len)
            print(f"  --> Perplexity (WikiText-2): {ppl:.4f}")
            
            run_metrics = []
            num_runs = 10
            for run_num in range(1, num_runs + 1):
                print(f"  Run {run_num}/{num_runs}...")
                metrics = run_benchmark(model_path, llama_cli, thermal_log_csv, context_length=ctx_len)
                metrics['run_number'] = run_num
                metrics['context_length'] = ctx_len
                metrics['perplexity'] = ppl
                results.append(metrics)
                run_metrics.append(metrics)
            
            tps_list = [m['tokens_per_sec'] for m in run_metrics]
            vram_list = [m['max_vram_mb'] for m in run_metrics]
            temp_list = [m['avg_temp_c'] for m in run_metrics]
            clock_list = [m['avg_clock_mhz'] for m in run_metrics]
            
            avg_tps = statistics.mean(tps_list)
            std_tps = statistics.stdev(tps_list) if len(tps_list) > 1 else 0.0
            ci_tps = 1.96 * (std_tps / math.sqrt(len(tps_list))) if len(tps_list) > 1 else 0.0
            
            avg_vram = statistics.mean(vram_list)
            std_vram = statistics.stdev(vram_list) if len(vram_list) > 1 else 0.0
            ci_vram = 1.96 * (std_vram / math.sqrt(len(vram_list))) if len(vram_list) > 1 else 0.0
            
            avg_temp = statistics.mean(temp_list)
            avg_clock = statistics.mean(clock_list)
            
            print(f"  --> Avg Throughput: {avg_tps:.2f} ± {ci_tps:.2f} tokens/s (95% CI)")
            print(f"  --> Avg Max VRAM: {avg_vram:.2f} ± {ci_vram:.2f} MB (95% CI)")
            print(f"  --> Avg GPU Temp: {avg_temp:.2f} °C, Avg SM Clock: {avg_clock:.2f} MHz\n")
        
    print(f"Saving results to {output_csv}")
    with open(output_csv, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model", "family", "context_length", "run_number", 
            "latency_sec", "tokens_per_sec", "max_vram_mb", 
            "avg_power_watts", "tokens_per_joule",
            "avg_temp_c", "avg_clock_mhz", "perplexity"
        ])
        writer.writeheader()
        for r in results:
            writer.writerow(r)
            
    print(f"Thermal log saved to {thermal_log_csv}")
    print("Benchmarking complete!")

if __name__ == "__main__":
    main()
