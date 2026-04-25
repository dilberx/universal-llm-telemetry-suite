"""
LLM Inference Benchmark — 1-Click CLI

Usage:
    # Auto-detect hardware, download models, run everything:
    python -m llm_bench run

    # Just setup (detect hardware + download models):
    python -m llm_bench setup

    # Run specific experiment:
    python -m llm_bench run --experiment kv_cache
    python -m llm_bench run --experiment token_confidence
    python -m llm_bench run --experiment head_pruning
    python -m llm_bench run --experiment all

    # Generate charts from existing data:
    python -m llm_bench charts

    # Show system info:
    python -m llm_bench info
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path


# ── Constants ────────────────────────────────────────────────────────

MODELS = {
    "tiny": {
        "name": "Qwen/Qwen2.5-Coder-0.5B",
        "size_gb": 1.0,
        "min_vram_gb": 2,
        "description": "Smallest code model — runs on any GPU",
    },
    "small": {
        "name": "Qwen/Qwen2.5-Coder-1.5B",
        "size_gb": 3.0,
        "min_vram_gb": 4,
        "description": "Small code model — good for 4GB+ VRAM",
    },
    "medium": {
        "name": "microsoft/phi-2",
        "size_gb": 5.5,
        "min_vram_gb": 8,
        "description": "Medium model — needs 8GB VRAM",
    },
}

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║          LLM Inference Optimization Benchmark               ║
║          ─────────────────────────────────────               ║
║  Systematic comparison of inference optimization techniques  ║
║  KV cache · Head pruning · Token confidence · Quantization   ║
╚══════════════════════════════════════════════════════════════╝
"""


# ── Hardware Detection ───────────────────────────────────────────────

def detect_hardware() -> dict:
    """Auto-detect hardware capabilities."""
    info = {
        "platform": platform.system(),
        "cpu": platform.processor() or "unknown",
        "cpu_cores": os.cpu_count() or 1,
        "python": platform.python_version(),
        "gpu_available": False,
        "gpu_name": "N/A",
        "gpu_vram_gb": 0,
        "gpu_count": 0,
        "mps_available": False,
        "recommended_model": "tiny",
        "device": "cpu",
    }

    # Check CUDA
    try:
        import torch
        if torch.cuda.is_available():
            info["gpu_available"] = True
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_vram_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / (1024**3), 1
            )
            info["device"] = "cuda"

            # Recommend model based on VRAM
            if info["gpu_vram_gb"] >= 8:
                info["recommended_model"] = "medium"
            elif info["gpu_vram_gb"] >= 4:
                info["recommended_model"] = "small"
            else:
                info["recommended_model"] = "tiny"

        # Check MPS (Apple Silicon)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            info["mps_available"] = True
            if not info["gpu_available"]:
                info["device"] = "mps"
                info["gpu_name"] = "Apple Silicon (MPS)"
                info["recommended_model"] = "small"

    except ImportError:
        pass

    return info


def print_hardware(info: dict):
    """Pretty-print hardware info."""
    print("\n🖥️  System Information")
    print(f"   Platform:  {info['platform']}")
    print(f"   CPU:       {info['cpu']} ({info['cpu_cores']} cores)")
    print(f"   Python:    {info['python']}")

    if info["gpu_available"]:
        print(f"   GPU:       {info['gpu_name']}")
        print(f"   VRAM:      {info['gpu_vram_gb']} GB")
        print(f"   GPUs:      {info['gpu_count']}")
    elif info["mps_available"]:
        print(f"   GPU:       Apple Silicon (MPS)")
    else:
        print(f"   GPU:       None detected (CPU-only mode)")

    print(f"   Device:    {info['device']}")
    model = MODELS[info["recommended_model"]]
    print(f"   Model:     {model['name']} (auto-selected for your hardware)")


# ── Model Management ─────────────────────────────────────────────────

def ensure_model(model_key: str, cache_dir: str = "./models") -> str:
    """Download model if not already cached. Returns model path."""
    model = MODELS[model_key]
    model_name = model["name"]
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    model_dir = cache / model_name.replace("/", "--")
    if model_dir.exists() and any(model_dir.iterdir()):
        print(f"   ✓ Model cached: {model_name}")
        return str(model_dir)

    print(f"   ⬇ Downloading {model_name} (~{model['size_gb']}GB)...")

    try:
        from huggingface_hub import snapshot_download
        path = snapshot_download(
            model_name,
            local_dir=str(model_dir),
            local_dir_use_symlinks=False,
        )
        print(f"   ✓ Downloaded to {path}")
        return path
    except ImportError:
        # Fallback: use transformers
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            print(f"   Using transformers to download...")
            AutoTokenizer.from_pretrained(model_name, cache_dir=str(cache))
            AutoModelForCausalLM.from_pretrained(
                model_name, cache_dir=str(cache),
                torch_dtype="auto",
            )
            print(f"   ✓ Downloaded via transformers")
            return model_name  # transformers will use its cache
        except ImportError:
            print(f"   ⚠ Install huggingface_hub or transformers to auto-download models")
            print(f"   Run: pip install huggingface_hub")
            return model_name


# ── Dependency Check ─────────────────────────────────────────────────

def check_dependencies() -> bool:
    """Check and install missing dependencies."""
    required = ["torch", "numpy", "matplotlib"]
    optional = ["huggingface_hub", "transformers"]
    missing = []

    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"\n⚠ Missing required packages: {', '.join(missing)}")
        print(f"  Install with: pip install {' '.join(missing)}")
        return False

    for pkg in optional:
        try:
            __import__(pkg)
        except ImportError:
            print(f"  ℹ Optional: {pkg} not installed (needed for model download)")

    return True


# ── Experiment Runners ───────────────────────────────────────────────

def run_experiment(name: str, device: str = "cpu"):
    """Run a specific experiment."""
    print(f"\n🔬 Running experiment: {name}")
    print(f"   Device: {device}")
    print("─" * 50)

    if name == "token_confidence":
        from src.experiments.exp_token_confidence import run_token_confidence_experiment
        return run_token_confidence_experiment()

    elif name == "head_pruning":
        from src.experiments.exp_attention_heads import run_attention_head_experiment
        return run_attention_head_experiment()

    elif name == "kv_cache":
        from src.kv_cache_bench.benchmark import KVCacheBenchmark, BenchmarkConfig
        config = BenchmarkConfig(device=device)
        bench = KVCacheBenchmark(config)
        report = bench.run()
        report.save("./reports/kv_cache/latest_results.json")
        print(report.summary_table())
        return report

    else:
        print(f"  ⚠ Unknown experiment: {name}")
        return None


def run_all(device: str = "cpu"):
    """Run all experiments."""
    experiments = ["kv_cache", "token_confidence", "head_pruning"]
    results = {}

    for exp in experiments:
        try:
            results[exp] = run_experiment(exp, device)
        except Exception as e:
            print(f"  ✗ {exp} failed: {e}")
            results[exp] = None

    return results


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="llm_bench",
        description="LLM Inference Optimization Benchmark — 1-click setup & run",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # setup
    setup_parser = subparsers.add_parser("setup", help="Detect hardware and download models")
    setup_parser.add_argument("--model", choices=list(MODELS.keys()), help="Override model selection")

    # run
    run_parser = subparsers.add_parser("run", help="Run benchmark experiments")
    run_parser.add_argument("--experiment", "-e", default="all",
                            choices=["all", "kv_cache", "token_confidence", "head_pruning"],
                            help="Which experiment to run")
    run_parser.add_argument("--device", default=None, help="Override device (cuda/cpu/mps)")
    run_parser.add_argument("--model", choices=list(MODELS.keys()), help="Override model")

    # charts
    subparsers.add_parser("charts", help="Generate charts from existing data")

    # info
    subparsers.add_parser("info", help="Show system information")

    args = parser.parse_args()

    if not args.command:
        print(BANNER)
        parser.print_help()
        return

    print(BANNER)

    if args.command == "info":
        hw = detect_hardware()
        print_hardware(hw)

    elif args.command == "setup":
        print("🔧 Setting up benchmark environment...\n")

        # 1. Check deps
        print("1. Checking dependencies...")
        if not check_dependencies():
            sys.exit(1)
        print("   ✓ All required packages available\n")

        # 2. Detect hardware
        print("2. Detecting hardware...")
        hw = detect_hardware()
        print_hardware(hw)

        # 3. Download model
        model_key = getattr(args, "model", None) or hw["recommended_model"]
        print(f"\n3. Ensuring model is available...")
        ensure_model(model_key)

        # 4. Save config
        config = {"hardware": hw, "model": model_key, "device": hw["device"]}
        config_path = Path("./benchmark_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"\n   ✓ Config saved to {config_path}")

        print("\n✅ Setup complete! Run experiments with: python -m llm_bench run")

    elif args.command == "run":
        # Load or create config
        config_path = Path("./benchmark_config.json")
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            device = args.device or config.get("device", "cpu")
        else:
            hw = detect_hardware()
            print_hardware(hw)
            device = args.device or hw["device"]

        start = time.time()

        if args.experiment == "all":
            run_all(device)
        else:
            run_experiment(args.experiment, device)

        elapsed = time.time() - start
        print(f"\n⏱ Total time: {elapsed:.1f}s")

        # Auto-generate charts
        print("\n📊 Generating charts...")
        try:
            from src.experiments.visualize import generate_all_charts
            generate_all_charts()
        except Exception as e:
            print(f"  ⚠ Chart generation failed: {e}")

        print("\n✅ Benchmark complete! Results in ./reports/")

    elif args.command == "charts":
        from src.experiments.visualize import generate_all_charts
        generate_all_charts()


if __name__ == "__main__":
    main()
