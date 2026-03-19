"""
setup_env.py — LLM Inference Telemetry Suite Environment Setup

Performs pre-flight checks before any benchmarking run:
  1. Sudo / powermetrics availability (Apple Silicon power telemetry)
  2. llama-cli binary resolution (Metal build recommended on macOS)
  3. Model directory scaffolding for all supported families

Run this once after cloning:
  python src/setup_env.py
"""

import os
import shutil
import subprocess
import sys
import platform


# ---------------------------------------------------------------------------
# 1.  Sudo / powermetrics check (Apple Silicon only)
# ---------------------------------------------------------------------------

def check_sudo_privileges() -> bool:
    """
    Returns True if the process has effective root (uid 0).
    On Apple Silicon, powermetrics requires root for SoC power telemetry.
    Without it, the suite falls back to NullProvider — benchmarks still run
    but Tokens-per-Joule will not be reported.
    """
    return os.geteuid() == 0


def check_powermetrics() -> bool:
    """Returns True if powermetrics is available on this system."""
    return shutil.which("powermetrics") is not None


def print_sudo_status() -> None:
    is_apple_silicon = sys.platform == "darwin" and platform.machine() == "arm64"
    if not is_apple_silicon:
        return  # Not relevant on non-Apple hosts

    has_root  = check_sudo_privileges()
    has_pm    = check_powermetrics()

    print("\n[Apple Silicon — Power Telemetry Check]")

    if has_root and has_pm:
        print("  ✅ sudo: active — AppleSiliconProvider will capture SoC power (Tokens/Joule enabled)")
    elif not has_root:
        print("  ⚠️  sudo: NOT active")
        print("     → Benchmarks will run normally BUT Tokens/Joule = 0.0 (NullProvider fallback)")
        print("     → To enable power telemetry, run: sudo python src/orchestrator.py --path <dir>")
    elif not has_pm:
        print("  ⚠️  powermetrics: not found (unexpected on macOS)")
        print("     → Falling back to NullProvider for power metrics")


# ---------------------------------------------------------------------------
# 2.  llama-cli / llama-perplexity check
# ---------------------------------------------------------------------------

_LLAMA_METAL_BUILD_STEPS = """
  ── How to build llama.cpp with Metal (Apple Silicon): ──────────────────────
  git clone https://github.com/ggerganov/llama.cpp
  cd llama.cpp
  cmake -B build -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON
  cmake --build build --config Release -j$(sysctl -n hw.ncpu)
  # Binaries land at: llama.cpp/build/bin/llama-cli
  #                   llama.cpp/build/bin/llama-perplexity
  # Option A: add to PATH:
  export PATH="$PWD/build/bin:$PATH"
  # Option B: place llama.cpp/ as a sibling of this repo (the orchestrator
  #           will detect it automatically at ../../llama.cpp/build/bin/).
  ─────────────────────────────────────────────────────────────────────────────
"""

_LLAMA_LINUX_BUILD_STEPS = """
  ── How to build llama.cpp on Linux (CUDA): ─────────────────────────────────
  git clone https://github.com/ggerganov/llama.cpp
  cd llama.cpp
  cmake -B build -DGGML_CUDA=ON
  cmake --build build --config Release -j$(nproc)
  export PATH="$PWD/build/bin:$PATH"
  ─────────────────────────────────────────────────────────────────────────────
"""


def _detect_metal_in_binary(binary_path: str) -> bool | None:
    """
    Probe llama-cli for Metal support by calling it with --version.
    Returns True if Metal is mentioned, False if not, None on failure.
    """
    try:
        result = subprocess.run(
            [binary_path, "--version"],
            capture_output=True, text=True, timeout=5,
        )
        combined = (result.stdout + result.stderr).lower()
        return "metal" in combined
    except Exception:
        return None


def check_llama_binaries(base_dir: str) -> dict:
    """
    Locate llama-cli and llama-perplexity. Returns a dict with paths and
    Metal status. Prints a human-readable status report.
    """
    sibling_dir  = os.path.normpath(os.path.join(base_dir, "../../llama.cpp/build/bin"))
    cli_path     = shutil.which("llama-cli") or os.path.join(sibling_dir, "llama-cli")
    ppl_path     = shutil.which("llama-perplexity") or os.path.join(sibling_dir, "llama-perplexity")

    cli_found = os.path.isfile(cli_path)
    ppl_found = os.path.isfile(ppl_path)

    print("\n[llama.cpp Binary Check]")

    if cli_found:
        metal_status = _detect_metal_in_binary(cli_path)
        if metal_status is True:
            print(f"  ✅ llama-cli        : {cli_path}  [Metal: YES]")
        elif metal_status is False:
            print(f"  ⚠️  llama-cli        : {cli_path}  [Metal: NO — rebuild recommended]")
            if sys.platform == "darwin":
                print(_LLAMA_METAL_BUILD_STEPS)
        else:
            print(f"  ✅ llama-cli        : {cli_path}  [Metal: unknown]")
    else:
        print("  ❌ llama-cli        : NOT FOUND")
        if sys.platform == "darwin" and platform.machine() == "arm64":
            print(_LLAMA_METAL_BUILD_STEPS)
        else:
            print(_LLAMA_LINUX_BUILD_STEPS)

    if ppl_found:
        print(f"  ✅ llama-perplexity : {ppl_path}")
    else:
        print("  ⚠️  llama-perplexity : NOT FOUND — perplexity tests will be skipped")

    return {"cli": cli_path if cli_found else None, "ppl": ppl_path if ppl_found else None}


# ---------------------------------------------------------------------------
# 3.  Model directory scaffolding
# ---------------------------------------------------------------------------

MODEL_FAMILIES = ["qwen", "mistral", "llama", "deepseek", "phi", "gemma"]


def scaffold_model_dirs(models_root: str) -> None:
    print(f"\n[Model Directory Scaffold → {models_root}]")
    for family in MODEL_FAMILIES:
        path = os.path.join(models_root, family)
        os.makedirs(path, exist_ok=True)
        keepfile = os.path.join(path, ".gitkeep")
        if not os.path.exists(keepfile):
            open(keepfile, "w").close()
        print(f"  ✅ {family}/.gitkeep")

    print(f"\n  Place your .gguf files inside the corresponding subdirectory.")
    print(f"  Example: {models_root}/qwen/qwen2.5-3b-instruct-q4_k_m.gguf")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def setup() -> None:
    base_dir   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "llm_models")

    print("=" * 60)
    print("  LLM Inference Telemetry Suite — Environment Setup")
    print("=" * 60)

    # Check 1: sudo / powermetrics
    print_sudo_status()

    # Check 2: llama-cli binaries
    binaries = check_llama_binaries(os.path.join(base_dir, "src"))

    # Check 3: model dirs
    scaffold_model_dirs(models_dir)

    # Summary
    print("\n" + "=" * 60)
    ready = binaries["cli"] is not None
    if ready:
        print("  ✅ Environment ready. Run the benchmark with:")
        print("     sudo python src/orchestrator.py --path llm_models/")
    else:
        print("  ⚠️  Environment NOT ready. Build llama.cpp first (see above).")
        print("     Then re-run: python src/setup_env.py")
    print("=" * 60)


if __name__ == "__main__":
    setup()
