"""
download_models.py — One-shot GGUF model downloader for the Telemetry Suite.

Downloads recommended benchmark models from HuggingFace Hub into the
correct llm_models/ subdirectories.

Usage:
    HF_HUB_ENABLE_HF_TRANSFER=1 ./venv/bin/python src/download_models.py

    # Download only specific families:
    ./venv/bin/python src/download_models.py --families qwen phi

    # Dry-run (show what would be downloaded):
    ./venv/bin/python src/download_models.py --dry-run

Requirements:
    pip install huggingface_hub hf_transfer tqdm
"""

import argparse
import os
import sys

try:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError
except ImportError:
    print("❌ huggingface_hub not installed. Run: pip install huggingface_hub hf_transfer")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Model registry
# All models are Q4_K_M by default (best T/J on Apple Silicon).
# Add Q5 / Q8 entries as additional rows to expand the matrix.
# ---------------------------------------------------------------------------

MODELS = [
    # ── Qwen 2.5 3B — Efficiency Frontier reference ───────────────────────
    {
        "family": "qwen",
        "repo":   "Qwen/Qwen2.5-3B-Instruct-GGUF",
        "file":   "qwen2.5-3b-instruct-q4_k_m.gguf",
        "desc":   "Qwen-3B Q4_K_M — peak efficiency anchor (0.9037 T/J on RTX 3080)",
    },
    {
        "family": "qwen",
        "repo":   "Qwen/Qwen2.5-3B-Instruct-GGUF",
        "file":   "qwen2.5-3b-instruct-q5_k_m.gguf",
        "desc":   "Qwen-3B Q5_K_M — Accuracy Cliff comparison",
    },
    {
        "family": "qwen",
        "repo":   "Qwen/Qwen2.5-3B-Instruct-GGUF",
        "file":   "qwen2.5-3b-instruct-q8_0.gguf",
        "desc":   "Qwen-3B Q8_0 — quality ceiling",
    },
    # ── Mistral 7B v0.2 — Memory bandwidth reference (legacy) ─────────────
    {
        "family": "mistral",
        "repo":   "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        "file":   "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "desc":   "Mistral-7B v0.2 Q4_K_M — memory-bandwidth-bound baseline",
    },
    # ── Phi-4-mini — Ultra-Mobile Frontier ───────────────────────────────
    {
        "family": "phi",
        "repo":   "bartowski/microsoft_Phi-4-mini-instruct-GGUF",
        "file":   "microsoft_Phi-4-mini-instruct-Q4_K_M.gguf",
        "desc":   "Phi-4-mini Q4_K_M — Ultra-Mobile Frontier test (3.8B)",
    },
    # ── DeepSeek-R1-Distill — Reasoning Efficiency ───────────────────────
    {
        "family": "deepseek",
        "repo":   "bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF",
        "file":   "DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf",
        "desc":   "DeepSeek-R1-Distill-7B Q4_K_M — Reasoning Efficiency test",
    },

    # ═══════════════════════════════════════════════════════════════════════
    # 3080 Parity Models (for direct cross-platform comparison)
    # ═══════════════════════════════════════════════════════════════════════

    # ── Llama 3.1 8B — Same model as RTX 3080 baseline ───────────────────
    {
        "family": "llama",
        "repo":   "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        "file":   "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "desc":   "Llama-3.1-8B Q4_K_M — 3080 parity baseline (~4.92 GB)",
    },
    # ── Mistral 7B v0.3 Q5_K_M — Exact 3080 baseline match ──────────────
    {
        "family": "mistral7b",
        "repo":   "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
        "file":   "Mistral-7B-Instruct-v0.3-Q5_K_M.gguf",
        "desc":   "Mistral-7B v0.3 Q5_K_M — exact 3080 parity model (~5.13 GB)",
    },

    # ═══════════════════════════════════════════════════════════════════════
    # Apple Silicon Recommendations (find best model for the hardware)
    # ═══════════════════════════════════════════════════════════════════════

    # ── Qwen 2.5 7B — Larger Qwen for scaling analysis ───────────────────
    {
        "family": "qwen",
        "repo":   "bartowski/Qwen2.5-7B-Instruct-GGUF",
        "file":   "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        "desc":   "Qwen-7B Q4_K_M — scaling comparison vs 3B efficiency champion",
    },
    # ── Gemma 2 9B — Google's best mid-size model ────────────────────────
    {
        "family": "gemma",
        "repo":   "bartowski/gemma-2-9b-it-GGUF",
        "file":   "gemma-2-9b-it-Q4_K_M.gguf",
        "desc":   "Gemma-2-9B Q4_K_M — Google mid-size quality contender (~5.76 GB)",
    },
    # ── Falcon 11B — TII's dense architecture ────────────────────────────
    {
        "family": "falcon",
        "repo":   "bartowski/falcon-11B-GGUF",
        "file":   "falcon-11B-Q4_K_M.gguf",
        "desc":   "Falcon-11B Q4_K_M — dense architecture efficiency test (~6.84 GB)",
    },
]

# ---------------------------------------------------------------------------
# Downloader
# ---------------------------------------------------------------------------


def download_model(entry: dict, models_root: str, dry_run: bool = False) -> bool:
    dest_dir  = os.path.join(models_root, entry["family"])
    dest_file = os.path.join(dest_dir, entry["file"])
    os.makedirs(dest_dir, exist_ok=True)

    if os.path.exists(dest_file):
        size_gb = os.path.getsize(dest_file) / (1024 ** 3)
        print(f"  ✅ Already downloaded: {entry['file']} ({size_gb:.2f} GB)")
        return True

    if dry_run:
        print(f"  [DRY-RUN] Would download: {entry['repo']}/{entry['file']}")
        print(f"            → {dest_file}")
        return True

    print(f"  ⬇  Downloading {entry['file']} ...")
    print(f"     {entry['desc']}")
    try:
        path = hf_hub_download(
            repo_id=entry["repo"],
            filename=entry["file"],
            local_dir=dest_dir,
            local_dir_use_symlinks=False,
        )
        size_gb = os.path.getsize(path) / (1024 ** 3)
        print(f"  ✅ Saved → {path}  ({size_gb:.2f} GB)\n")
        return True
    except (EntryNotFoundError, RepositoryNotFoundError) as e:
        print(f"  ⚠️  Not found on HuggingFace: {e}")
        print(f"     Try searching manually: https://huggingface.co/models?search={entry['file']}\n")
        return False
    except Exception as e:
        print(f"  ❌ Download failed: {e}\n")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="GGUF model downloader for Telemetry Suite")
    parser.add_argument(
        "--families",
        nargs="*",
        default=None,
        help="Filter to specific families (e.g. qwen phi deepseek). Default: all.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would be downloaded.")
    args = parser.parse_args()

    base_dir    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_root = os.path.join(base_dir, "llm_models")

    # Check for HF transfer acceleration
    if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") == "1":
        print("✅ HF_HUB_ENABLE_HF_TRANSFER=1 — using accelerated multi-part downloads\n")
    else:
        print("ℹ️  Set HF_HUB_ENABLE_HF_TRANSFER=1 for faster downloads\n")

    targets = [
        m for m in MODELS
        if args.families is None or m["family"] in args.families
    ]

    print(f"{'DRY-RUN: ' if args.dry_run else ''}Downloading {len(targets)} model file(s) to {models_root}/\n")

    successes = 0
    for entry in targets:
        if download_model(entry, models_root, dry_run=args.dry_run):
            successes += 1

    print(f"\n{'─'*50}")
    print(f"Done: {successes}/{len(targets)} model(s) ready.")
    print(f"\nNext step:")
    print(f"  sudo ./venv/bin/python src/orchestrator.py --path {models_root}/")


if __name__ == "__main__":
    main()
