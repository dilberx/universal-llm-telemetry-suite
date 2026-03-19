#!/usr/bin/env bash
# =============================================================================
# run_benchmark.sh — LLM Inference Telemetry Suite Benchmark Runner
#
# Usage:
#   sudo ./run_benchmark.sh [MODEL_DIR]
#
# Arguments:
#   MODEL_DIR    Path to directory of .gguf files (default: ./llm_models)
#
# What this script does:
#   1. Pre-flight check via setup_env.py
#   2. Runs the benchmark orchestrator with sudo (needed for powermetrics)
#   3. Tees all stdout + stderr to results/errors.log, highlighting errors
#   4. Runs the visualizer in aggregate mode on success
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${1:-${SCRIPT_DIR}/llm_models}"
RESULTS_DIR="${SCRIPT_DIR}/results"
LOG_FILE="${RESULTS_DIR}/errors.log"
TIMESTAMP="$(date '+%Y-%m-%d_%H-%M-%S')"

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
RED='\033[0;31m'; YELLOW='\033[1;33m'; GREEN='\033[0;32m'; NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

mkdir -p "${RESULTS_DIR}"

# --------------------------------------------------------------------------
# 1. Pre-flight
# --------------------------------------------------------------------------
log_info "Running environment pre-flight (setup_env.py)..."
python3 "${SCRIPT_DIR}/src/setup_env.py" 2>&1 | tee -a "${LOG_FILE}"

if ! command -v llama-cli &>/dev/null; then
    # Try sibling path
    SIBLING_CLI="${SCRIPT_DIR}/../llama.cpp/build/bin/llama-cli"
    if [[ ! -f "${SIBLING_CLI}" ]]; then
        log_error "llama-cli not found. Build llama.cpp first — see setup_env.py output above."
        log_error "Blocked. Exiting."
        exit 1
    fi
fi

# --------------------------------------------------------------------------
# 2. Sudo check — powermetrics requires root on Apple Silicon
# --------------------------------------------------------------------------
if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
    if [[ "${EUID}" -ne 0 ]]; then
        log_warn "Not running as root. Tokens/Joule will be 0.0 (NullProvider fallback)."
        log_warn "For full power telemetry, run: sudo ./run_benchmark.sh ${MODEL_DIR}"
    else
        log_info "sudo: active. AppleSiliconProvider will capture SoC power."
    fi
fi

# --------------------------------------------------------------------------
# 3. Run benchmark — tee stdout + stderr to errors.log
# --------------------------------------------------------------------------
log_info "Starting benchmark run at ${TIMESTAMP}"
log_info "Model path : ${MODEL_DIR}"
log_info "Log file   : ${LOG_FILE}"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "${LOG_FILE}"
echo "  Benchmark Run: ${TIMESTAMP}" | tee -a "${LOG_FILE}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "${LOG_FILE}"

# Stream orchestrator output; filter errors/warnings into errors.log with markers
python3 "${SCRIPT_DIR}/src/orchestrator.py" --path "${MODEL_DIR}" 2>&1 \
    | tee >(grep -E "⚠️|❌|Error|error|timed out|OOM|out of memory|VRAM" \
            | while IFS= read -r line; do
                echo "[ERROR_MONITOR ${TIMESTAMP}] ${line}" >> "${LOG_FILE}"
              done) \
    | tee -a "${LOG_FILE}"

BENCH_EXIT="${PIPESTATUS[0]}"

echo ""
if [[ "${BENCH_EXIT}" -eq 0 ]]; then
    log_info "Benchmark complete ✅"
else
    log_error "Benchmark exited with code ${BENCH_EXIT}. Check ${LOG_FILE} for details."
fi

# --------------------------------------------------------------------------
# 4. Visualizer — aggregate mode
# --------------------------------------------------------------------------
if [[ "${BENCH_EXIT}" -eq 0 ]]; then
    log_info "Generating cross-platform dashboard (aggregate mode)..."
    python3 "${SCRIPT_DIR}/src/visualizer.py" \
        --results-dir "${RESULTS_DIR}" \
        --aggregate 2>&1 | tee -a "${LOG_FILE}"

    log_info "Dashboard generated."
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Full log → ${LOG_FILE}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

exit "${BENCH_EXIT}"
