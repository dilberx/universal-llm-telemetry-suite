#!/usr/bin/env bash
set -e

echo "🚀 Automating LLM-Inference-Telemetry-Suite Setup..."

# 1. Create and Activate Virtual Environment
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
else
    echo "Virtual environment already exists. Skipping creation."
fi

echo "Activating virtual environment..."
source venv/bin/activate

# 2. Install Core Requirements
echo "Installing Core Dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# 3. Detect OS & Install Hardware-Optimized llama-cpp-python
OS_TYPE=$(uname)

if [ "$OS_TYPE" = "Darwin" ]; then
    echo "🍏 Apple Silicon (macOS) Detected. Installing Metal optimization..."
    CMAKE_ARGS="-DGGML_METAL=on" pip install --force-reinstall --no-cache-dir llama-cpp-python
elif [ "$OS_TYPE" = "Linux" ]; then
    echo "🐧 Linux Detected. Checking for NVIDIA GPU..."
    if command -v nvidia-smi &> /dev/null; then
        echo "🟢 NVIDIA GPU Found. Installing CUDA hardware optimization..."
        CMAKE_ARGS="-DGGML_CUDA=on" pip install --force-reinstall --no-cache-dir llama-cpp-python
    else
        echo "⚠️ No NVIDIA GPU found on Linux. Falling back to CPU version."
        pip install llama-cpp-python
    fi
else
    echo "⚠️ Unknown OS. Installing CPU fallback."
    pip install llama-cpp-python
fi

echo "✅ Setup Complete!"
echo "--------------------------------------------------------"
echo "To begin using the suite, run:"
echo "  source venv/bin/activate"
echo "  sudo ./venv/bin/python src/orchestrator.py --path ./llm_models/"
echo "--------------------------------------------------------"
