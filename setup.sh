#!/bin/bash
# Setup script for Math-PRM Verifier
# This script installs all dependencies for training and inference

set -e  # Exit on error

echo "=============================================="
echo "Math-PRM Verifier Setup"
echo "=============================================="

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
REQUIRED_VERSION="3.10"

echo "Detected Python version: $PYTHON_VERSION"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "Error: Python >= 3.10 is required"
    exit 1
fi

# Check for CUDA
echo ""
echo "Checking CUDA availability..."
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
    echo "CUDA Version: $CUDA_VERSION"
    HAS_CUDA=true
else
    echo "CUDA not detected. Will install CPU-only versions."
    HAS_CUDA=false
fi

# Create virtual environment if not exists
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch (with or without CUDA)
echo ""
echo "Installing PyTorch..."
if [ "$HAS_CUDA" = true ]; then
    # Install CUDA version
    pip install torch --index-url https://download.pytorch.org/whl/cu121
else
    # Install CPU version
    pip install torch --index-url https://download.pytorch.org/whl/cpu
fi

# Install main dependencies
echo ""
echo "Installing main dependencies..."
pip install numpy"<2.0.0"
pip install transformers>=4.36.0 datasets>=2.14.0 accelerate>=0.25.0
pip install trl>=0.7.0 peft>=0.7.0
pip install pyyaml>=6.0 tqdm>=4.66.0

# Install bitsandbytes (CUDA only)
if [ "$HAS_CUDA" = true ]; then
    echo ""
    echo "Installing bitsandbytes..."
    pip install bitsandbytes>=0.41.0
fi

# Install Unsloth (CUDA only, for training)
if [ "$HAS_CUDA" = true ]; then
    echo ""
    echo "Installing Unsloth..."
    # Detect CUDA version for appropriate Unsloth installation
    CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d'.' -f1)

    if [ "$CUDA_MAJOR" -ge "12" ]; then
        pip install "unsloth[cu121-ampere-torch220] @ git+https://github.com/unslothai/unsloth.git"
    else
        pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
    fi
fi

# Install vLLM (for inference)
if [ "$HAS_CUDA" = true ]; then
    echo ""
    echo "Installing vLLM..."
    pip install vllm>=0.2.0
else
    echo ""
    echo "Note: vLLM requires CUDA. Skipping vLLM installation."
    echo "You can run training without vLLM, but inference will require a CUDA-enabled system."
fi

# Install the package in editable mode
echo ""
echo "Installing math-prm-verifier package..."
pip install -e .

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p logs checkpoints eval_results

echo ""
echo "=============================================="
echo "Setup complete!"
echo "=============================================="
echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To start training, run:"
echo "  make train CONFIG=configs/default.yaml"
echo ""
echo "To run inference, run:"
echo "  make inference CHECKPOINT=checkpoints/merged_model"
echo ""

if [ "$HAS_CUDA" = false ]; then
    echo "WARNING: CUDA not detected. Training will be very slow on CPU."
    echo "Consider using a CUDA-enabled environment (Google Colab, Lambda Labs, etc.)"
fi
