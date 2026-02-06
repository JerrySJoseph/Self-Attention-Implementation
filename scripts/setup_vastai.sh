#!/bin/bash
# Vast.ai Setup Script
# Run this after SSH-ing into your Vast.ai instance

set -e

echo "=============================================="
echo "Setting up training environment on Vast.ai"
echo "=============================================="

# Check GPU
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements-cloud.txt

# Verify PyTorch CUDA
echo ""
echo "Verifying PyTorch CUDA..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Check data files
echo ""
echo "Checking data files..."
if [ -f "data/train.bin" ] && [ -f "data/val.bin" ]; then
    echo "Data files found!"
    ls -lh data/*.bin
else
    echo "WARNING: Data files not found in data/"
    echo "Upload your data files: train.bin, val.bin, tokenizer.model"
fi

echo ""
echo "=============================================="
echo "Setup complete!"
echo "=============================================="
echo ""
echo "To start training:"
echo "  python -m src.train_torch --config configs/model_config_h100.yaml"
echo ""
echo "To resume from checkpoint:"
echo "  python -m src.train_torch --config configs/model_config_h100.yaml --resume checkpoints/step_XXX"
echo ""
