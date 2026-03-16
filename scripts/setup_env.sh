#!/bin/bash
# setup_env.sh — Environment setup for distributed-HCVGLoc

set -e

echo "============================================================"
echo "  Setting up distributed-HCVGLoc environment"
echo "============================================================"

# Create conda env (adjust Python version if needed)
conda create -n dist-hcvgloc python=3.10 -y
conda activate dist-hcvgloc

# PyTorch with CUDA 12.1 (matches Ada 6000 driver)
pip install torch==2.2.0 torchvision==0.17.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies
pip install -r requirements.txt

# Install project as editable package
pip install -e .

echo ""
echo "Setup complete. Activate with: conda activate dist-hcvgloc"
echo "Verify GPU access: python -c 'import torch; print(torch.cuda.device_count(), \"GPUs available\")'"
