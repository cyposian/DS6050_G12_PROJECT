#!/bin/bash
# ============================================================
# G12 Fashion-MNIST: One-Time Rivanna Package Setup
# ============================================================
# Run this ONCE from a VS Code terminal connected to Rivanna.
# OR
# Run this ONCE from a Rivanna terminal.
# You must be on an interactive GPU node to verify CUDA works.
#
# Usage:
#   # 1. Connect VPN (Cisco AnyConnect) + VS Code Remote SSH
#   # 2. Get an interactive GPU session:
#   ijob -J setup -A shakeri_ds6050 -p gpu --gres=gpu:1 -t 01:00:00 --mem=8G -v
#   # 3. Run this script:
#   bash slurm/setup_env.sh
#
# What this does:
#   1. Loads the class Miniforge module (no separate conda env needed)
#   2. Installs PyTorch with CUDA 13.0 support
#   3. Installs all project dependencies
#   4. Verifies GPU visibility
#   5. Creates project directories
#
# NOTE: Packages install to ~/.local/lib/python3.12/ (pip user site).
#       Each team member must run this script independently.
# ============================================================

set -e

echo "============================================"
echo "G12 Rivanna Package Setup"
echo "============================================"

# ---- Step 1: Load Miniforge ----
echo "[1/5] Loading Miniforge..."
module purge
module load miniforge/24.11.3-py3.12

echo "  Python: $(which python)"
echo "  Version: $(python --version)"

# ---- Step 2: Install PyTorch with CUDA 13.0 ----
echo ""
echo "[2/5] Installing PyTorch (CUDA 13.0)..."
echo "  (This matches Rivanna's NVIDIA driver 580.95.05)"
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 \
    --index-url https://download.pytorch.org/whl/cu130

# ---- Step 3: Install project dependencies ----
echo ""
echo "[3/5] Installing project dependencies..."
pip install pyyaml scikit-learn xgboost thop

# ---- Step 4: Verify installation ----
echo ""
echo "[4/5] Verifying installation..."

python -c "
import torch
import torchvision
import yaml
import sklearn
import xgboost

print('Package versions:')
print(f'  PyTorch:      {torch.__version__}')
print(f'  TorchVision:  {torchvision.__version__}')
print(f'  CUDA build:   {torch.version.cuda}')
print(f'  scikit-learn: {sklearn.__version__}')
print(f'  XGBoost:      {xgboost.__version__}')
print()

if torch.cuda.is_available():
    print(f'GPU detected:   {torch.cuda.get_device_name(0)}')
    #print(f'torch.cuda.get_device_properties(0).total_memory / (1024**3))
    #device = torch.cuda.get_device_name(0)
    #mem_gb = torch.cuda.get_device_properties(0).total_mem / 1e9 *** DoesNotWork
    #mem_gb = device.total_memory / (1024**3)
    print(f'GPU memory:     {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB')
    #print(f'GPU memory:     {mem_gb:.1f} GB')
    print('STATUS:         *** GPU READY ***')
else:
    print('WARNING: No GPU detected!')
    print('  Make sure you ran this from an interactive GPU session:')
    print('  ijob -J setup -A shakeri_ds6050 -p gpu --gres=gpu:1 -t 01:00:00 --mem=8G -v')
"

# ---- Step 5: Create project directories ----
echo ""
echo "[5/5] Creating project directories..."
PROJ_DIR="$HOME/development"
mkdir -p "$PROJ_DIR"

# ---- Step 5a: Clone Git Repo ----
#echo ""
#cho "[5a/5] Cloning Git Repo https://github.com/d26clarke/DS6050_G12_PROJECT.git..."
EPO_URL="https://github.com/d26clarke/DS6050_G12_PROJECT.git"
TARGET_DIR="$PROJ_DIR"
#git clone "$REPO_URL" "$TARGET_DIR" This command syntax does not Work

#cd $PROJ_DIR
#git clone "$REPO_URL"

# --- These directories will be part of the git repo
PROJ_DIR="$PROJ_DIR/DS6050_G12_PROJECT"
mkdir -p "$PROJ_DIR/logs"
mkdir -p "$PROJ_DIR/results/curves"
mkdir -p "$PROJ_DIR/configs"
mkdir -p "$PROJ_DIR/data"
echo "  Project dir: $PROJ_DIR"
echo "  logs/:       ready"
echo "  results/:    ready"
echo "  configs/:    ready"
echo "  data/:       ready"

echo ""
echo "============================================"
echo "Setup complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  cd $PROJ_DIR"
echo "  python scripts/generate_ablation_configs_per_model.py"
echo "  sbatch slurm/run_single.slurm configs/baseline_simple_cnn.yaml"
echo "  squeue -u \$USER"
echo ""
echo "For future sessions, just load Miniforge:"
echo "  module load miniforge/24.11.3-py3.12"
echo ""
