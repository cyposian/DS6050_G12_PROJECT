#!/bin/bash
# ============================================================
# G12 Fashion-MNIST: Single GPU Experiment By Model (Rivanna/Afton)
# ============================================================
# Assumptions (You are logged onto a Rivanna terminal and are in the ${USER}/development/DS6050_G12_PROJECT directory):
#
#
#   # 1. This script will Request an interactive GPU session to install packages:
#   ijob -J setup -A shakeri_ds6050 -p gpu --gres=gpu:1 -t 01:00:00 --mem=8G -v
#
#   # 2. This script will Load Miniforge and install packages: 
#   module load miniforge/24.11.3-py3.12
#   pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu130
#   pip install pyyaml scikit-learn xgboost thop
#
#   # 3. This script will verify GPU:
#   python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
#
#   # 4. Create project directories:
#   mkdir -p ~/DS6050/Project/logs ~/DS6050/Project/results/curves
#
# Usage:
#   cd ~/DS6050/Project
#   sbatch slurm/run_model.slurm configs/{YOUR_SELECTED_MODEL}.yaml
#
# Monitor:
#   squeue -u $USER              # Check queue status
#   scancel JOB_ID               # Cancel a specific job
#   seff JOB_ID                  # Efficiency report after completion
#   cat logs/g12_JOBID.out       # Read stdout
# ============================================================

set -e

# Check for required selected model
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <your_selected_model>"
    exit 1
fi

# Get the requested model
SELECTED_MODEL=$1

echo "============================================"
echo "You are preparing to execute run_model.slurm for model: ${SELECTED_MODEL}"
echo "============================================"

# ---- Step 1: Request an interactive GPU session to install packages ----
echo "[1/3] Requesting an interactive GPU session..."
ijob -J setup -A shakeri_ds6050 -p gpu --gres=gpu:1 -t 01:00:00 --mem=8G -v

# ---- Step 2: Load Miniforge ----
echo "[2/3] Loading Miniforge..."
module purge
module load miniforge/24.11.3-py3.12

echo "  Python: $(which python)"
echo "  Version: $(python --version)"

# ---- Step 3: Install PyTorch ----
echo ""
echo "[3/3] Installing PyTorch with CUDA 13.0..."
echo "  (This matches Rivanna's NVIDIA driver 580.95.05)"
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu130
pip install pyyaml scikit-learn xgboost thop

# ---- Verify GPU ----
echo "============================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $SLURM_NODELIST"
echo "Partition:     $SLURM_JOB_PARTITION"
echo "Start time:    $(date)"
echo "--------------------------------------------"
echo "Python:        $(which python)"
echo "PyTorch:       $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA avail:    $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU:           $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")')"
echo "CUDA version:  $(python -c 'import torch; print(torch.version.cuda)')"
echo "============================================"

# ---- Read Config ----
CONFIG_FILE=${1:-configs/${SELECTED_MODEL}.yaml}
echo "Config:        $CONFIG_FILE"

# ---- Run SLURM ----
sbatch ~/development/DS6050_G12_PROJECT/slurm/run-model.slurm ${CONFIG_FILE}

EXIT_CODE=$?
echo "============================================"
echo "End time:      $(date)"
echo "Exit code:     $EXIT_CODE"
echo "============================================"
exit $EXIT_CODE
