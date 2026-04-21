#!/bin/bash
# ============================================================
# G12 Fashion-MNIST: Single GPU Experiment By Model (Rivanna/Afton)
# ============================================================
# Assumptions (You are logged onto a Rivanna terminal and are in the ${USER}/development/DS6050_G12_PROJECT directory):
#
# Usage:
#   sbatch slurm/run_single.slurm configs/{YOUR_SELECTED_MODEL}.yaml
#
# Monitor:
#   squeue -u $USER              # Check queue status
#   scancel JOB_ID               # Cancel a specific job
#   seff JOB_ID                  # Efficiency report after completion
#   cat logs/g12_JOBID.out       # Read stdout
# ============================================================

# ---- SLURM Directives ----
#SBATCH --job-name=g12_fmnist
#SBATCH --account=shakeri_ds6050
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100_80gb
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --mail-user=thq3hn@virginia.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=logs/g12_%j.out
#SBATCH --error=logs/g12_%j.err
#SBATCH --export=NONE

# Get the config file
CONFIG_FILE=$1

# ---- Step 1: Load Miniforge ----
echo "[1/1] Loading Miniforge..."
module purge
module load miniforge/24.11.3-py3.12

echo "  Python: $(which python)"
echo "  Version: $(python --version)"

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

# ---- Run Experiment ----
python ~/ds6050/DS6050_G12_PROJECT/scripts/train_models.py --config $CONFIG_FILE

EXIT_CODE=$?
echo "============================================"
echo "End time:      $(date)"
echo "Exit code:     $EXIT_CODE"
echo "============================================"
exit $EXIT_CODE
