#!/bin/bash

#SBATCH --account=shakeri_ds6050 
#SBATCH --job-name=python_test            # Job name
#SBATCH --output=logs/slurm_%j.out             # Standard output and error log (%j inserts the job ID)
#SBATCH --nodes=1                         # number of nodes
#SBATCH --ntasks=1                        # Run a single task
#SBATCH --nodes=1                         # number of nodes
#SBATCH --ntasks-per-node=1               # number of tasks per node
#SBATCH --cpus-per-task=1                 # Number of CPU cores per task
#SBATCH --time=00:03:00                   # Time limit hrs:min:sec
#SBATCH --gres=gpu:1                      # gpu devices per node
#SBATCH --partition=gpu                   # partition
#SBATCH --mem=4G                          # Memory Allocation
##SBATCH --partition=interactive                   # partition
#SBATCH --mail-user=thq3hn@virginia.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --no-requeue                 # prevents job returning to queue after node failure

set -e
set -o pipefail

# Get the requested model
SELECTED_MODEL=$1
echo "Selected Model: $SELECTED_MODEL"

# ---- Step 1: Load Miniforge ----
echo "[1/1] Loading Miniforge..."
module purge
module load miniforge/24.11.3-py3.12

echo "  Python: $(which python)"
echo "  Version: $(python --version)"

# ---- Step 2: Install PyTorch ----
#echo ""
#echo "[2/3] Installing PyTorch with CUDA 13.0..."
#echo "  (This matches Rivanna's NVIDIA driver 580.95.05)"
#pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu130
#pip install pyyaml scikit-learn xgboost thop

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
python -u ~/development/DS6050_G12_PROJECT/test/python/slurm_test.py --config $SELECTED_MODEL

EXIT_CODE=$?
echo "============================================"
echo "End time:      $(date)"
echo "Exit code:     $EXIT_CODE"
echo "============================================"
exit $EXIT_CODE

