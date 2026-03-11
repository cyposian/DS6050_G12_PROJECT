#!/bin/bash
# ============================================================
# G12 Fashion-MNIST: XGBoost (CPU-only) SLURM Job
# ============================================================
# XGBoost does NOT need a GPU. Using the 'standard' partition
# avoids the GPU queue entirely — shorter wait times.
#
# Usage:
#   cd ${USER}/development/DS6050_G12_PROJECT
#   sbatch slurm/run_xgboost.slurm configs/baseline_xgboost.yaml
# ============================================================

#SBATCH --job-name=g12_xgb
#SBATCH --account=shakeri_ds6050
#SBATCH --partition=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/g12_xgb_%j.out
#SBATCH --error=logs/g12_xgb_%j.err
#SBATCH --export=NONE

module purge
module load miniforge/24.11.3-py3.12

CONFIG_FILE=${1:-configs/baseline_xgboost.yaml}

echo "============================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $SLURM_NODELIST"
echo "CPUs:          $SLURM_CPUS_PER_TASK"
echo "Config:        $CONFIG_FILE"
echo "Start time:    $(date)"
echo "============================================"

python scripts/train.py --config "$CONFIG_FILE"

EXIT_CODE=$?
echo "============================================"
echo "End time:      $(date)"
echo "Exit code:     $EXIT_CODE"
echo "============================================"
exit $EXIT_CODE
