#!/bin/bash
# ============================================================
# G12 Fashion-MNIST: Batch Ablation Launcher
# ============================================================
# Submits all ablation configs to Rivanna in one shot.
# Each config becomes a separate SLURM job in the GPU queue.
#
# Usage:
#   chmod +x slurm/launch_ablations.sh
#   ./slurm/launch_ablations.sh
#
# What this does:
#   - Finds every .yaml file in configs/
#   - Submits each one as a separate GPU job via sbatch
#   - Each job runs independently and logs to results/all_experiments.csv
#   - You can monitor progress with: squeue -u $USER
#
# Tip: Run this once at the end of the day and let jobs run overnight.
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$HOME/ds6050/DS6050_G12_PROJECT"
CONFIG_DIR="$PROJECT_DIR/configs"
SLURM_SCRIPT="$PROJECT_DIR/slurm/run_single.slurm"

echo "============================================"
echo "G12 Batch Ablation Launcher"
echo "Project dir:  $PROJECT_DIR"
echo "Config dir:   $CONFIG_DIR"
echo "============================================"

# Count configs
NUM_CONFIGS=$(find "$CONFIG_DIR" -name "*.yaml" | wc -l)
echo "Found $NUM_CONFIGS config files to submit."
echo ""

if [ "$NUM_CONFIGS" -eq 0 ]; then
    echo "ERROR: No .yaml files found in $CONFIG_DIR"
    exit 1
fi

# Confirm before submitting (safety check)
read -p "Submit $NUM_CONFIGS jobs to the GPU queue? (y/n): " CONFIRM
if [ "$CONFIRM" != "y" ]; then
    echo "Aborted."
    exit 0
fi

# Submit each config as a separate job
SUBMITTED=0
for CONFIG in "$CONFIG_DIR"/*.yaml; do
    CONFIG_NAME=$(basename "$CONFIG" .yaml)

    # Route XGBoost and data-efficiency XGBoost jobs to CPU partition
    if [[ "$CONFIG_NAME" == *"xgboost"* ]]; then
        SLURM_TO_USE="$SCRIPT_DIR/run_xgboost.slurm"
    else
        SLURM_TO_USE="$SLURM_SCRIPT"
    fi

    # Submit with config as argument, override job name for easy tracking
    JOB_ID=$(sbatch \
        --job-name="g12_${CONFIG_NAME}" \
        "$SLURM_TO_USE" "$CONFIG" \
        | awk '{print $4}')

    echo "  Submitted: $CONFIG_NAME -> Job $JOB_ID ($(basename $SLURM_TO_USE))"
    SUBMITTED=$((SUBMITTED + 1))

    echo "Stopping after first submission for testing."         #delete later
    break  # This exits the loop after the first job is sent    #delete later
done

echo ""
echo "============================================"
echo "Submitted $SUBMITTED jobs."
echo "Monitor with:  squeue -u $USER"
echo "Cancel all:    scancel -u $USER -n g12_fmnist"
echo "============================================"
