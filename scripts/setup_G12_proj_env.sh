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
#   # 2. Run this script:
#   bash scripts/setup_env.sh
#
# What this does:
#   1. Clones the DS6050_G12_PROJECT repo
#   2. Creates required project directories
#
# ============================================================

set -e

echo "============================================"
echo "G12 Rivanna Package Setup"
echo "============================================"

# Get environemnt from user
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <your selected environment: dev | sit | prod >"
    exit 1
fi

# Set the user environment
SELECTED_USER_ENV=$1

# ---- Step 1: Create project directories ----
echo ""
echo "[1/2] Creating project directories..."
DEV_DIR="$HOME/$SELECTED_USER_ENV"
mkdir -p "$DEV_DIR"

# ---- Step 2: Clone Git Repo ----
echo ""
echo "[2/2] Cloning Git Repo https://github.com/d26clarke/DS6050_G12_PROJECT.git..."
REPO_URL="https://github.com/d26clarke/DS6050_G12_PROJECT.git"
PROJ_DIR="$DEV_DIR/DS6050_G12_PROJECT"

if [ -d "$PROJ_DIR" ]; then
    echo "  Repo already exists at $PROJ_DIR — skipping clone...."
else
    git clone "$REPO_URL" "$PROJ_DIR"
fi

# ---- Step 3: Create project subdirectories ----
echo ""
echo "[3/3] Creating project subdirectories..."
mkdir -p "$PROJ_DIR/logs"
mkdir -p "$PROJ_DIR/results/curves"
mkdir -p "$PROJ_DIR/configs"
mkdir -p "$PROJ_DIR/data"
echo "  Project dir: $PROJ_DIR"
echo "  $PROJ_DIR/logs/:       ready"
echo "  $PROJ_DIR/results/:    ready"
echo "  $PROJ_DIR/configs/:    ready"
echo "  $PROJ_DIR/data/:       ready"

echo ""
echo "============================================"
echo "Setup complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  cd $PROJ_DIR"
echo "  module purge "
echo "  module load miniforge/24.11.3-py3.12 "
echo "  python scripts/generate_ablation_configs_per_model.py"
echo "  scripts/slurm_runner.sh baseline_simple_cnn.yaml"
echo "  squeue -u \$USER"
echo ""
echo "For future sessions, just load Miniforge:"
echo "  module load miniforge/24.11.3-py3.12"
echo ""
