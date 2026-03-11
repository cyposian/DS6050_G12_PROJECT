"""
G12 Fashion-MNIST: Generate Ablation Config Files
===================================================
Generates one YAML config per ablation experiment using the
One-At-a-Time (OAT) approach the professor recommended.

Baseline defaults (held constant unless ablated):
    optimizer=adamw, lr=0.001, batch_size=128, dropout=0.3,
    augmentation=none, initialization=he

Usage:
    python scripts/generate_ablation_configs.py

Output:
    Creates YAML files in configs/ directory.
    Also creates baseline configs for all model types.
"""

import os
import yaml
from copy import deepcopy

# ============================================================
# BASELINE CONFIG TEMPLATE
# ============================================================

BASELINE = {
    "experiment": {
        "name": "",   # Filled per experiment
        "seed": 42,
        "description": "",
    },
    "data": {
        "dataset": "fashion_mnist",
        "data_dir": "./data",
        "train_size": 50000,
        "val_size": 10000,
        "test_size": 10000,
        "num_workers": 4,
        "augmentation": "none",
        "normalize_mean": 0.2860,
        "normalize_std": 0.3530,
    },
    "model": {
        "type": "simple_cnn",
    },
    "training": {
        "optimizer": "adamw",
        "lr": 0.001,
        "weight_decay": 0.01,
        "momentum": 0.9,
        "batch_size": 128,
        "epochs": 50,
        "dropout": 0.3,
        "early_stopping": {
            "enabled": True,
            "patience": 5,
            "monitor": "val_loss",
            "min_delta": 0.001,
        },
        "initialization": "he",
    },
    "logging": {
        "results_csv": "results/all_experiments.csv",
        "save_model": False,
        "save_curves": True,
        "curves_dir": "results/curves",
    },
    "hardware": {
        "device": "auto",
        "log_gpu_memory": True,
        "log_flops": True,
    },
}

# ============================================================
# ABLATION DEFINITIONS (6 factors x 4 values = 24 experiments)
# ============================================================

ABLATIONS = {
    "optimizer": {
        "path": ["training", "optimizer"],
        "values": ["sgd", "sgd_momentum", "adam", "adamw"],
    },
    "lr": {
        "path": ["training", "lr"],
        "values": [0.1, 0.01, 0.001, 0.0001],
    },
    "batch_size": {
        "path": ["training", "batch_size"],
        "values": [32, 64, 128, 256],
    },
    "dropout": {
        "path": ["training", "dropout"],
        "values": [0.0, 0.25, 0.4, 0.5],
    },
    "augmentation": {
        "path": ["data", "augmentation"],
        "values": ["none", "hflip", "crop_flip"],
    },
    "initialization": {
        "path": ["training", "initialization"],
        "values": ["he", "xavier", "default"],
    },
}


def set_nested(d: dict, keys: list, value):
    """Set a value in a nested dict using a list of keys."""
    for key in keys[:-1]:
        d = d[key]
    d[keys[-1]] = value


def generate_configs(output_dir: str = "../configs"):
    os.makedirs(output_dir, exist_ok=True)
    count = 0

    # ---- Model baselines ----
    for model_type in ["xgboost", "mlp", "simple_cnn", "deeper_cnn"]:
        cfg = deepcopy(BASELINE)
        cfg["experiment"]["name"] = f"baseline_{model_type}"
        cfg["experiment"]["description"] = f"Baseline {model_type} with default hyperparameters"
        cfg["model"]["type"] = model_type

        if model_type == "xgboost":
            cfg["model"]["xgb_params"] = {
                "max_depth": 6,
                "n_estimators": 100,
                "learning_rate": 0.1,
            }

        if model_type == "deeper_cnn":
            cfg["model"]["use_residual"] = True

        path = os.path.join(output_dir, f"baseline_{model_type}.yaml")
        with open(path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        count += 1
        print(f"  Created: {path}")

    # ---- OAT Ablations on Simple CNN ----
    for factor_name, factor_def in ABLATIONS.items():
        for value in factor_def["values"]:
            # Clean value for filename
            val_str = str(value).replace(".", "").replace("-", "")
            cfg = deepcopy(BASELINE)
            cfg["experiment"]["name"] = f"ablation_{factor_name}_{val_str}"
            cfg["experiment"]["description"] = (
                f"Ablation: {factor_name}={value} (all others at default)"
            )
            cfg["model"]["type"] = "simple_cnn"
            set_nested(cfg, factor_def["path"], value)

            path = os.path.join(output_dir, f"ablation_{factor_name}_{val_str}.yaml")
            with open(path, "w") as f:
                yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
            count += 1
            print(f"  Created: {path}")

    # ---- Data efficiency experiments ----
    for n_samples in [1000, 5000, 10000, 50000]:
        for model_type in ["xgboost", "mlp", "simple_cnn"]:
            cfg = deepcopy(BASELINE)
            n_str = f"{n_samples // 1000}k"
            cfg["experiment"]["name"] = f"dataeff_{model_type}_{n_str}"
            cfg["experiment"]["description"] = (
                f"Data efficiency: {model_type} trained on {n_samples} samples"
            )
            cfg["model"]["type"] = model_type
            cfg["data"]["train_size"] = n_samples
            # Keep val at 10K for consistent evaluation
            cfg["data"]["val_size"] = 10000

            if model_type == "xgboost":
                cfg["model"]["xgb_params"] = {
                    "max_depth": 6,
                    "n_estimators": 100,
                    "learning_rate": 0.1,
                }

            path = os.path.join(output_dir, f"dataeff_{model_type}_{n_str}.yaml")
            with open(path, "w") as f:
                yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
            count += 1
            print(f"  Created: {path}")

    print(f"\nGenerated {count} config files in {output_dir}/")


if __name__ == "__main__":
    generate_configs()
