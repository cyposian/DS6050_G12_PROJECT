"""
G12 Fashion-MNIST: Unified Training Harness
============================================
One script to train any model (XGBoost, MLP, CNN, pretrained) with
consistent logging, early stopping, and metrics collection.

Usage:
    python scripts/train_models.py --config configs/baseline_simple_cnn.yaml

What this script does:
    1. Reads a YAML config file
    2. Loads and splits Fashion-MNIST
    3. Builds the requested model
    4. Trains with early stopping on validation LOSS
    5. Evaluates on the validation set (test set only for final eval)
    6. Logs one row to results/all_experiments.csv
    7. Optionally saves per-epoch loss/accuracy curves

Dependencies:
    pip install torch torchvision pyyaml scikit-learn xgboost thop
"""

import argparse
import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# ============================================================
# 1. CONFIG LOADING
# ============================================================

def load_config(config_path: str) -> dict:
    """Load YAML config and return as nested dict."""
    with open(config_path, "r") as f:
        config: dict = yaml.safe_load(f)
    return config


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

def set_seed(seed: int):
    """Set random seeds for reproducibility across all libraries."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# 3. DATA LOADING
# ============================================================

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def get_transforms(config: dict):
    """Build train/val transforms from config."""
    mean: float = config["data"].get("normalize_mean", 0.2860)
    std: float = config["data"].get("normalize_std", 0.3530)

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,)),
    ])

    aug = config["data"].get("augmentation", "none")
    if aug == "hflip":
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((mean,), (std,)),
        ])
    elif aug == "crop_flip":
        train_transform = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((mean,), (std,)),
        ])
    else:  # "none"
        train_transform = val_transform

    return train_transform, val_transform


def load_data(config: dict):
    """
    Load Fashion-MNIST and split into train/val/test.

    Returns DataLoaders for train, val, and the raw val dataset
    (for XGBoost which doesn't use DataLoaders).
    """
    data_dir: str = config["data"].get("data_dir", "./data")
    batch_size: int = config["training"]["batch_size"]
    num_workers: int = config["data"].get("num_workers", 4)
    train_size: int = config["data"].get("train_size", 50000)
    val_size: int = config["data"].get("val_size", 10000)

    train_transform, val_transform = get_transforms(config)

    # Full training set (60K)
    full_train = datasets.FashionMNIST(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    # We also need val with val_transform (no augmentation)
    full_train_val = datasets.FashionMNIST(
        root=data_dir, train=True, download=True, transform=val_transform
    )

    # Stratified split using fixed seed
    generator = torch.Generator().manual_seed(config["experiment"]["seed"])
    indices = torch.randperm(len(full_train), generator=generator).tolist()
    train_indices: list = indices[:train_size]
    val_indices: list = indices[train_size : train_size + val_size]

    train_subset = Subset(full_train, train_indices)
    val_subset = Subset(full_train_val, val_indices)

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_subset, batch_size=256, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    # Test set (only for final evaluation — do NOT use during development)
    test_dataset = datasets.FashionMNIST(
        root=data_dir, train=False, download=True, transform=val_transform
    )
    test_loader = DataLoader(
        test_dataset, batch_size=256, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    # Dataset verification (professor's requirement)
    print(f"[DATA] Train: {len(train_subset)}, Val: {len(val_subset)}, "
          f"Test: {len(test_dataset)}")

    # Verify class balance
    train_labels: list = [full_train.targets[i].item() for i in train_indices]
    val_labels: list = [full_train.targets[i].item() for i in val_indices]
    print(f"[DATA] Train class counts: {np.bincount(train_labels)}")
    print(f"[DATA] Val class counts:   {np.bincount(val_labels)}")
    print(f"[DATA] Trivial baseline (random): {1.0/10:.1%}")

    return train_loader, val_loader, test_loader, train_indices, val_indices


# ============================================================
# 4. MODEL DEFINITIONS
# ============================================================

class MLP(nn.Module):
    """784 -> 512 -> 256 -> 10 with ReLU and dropout."""

    def __init__(self, dropout: float = 0.5):
        super().__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.net(self.flatten(x))


class SimpleCNN(nn.Module):
    """2 conv blocks (16, 32 filters), BatchNorm, ReLU, MaxPool, 2 FC layers."""

    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 1 -> 16 channels
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28x28 -> 14x14

            # Block 2: 16 -> 32 channels
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14 -> 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class DeeperCNN(nn.Module):
    """
    3 conv blocks (32, 64, 128 filters) with optional residual connections.
    This is the 'mini-ResNet' the professor suggested.
    """

    def __init__(self, dropout: float = 0.3, use_residual: bool = True):
        super().__init__()
        self.use_residual = use_residual

        # Block 1: 1 -> 32
        self.conv1a = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1a = nn.BatchNorm2d(32)
        self.conv1b = nn.Conv2d(32, 32, 3, padding=1)
        self.bn1b = nn.BatchNorm2d(32)
        self.shortcut1 = nn.Conv2d(1, 32, 1)  # Match channels
        self.pool1 = nn.MaxPool2d(2)  # 28 -> 14

        # Block 2: 32 -> 64
        self.conv2a = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2a = nn.BatchNorm2d(64)
        self.conv2b = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2b = nn.BatchNorm2d(64)
        self.shortcut2 = nn.Conv2d(32, 64, 1)
        self.pool2 = nn.MaxPool2d(2)  # 14 -> 7

        # Block 3: 64 -> 128
        self.conv3a = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3a = nn.BatchNorm2d(128)
        self.conv3b = nn.Conv2d(128, 128, 3, padding=1)
        self.bn3b = nn.BatchNorm2d(128)
        self.shortcut3 = nn.Conv2d(64, 128, 1)
        self.pool3 = nn.AdaptiveAvgPool2d(1)  # 7x7 -> 1x1

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, 10),
        )

    def _block(self, x, conv_a, bn_a, conv_b, bn_b, shortcut, pool):
        identity = shortcut(x) if self.use_residual else None
        out = torch.relu(bn_a(conv_a(x)))
        out = bn_b(conv_b(out))
        if self.use_residual:
            out = out + identity
        out = torch.relu(out)
        return pool(out)

    def forward(self, x):
        x = self._block(x, self.conv1a, self.bn1a, self.conv1b, self.bn1b,
                         self.shortcut1, self.pool1)
        x = self._block(x, self.conv2a, self.bn2a, self.conv2b, self.bn2b,
                         self.shortcut2, self.pool2)
        x = self._block(x, self.conv3a, self.bn3a, self.conv3b, self.bn3b,
                         self.shortcut3, self.pool3)
        return self.classifier(x)


def build_model(config: dict) -> nn.Module:
    """Factory function: build model from config."""
    model_type: str = config["model"]["type"]
    dropout: float = config["training"].get("dropout", 0.3)

    if model_type == "mlp":
        return MLP(dropout=dropout)
    elif model_type == "simple_cnn":
        return SimpleCNN(dropout=dropout)
    elif model_type == "deeper_cnn":
        use_res = config["model"].get("use_residual", True)
        return DeeperCNN(dropout=dropout, use_residual=use_res)
    elif model_type == "xgboost":
        return None  # Handled separately
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def init_weights(model: nn.Module, method: str = "he"):
    """Apply weight initialization."""
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            if method == "he":
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif method == "xavier":
                nn.init.xavier_normal_(m.weight)
            # "default" = PyTorch default, do nothing
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def build_optimizer(model: nn.Module, config: dict):
    """Build optimizer from config."""
    opt_name = config["training"]["optimizer"]
    lr: float = config["training"]["lr"]
    wd: float = config["training"].get("weight_decay", 0.01)
    mom: float = config["training"].get("momentum", 0.9)

    if opt_name == "sgd":
        return optim.SGD(model.parameters(), lr=lr)
    elif opt_name == "sgd_momentum":
        return optim.SGD(model.parameters(), lr=lr, momentum=mom)
    elif opt_name == "adam":
        return optim.Adam(model.parameters(), lr=lr)
    elif opt_name == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")


# ============================================================
# 5. METRICS & FLOPS
# ============================================================

def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_flops(model: nn.Module, input_shape=(1, 1, 28, 28), device="cpu"):
    """
    Compute FLOPs using thop library.
    Returns FLOPs as an integer, or -1 if thop is not installed.
    """
    try:
        from thop import profile
        dummy = torch.randn(*input_shape).to(device)
        model_copy = model.to(device)
        model_copy.eval()
        flops, _ = profile(model_copy, inputs=(dummy,), verbose=False)
        return int(flops)
    except ImportError:
        print("[WARN] thop not installed. Skipping FLOPs. Install: pip install thop")
        return -1


def get_gpu_memory_mb() -> float:
    """Get peak GPU memory allocated in MB. Returns 0 if no GPU."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return 0.0


# ============================================================
# 6. TRAINING LOOP (PyTorch models)
# ============================================================

class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience: int = 5, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss: float):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch. Returns average loss and accuracy."""
    model.train()
    total_loss: float = 0.0
    correct: int = 0
    total: int = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    """Evaluate model. Returns loss, accuracy, all predictions, all labels."""
    model.eval()
    total_loss: float = 0.0
    correct: int = 0
    total: int = 0
    all_preds: list = []
    all_labels: list = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())

    return total_loss / total, correct / total, np.array(all_preds), np.array(all_labels)


def train_pytorch_model(model, config, train_loader, val_loader, device):
    """
    Full training loop with early stopping, per-epoch logging.
    Returns a dict of final metrics.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, config)
    epochs = config["training"]["epochs"]
    es_cfg = config["training"].get("early_stopping", {})
    early_stopper = EarlyStopping(
        patience=es_cfg.get("patience", 5),
        min_delta=es_cfg.get("min_delta", 0.001),
    )

    # Per-epoch curves (for plotting later)
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
    }

    # Reset GPU memory tracking
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    start_time = time.time()

    best_val_loss = float("inf")
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch

        print(
            f"  Epoch {epoch:3d}/{epochs} | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}"
        )

        early_stopper.step(val_loss)
        if early_stopper.should_stop:
            print(f"  [EARLY STOP] No improvement for {es_cfg.get('patience', 5)} epochs.")
            break

    elapsed = time.time() - start_time

    # Final evaluation on val set for per-class metrics
    _, val_acc_final, preds, labels = evaluate(
        model, val_loader, criterion, device
    )

    return {
        "accuracy": val_acc_final,
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_per_class": f1_score(labels, preds, average=None).tolist(),
        "confusion_matrix": confusion_matrix(labels, preds).tolist(),
        "train_time_sec": round(elapsed, 2),
        "epochs_run": epoch,
        "best_epoch": best_epoch,
        "gpu_mem_mb": round(get_gpu_memory_mb(), 1),
        "history": history,
    }


# ============================================================
# 7. XGBOOST TRAINING
# ============================================================

def train_xgboost(config, train_loader, val_loader):
    """Train XGBoost on flattened pixel vectors."""
    import xgboost as xgb

    # Extract numpy arrays from DataLoaders
    def loader_to_numpy(loader):
        X_list, y_list = [], []
        for inputs, targets in loader:
            X_list.append(inputs.view(inputs.size(0), -1).numpy())
            y_list.append(targets.numpy())
        return np.vstack(X_list), np.concatenate(y_list)

    X_train, y_train = loader_to_numpy(train_loader)
    X_val, y_val = loader_to_numpy(val_loader)

    # XGBoost params from config or defaults
    xgb_params = config["model"].get("xgb_params", {})
    params = {
        "max_depth": xgb_params.get("max_depth", 6),
        "n_estimators": xgb_params.get("n_estimators", 100),
        "learning_rate": xgb_params.get("learning_rate", 0.1),
        "objective": "multi:softmax",
        "num_class": 10,
        "n_jobs": -1,
        "random_state": config["experiment"]["seed"],
    }

    print(f"  XGBoost params: {params}")

    start_time: float = time.time()
    clf = xgb.XGBClassifier(**params)
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    elapsed = time.time() - start_time

    preds = clf.predict(X_val)

    return {
        "accuracy": accuracy_score(y_val, preds),
        "f1_macro": f1_score(y_val, preds, average="macro"),
        "f1_per_class": f1_score(y_val, preds, average=None).tolist(),
        "confusion_matrix": confusion_matrix(y_val, preds).tolist(),
        "train_time_sec": round(elapsed, 2),
        "epochs_run": params["n_estimators"],
        "best_epoch": -1,
        "gpu_mem_mb": 0.0,
        "history": {},
    }


# ============================================================
# 8. CSV LOGGING
# ============================================================

CSV_COLUMNS = [
    "timestamp",
    "experiment_name",
    "model_type",
    "optimizer",
    "lr",
    "batch_size",
    "dropout",
    "augmentation",
    "initialization",
    "accuracy",
    "f1_macro",
    "f1_per_class",
    "train_time_sec",
    "epochs_run",
    "best_epoch",
    "num_params",
    "flops",
    "gpu_mem_mb",
    "device",
    "seed",
    "config_file",
    "slurm_job_id",
]


def log_result(config: dict, metrics: dict, config_path: str,
               num_params: int, flops: int, device_str: str):
    """Append one row to the shared CSV results file."""
    csv_path = config["logging"].get("results_csv", "results/all_experiments.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    file_exists = os.path.isfile(csv_path)

    row = {
        "timestamp": datetime.now().isoformat(),
        "experiment_name": config["experiment"]["name"],
        "model_type": config["model"]["type"],
        "optimizer": config["training"].get("optimizer", "n/a"),
        "lr": config["training"].get("lr", "n/a"),
        "batch_size": config["training"]["batch_size"],
        "dropout": config["training"].get("dropout", "n/a"),
        "augmentation": config["data"].get("augmentation", "none"),
        "initialization": config["training"].get("initialization", "default"),
        "accuracy": round(metrics["accuracy"], 5),
        "f1_macro": round(metrics["f1_macro"], 5),
        "f1_per_class": json.dumps([round(f, 4) for f in metrics["f1_per_class"]]),
        "train_time_sec": metrics["train_time_sec"],
        "epochs_run": metrics["epochs_run"],
        "best_epoch": metrics["best_epoch"],
        "num_params": num_params,
        "flops": flops,
        "gpu_mem_mb": metrics["gpu_mem_mb"],
        "device": device_str,
        "seed": config["experiment"]["seed"],
        "config_file": os.path.basename(config_path),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "local"),
    }

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"[LOG] Result appended to {csv_path}")


def save_curves(config: dict, history: dict):
    """Save per-epoch train/val curves as JSON for later plotting."""
    if not config["logging"].get("save_curves", True):
        return
    curves_dir = config["logging"].get("curves_dir", "results/curves")
    os.makedirs(curves_dir, exist_ok=True)
    name: str = config["experiment"]["name"]
    path: str = os.path.join(curves_dir, f"{name}.json")
    with open(path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[LOG] Curves saved to {path}")


# ============================================================
# 9. MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="G12 Fashion-MNIST Training Harness")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    # Load config
    config: dict = load_config(args.config)
    exp_name: str = config["experiment"]["name"]
    model_type: str = config["model"]["type"]

    print("=" * 60)
    print(f"Experiment: {exp_name}")
    print(f"Model:      {model_type}")
    print(f"Config:     {args.config}")
    print("=" * 60)

    # Reproducibility
    set_seed(config["experiment"]["seed"])

    # Device
    if config["hardware"].get("device", "auto") == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config["hardware"]["device"])
    print(f"[DEVICE] {device}")
    if device.type == "cuda":
        print(f"[DEVICE] GPU: {torch.cuda.get_device_name(0)}")

    # Load data
    train_loader, val_loader, test_loader, _, _ = load_data(config)

    # Build and train model
    if model_type == "xgboost":
        num_params: int = -1
        flops: int = -1
        metrics: dict = train_xgboost(config, train_loader, val_loader)
    else:
        model: nn.Module = build_model(config)
        init_method: str = config["training"].get("initialization", "he")
        if init_method != "default":
            init_weights(model, init_method)
        model = model.to(device)

        num_params = count_parameters(model)
        print(f"[MODEL] Parameters: {num_params:,}")

        # FLOPs
        if config["hardware"].get("log_flops", True):
            flops = compute_flops(model, device=device)
            if flops > 0:
                print(f"[MODEL] FLOPs: {flops:,.0f}")
        else:
            flops = -1

        # Train
        metrics = train_pytorch_model(model, config, train_loader, val_loader, device)

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Accuracy:      {metrics['accuracy']:.4f}")
    print(f"  F1 (macro):    {metrics['f1_macro']:.4f}")
    print(f"  Train time:    {metrics['train_time_sec']:.1f}s")
    print(f"  Epochs run:    {metrics['epochs_run']}")
    print(f"  GPU memory:    {metrics['gpu_mem_mb']:.1f} MB")
    print()
    print("  Per-class F1:")
    for i, name in enumerate(CLASS_NAMES):
        f1_val = metrics["f1_per_class"][i]
        flag = " *** HARD CLASS" if f1_val < 0.85 else ""
        print(f"    {name:15s}: {f1_val:.4f}{flag}")

    # Log to CSV
    device_str = str(device)
    if device.type == "cuda":
        device_str = torch.cuda.get_device_name(0)
    log_result(config, metrics, args.config, num_params, flops, device_str)

    # Save curves
    if metrics.get("history"):
        save_curves(config, metrics["history"])

    print("\n[DONE]")


if __name__ == "__main__":
    main()
