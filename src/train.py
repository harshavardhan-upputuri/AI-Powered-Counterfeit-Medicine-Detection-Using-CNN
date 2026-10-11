"""
Training pipeline for the AI-Powered Counterfeit Medicine Detection CNN.

Responsibilities:
    - Load and verify the Train and Val splits via src.data_loader
    - Build and compile the custom 11-layer CNN via src.model
    - Save the model architecture summary to outputs/model_summary.txt
    - Configure EarlyStopping, ModelCheckpoint, and ReduceLROnPlateau callbacks
    - Train the model with balanced class weights to address class imbalance
    - Save the best trained model to models/best_model.keras
    - Save the training history to outputs/training_history.json
    - Generate and save Training vs Validation Accuracy and Loss plots
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

from src.data_loader import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DATASET_DIR,
    DEFAULT_SEED,
    OUTPUTS_DIR,
    PROJECT_ROOT,
    create_tf_datasets,
    print_audit_report,
    set_global_seed,
)
from src.model import build_cnn_model, save_model_summary

MODELS_DIR = PROJECT_ROOT / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.keras"
TRAINING_HISTORY_PATH = OUTPUTS_DIR / "training_history.json"
ACCURACY_PLOT_PATH = OUTPUTS_DIR / "training_validation_accuracy.png"
LOSS_PLOT_PATH = OUTPUTS_DIR / "training_validation_loss.png"


def plot_training_curves(
    history_dict: Dict[str, list],
    acc_plot_path: Path = ACCURACY_PLOT_PATH,
    loss_plot_path: Path = LOSS_PLOT_PATH,
) -> None:
    """
    Generate and save high-resolution plots for:
        1. Training vs Validation Accuracy
        2. Training vs Validation Loss
    """
    acc_plot_path.parent.mkdir(parents=True, exist_ok=True)
    epochs_range = range(1, len(history_dict["loss"]) + 1)
    best_epoch = int(np.argmin(history_dict["val_loss"])) + 1

    # 1. Training vs Validation Accuracy Plot
    plt.figure(figsize=(8, 5), dpi=150)
    plt.plot(epochs_range, history_dict["accuracy"], "b-o", label="Training Accuracy", linewidth=2, markersize=4)
    plt.plot(epochs_range, history_dict["val_accuracy"], "r-s", label="Validation Accuracy", linewidth=2, markersize=4)
    plt.axvline(x=best_epoch, color="green", linestyle="--", alpha=0.7, label=f"Best Val Loss Epoch ({best_epoch})")
    plt.title("Training vs. Validation Accuracy (Custom CNN)", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Accuracy", fontsize=11)
    plt.ylim(0.0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(acc_plot_path)
    plt.close()

    # 2. Training vs Validation Loss Plot
    plt.figure(figsize=(8, 5), dpi=150)
    plt.plot(epochs_range, history_dict["loss"], "b-o", label="Training Loss", linewidth=2, markersize=4)
    plt.plot(epochs_range, history_dict["val_loss"], "r-s", label="Validation Loss", linewidth=2, markersize=4)
    plt.axvline(x=best_epoch, color="green", linestyle="--", alpha=0.7, label=f"Best Val Loss Epoch ({best_epoch})")
    plt.title("Training vs. Validation Loss (Custom CNN)", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Binary Cross-Entropy Loss", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(loss_plot_path)
    plt.close()


def train_model(
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    epochs: int = 30,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = 1e-3,
    split_mode: str = "deduplicated",
    seed: int = DEFAULT_SEED,
) -> Dict[str, Any]:
    """
    Execute the end-to-end training workflow and save the best model & training history.
    """
    set_global_seed(seed)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[Step 1/4] Loading and verifying dataset splits...")
    train_ds, val_ds, _, class_weights, audit_report, _ = create_tf_datasets(
        dataset_dir=dataset_dir,
        batch_size=batch_size,
        split_mode=split_mode,
        seed=seed,
    )
    print_audit_report(audit_report)
    print(f"Computed Balanced Class Weights: Real (0) = {class_weights[0]:.4f}, Fake (1) = {class_weights[1]:.4f}")

    print("\n[Step 2/4] Building and compiling custom CNN architecture...")
    model = build_cnn_model(learning_rate=learning_rate, seed=seed)
    summary_file = save_model_summary(model)
    print(f"Architecture summary saved to: {summary_file}")

    print("\n[Step 3/4] Configuring training callbacks...")
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(BEST_MODEL_PATH),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    print(f"\n[Step 4/4] Starting training for up to {epochs} epochs (batch_size={batch_size})...")
    start_time = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=2,
    )
    elapsed_sec = time.time() - start_time

    # Ensure best_model.keras is saved on disk
    if not BEST_MODEL_PATH.exists():
        model.save(str(BEST_MODEL_PATH))

    history_dict: Dict[str, Any] = {
        key: [float(v) for v in values] for key, values in history.history.items()
    }
    best_epoch_idx = int(np.argmin(history_dict["val_loss"]))
    best_summary = {
        "epochs_configured": epochs,
        "epochs_completed": len(history_dict["loss"]),
        "best_epoch": best_epoch_idx + 1,
        "best_val_loss": float(history_dict["val_loss"][best_epoch_idx]),
        "best_val_accuracy": float(history_dict["val_accuracy"][best_epoch_idx]),
        "best_train_loss": float(history_dict["loss"][best_epoch_idx]),
        "best_train_accuracy": float(history_dict["accuracy"][best_epoch_idx]),
        "training_time_seconds": round(elapsed_sec, 2),
        "batch_size": batch_size,
        "initial_learning_rate": learning_rate,
        "split_mode": split_mode,
        "class_weights": {str(k): v for k, v in class_weights.items()},
        "history": history_dict,
    }

    with open(TRAINING_HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(best_summary, f, indent=2)

    plot_training_curves(history_dict)

    print("\n" + "=" * 72)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 72)
    print(f"Epochs Completed         : {best_summary['epochs_completed']} / {epochs}")
    print(f"Best Validation Epoch    : {best_summary['best_epoch']}")
    print(f"Best Validation Loss     : {best_summary['best_val_loss']:.4f}")
    print(f"Best Validation Accuracy : {best_summary['best_val_accuracy'] * 100:.2f}%")
    print(f"Training Accuracy @ Best : {best_summary['best_train_accuracy'] * 100:.2f}%")
    print(f"Total Training Time      : {elapsed_sec:.1f} seconds")
    print(f"Saved Best Model         : {BEST_MODEL_PATH}")
    print(f"Saved Training History   : {TRAINING_HISTORY_PATH}")
    print(f"Saved Accuracy Plot      : {ACCURACY_PLOT_PATH}")
    print(f"Saved Loss Plot          : {LOSS_PLOT_PATH}")
    print("=" * 72)

    return best_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the custom CNN for Counterfeit Medicine Detection."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Path to the dataset directory.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Maximum number of training epochs (default: 30).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Training and validation batch size (default: 32).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Initial learning rate for Adam optimizer (default: 0.001).",
    )
    parser.add_argument(
        "--split-mode",
        type=str,
        default="deduplicated",
        choices=["deduplicated", "folder"],
        help="Split mode: 'deduplicated' (leakage-free 70/15/15) or 'folder' (raw directories).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for reproducibility (default: 42).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_model(
        dataset_dir=args.dataset_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        split_mode=args.split_mode,
        seed=args.seed,
    )
