"""
Model evaluation module for the AI-Powered Counterfeit Medicine Detection CNN.

Responsibilities:
    - Load the saved best model (models/best_model.keras)
    - Evaluate exclusively on the untouched Test split
    - Calculate Test Accuracy, Precision, Recall, F1-score (with Fake=1 as positive class),
      per-class metrics, ROC-AUC, and Confusion Matrix using scikit-learn
    - Generate and save:
        * outputs/confusion_matrix.png (Confusion matrix heatmap)
        * outputs/classification_report.txt (Detailed scikit-learn classification report)
        * outputs/test_metrics.json (Structured JSON of all evaluation metrics)
        * outputs/training_validation_accuracy.png & outputs/training_validation_loss.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tensorflow import keras

from src.data_loader import (
    CLASS_TO_INDEX,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DATASET_DIR,
    DEFAULT_SEED,
    OUTPUTS_DIR,
    create_tf_datasets,
)
from src.train import (
    ACCURACY_PLOT_PATH,
    BEST_MODEL_PATH,
    LOSS_PLOT_PATH,
    TRAINING_HISTORY_PATH,
    plot_training_curves,
)

CONFUSION_MATRIX_PLOT_PATH = OUTPUTS_DIR / "confusion_matrix.png"
CLASSIFICATION_REPORT_PATH = OUTPUTS_DIR / "classification_report.txt"
TEST_METRICS_JSON_PATH = OUTPUTS_DIR / "test_metrics.json"


def plot_confusion_matrix_heatmap(
    cm: np.ndarray,
    output_path: Path = CONFUSION_MATRIX_PLOT_PATH,
) -> None:
    """
    Generate and save a clear, annotated Confusion Matrix heatmap using Matplotlib.
    Rows: True labels [Real (0), Fake (1)]
    Cols: Predicted labels [Real (0), Fake (1)]
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    class_labels = ["Real (0)\n[Genuine]", "Fake (1)\n[Counterfeit]"]

    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=150)
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Number of Test Images", rotation=-90, va="bottom", fontsize=10)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_labels,
        yticklabels=class_labels,
        title="Test Set Confusion Matrix (Positive Class = Fake [1])",
        ylabel="Actual Class",
        xlabel="Predicted Class",
    )
    ax.title.set_fontsize(12)
    ax.title.set_fontweight("bold")
    ax.xaxis.label.set_fontsize(11)
    ax.yaxis.label.set_fontsize(11)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    row_sums = cm.sum(axis=1, keepdims=True)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            pct = (100.0 * count / row_sums[i, 0]) if row_sums[i, 0] > 0 else 0.0
            cell_label = f"{count}\n({pct:.1f}%)"
            ax.text(
                j,
                i,
                cell_label,
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                color="white" if count > thresh else "black",
            )

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def evaluate_model(
    model_path: Path = BEST_MODEL_PATH,
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    batch_size: int = DEFAULT_BATCH_SIZE,
    split_mode: str = "deduplicated",
    seed: int = DEFAULT_SEED,
) -> Dict[str, Any]:
    """
    Evaluate the saved best CNN model on the untouched Test split and save all artifacts.
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at '{model_path}'. "
            "Please run `python -m src.train` first to train and save the model."
        )

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # Ensure training curves exist if training_history.json is present
    if TRAINING_HISTORY_PATH.exists():
        with open(TRAINING_HISTORY_PATH, "r", encoding="utf-8") as f:
            hist_data = json.load(f)
        if "history" in hist_data:
            plot_training_curves(hist_data["history"], ACCURACY_PLOT_PATH, LOSS_PLOT_PATH)

    print(f"\nLoading trained model from: {model_path}")
    model = keras.models.load_model(str(model_path))

    print("Preparing untouched Test dataset...")
    _, _, test_ds, _, audit_report, splits_data = create_tf_datasets(
        dataset_dir=dataset_dir,
        batch_size=batch_size,
        split_mode=split_mode,
        seed=seed,
    )

    test_paths, test_labels = splits_data["Test"]
    y_true = np.asarray(test_labels, dtype=np.int32)

    # Evaluate loss via Keras and obtain sigmoid scores
    keras_eval = model.evaluate(test_ds, verbose=0, return_dict=True)
    test_loss = float(keras_eval.get("loss", 0.0))

    y_scores = model.predict(test_ds, verbose=0).reshape(-1)
    y_pred = (y_scores >= 0.5).astype(np.int32)

    # Calculate scikit-learn metrics with Fake (1) explicitly set as the positive class
    pos_label = CLASS_TO_INDEX["Fake"]  # 1
    neg_label = CLASS_TO_INDEX["Real"]  # 0

    test_accuracy = float(accuracy_score(y_true, y_pred))
    fake_precision = float(precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0))
    fake_recall = float(recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0))
    fake_f1 = float(f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0))

    real_precision = float(precision_score(y_true, y_pred, pos_label=neg_label, zero_division=0))
    real_recall = float(recall_score(y_true, y_pred, pos_label=neg_label, zero_division=0))
    real_f1 = float(f1_score(y_true, y_pred, pos_label=neg_label, zero_division=0))

    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    roc_auc = float(roc_auc_score(y_true, y_scores))

    cm = confusion_matrix(y_true, y_pred, labels=[neg_label, pos_label])
    tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1]))

    report_str = classification_report(
        y_true,
        y_pred,
        labels=[neg_label, pos_label],
        target_names=["Real (0 - Genuine)", "Fake (1 - Counterfeit)"],
        digits=4,
        zero_division=0,
    )

    # Save classification report text file
    report_header = (
        "AI-Powered Counterfeit Medicine Detection — Test Set Classification Report\n"
        "==========================================================================\n"
        f"Model Evaluated   : {model_path}\n"
        f"Split Mode        : {split_mode}\n"
        f"Test Samples      : {len(y_true)} (Real [0]: {int(np.sum(y_true == 0))}, Fake [1]: {int(np.sum(y_true == 1))})\n"
        f"Positive Class    : Fake (1)\n"
        f"Decision Threshold: Sigmoid Score >= 0.5 -> Fake (1), < 0.5 -> Real (0)\n\n"
        f"Overall Test Accuracy          : {test_accuracy * 100:.2f}% ({test_accuracy:.4f})\n"
        f"Test Loss (Binary CrossEntropy): {test_loss:.4f}\n"
        f"Precision (Positive = Fake [1]): {fake_precision:.4f}\n"
        f"Recall    (Positive = Fake [1]): {fake_recall:.4f}\n"
        f"F1-Score  (Positive = Fake [1]): {fake_f1:.4f}\n"
        f"ROC-AUC Score                  : {roc_auc:.4f}\n\n"
        "Confusion Matrix (Rows = Actual [Real=0, Fake=1], Cols = Predicted [Real=0, Fake=1]):\n"
        f"  True Negatives  (Actual Real -> Predicted Real) : {tn}\n"
        f"  False Positives (Actual Real -> Predicted Fake) : {fp}\n"
        f"  False Negatives (Actual Fake -> Predicted Real) : {fn}\n"
        f"  True Positives  (Actual Fake -> Predicted Fake) : {tp}\n\n"
        "Per-Class Classification Report (scikit-learn):\n"
        "--------------------------------------------------------------------------\n"
    )
    CLASSIFICATION_REPORT_PATH.write_text(report_header + report_str + "\n", encoding="utf-8")

    # Save confusion matrix heatmap
    plot_confusion_matrix_heatmap(cm, CONFUSION_MATRIX_PLOT_PATH)

    # Build and save test_metrics.json
    metrics_payload: Dict[str, Any] = {
        "model_path": str(model_path),
        "split_mode": split_mode,
        "positive_class": "Fake (1)",
        "negative_class": "Real (0)",
        "test_set_counts": {
            "total": int(len(y_true)),
            "Real_0": int(np.sum(y_true == 0)),
            "Fake_1": int(np.sum(y_true == 1)),
        },
        "test_loss": round(test_loss, 6),
        "test_accuracy": round(test_accuracy, 6),
        "precision": round(fake_precision, 6),
        "recall": round(fake_recall, 6),
        "f1_score": round(fake_f1, 6),
        "roc_auc": round(roc_auc, 6),
        "macro_f1": round(macro_f1, 6),
        "weighted_f1": round(weighted_f1, 6),
        "per_class_metrics": {
            "Real_0": {
                "precision": round(real_precision, 6),
                "recall": round(real_recall, 6),
                "f1_score": round(real_f1, 6),
                "support": int(np.sum(y_true == 0)),
            },
            "Fake_1": {
                "precision": round(fake_precision, 6),
                "recall": round(fake_recall, 6),
                "f1_score": round(fake_f1, 6),
                "support": int(np.sum(y_true == 1)),
            },
        },
        "confusion_matrix": {
            "labels": ["Real (0)", "Fake (1)"],
            "matrix": cm.tolist(),
            "true_negatives_real": tn,
            "false_positives_fake": fp,
            "false_negatives_real": fn,
            "true_positives_fake": tp,
        },
    }

    with open(TEST_METRICS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    print("\n" + "=" * 72)
    print("FINAL TEST SET EVALUATION RESULTS (Positive Class = Fake [1])")
    print("=" * 72)
    print(f"Test Samples      : {len(y_true)} (Real: {np.sum(y_true == 0)}, Fake: {np.sum(y_true == 1)})")
    print(f"Test Loss         : {test_loss:.4f}")
    print(f"Test Accuracy     : {test_accuracy * 100:.2f}% ({test_accuracy:.4f})")
    print(f"Precision (Fake)  : {fake_precision:.4f}")
    print(f"Recall (Fake)     : {fake_recall:.4f}")
    print(f"F1-Score (Fake)   : {fake_f1:.4f}")
    print(f"ROC-AUC Score     : {roc_auc:.4f}")
    print("-" * 72)
    print("Confusion Matrix:")
    print(f"                  Pred Real (0)    Pred Fake (1)")
    print(f"  Actual Real (0)     {tn:>5}            {fp:>5}")
    print(f"  Actual Fake (1)     {fn:>5}            {tp:>5}")
    print("-" * 72)
    print("Per-Class Report:")
    print(report_str)
    print("-" * 72)
    print(f"Saved Confusion Matrix Plot : {CONFUSION_MATRIX_PLOT_PATH}")
    print(f"Saved Classification Report : {CLASSIFICATION_REPORT_PATH}")
    print(f"Saved Test Metrics JSON     : {TEST_METRICS_JSON_PATH}")
    print("=" * 72)

    return metrics_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the trained Counterfeit Medicine Detection CNN on the Test set."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=BEST_MODEL_PATH,
        help="Path to the saved Keras model file.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Path to the dataset directory.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for evaluation (default: 32).",
    )
    parser.add_argument(
        "--split-mode",
        type=str,
        default="deduplicated",
        choices=["deduplicated", "folder"],
        help="Split mode used to evaluate test split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed matching training split (default: 42).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_model(
        model_path=args.model_path,
        dataset_dir=args.dataset_dir,
        batch_size=args.batch_size,
        split_mode=args.split_mode,
        seed=args.seed,
    )
