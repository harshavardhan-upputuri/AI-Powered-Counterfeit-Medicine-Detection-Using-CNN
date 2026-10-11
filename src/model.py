"""
Custom Convolutional Neural Network (CNN) architecture for Counterfeit Medicine Detection.

Strictly implements the required 11-layer design from scratch using Keras:
    1. Input layer: 224 x 224 x 3
    2. Conv2D: 32 filters, 3 x 3 kernel, ReLU activation
    3. MaxPooling2D: 2 x 2
    4. Conv2D: 64 filters, 3 x 3 kernel, ReLU activation
    5. MaxPooling2D: 2 x 2
    6. Conv2D: 128 filters, 3 x 3 kernel, ReLU activation
    7. MaxPooling2D: 2 x 2
    8. GlobalAveragePooling2D
    9. Dense: 128 neurons, ReLU activation
   10. Dropout: 0.5
   11. Output: Dense(1, activation="sigmoid")

Output Interpretation:
    - Sigmoid output in [0.0, 1.0] represents the model's uncalibrated confidence score
      toward the positive class Fake (1).
    - Score >= 0.5 -> Predicted Class: Fake (1, counterfeit medicine packaging)
    - Score <  0.5 -> Predicted Class: Real (0, genuine medicine packaging)
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.data_loader import INPUT_SHAPE, OUTPUTS_DIR, DEFAULT_SEED


def build_cnn_model(
    input_shape: Tuple[int, int, int] = INPUT_SHAPE,
    learning_rate: float = 1e-3,
    dropout_rate: float = 0.5,
    seed: int = DEFAULT_SEED,
) -> keras.Model:
    """
    Build and compile the custom CNN architecture from scratch.

    Args:
        input_shape: Input image tensor shape, default (224, 224, 3).
        learning_rate: Initial learning rate for the Adam optimizer.
        dropout_rate: Dropout rate before the final output layer (default 0.5).
        seed: Random seed for weight initialization and dropout reproducibility.

    Returns:
        Compiled tf.keras.Model ready for training.
    """
    keras.utils.set_random_seed(seed)

    model = keras.Sequential(
        [
            # 1. Input layer: 224 x 224 x 3
            layers.Input(shape=input_shape, name="input_rgb_224x224x3"),

            # 2. Conv2D: 32 filters, 3 x 3 kernel, ReLU activation
            layers.Conv2D(
                filters=32,
                kernel_size=(3, 3),
                activation="relu",
                name="conv2d_32",
            ),
            # 3. MaxPooling2D: 2 x 2
            layers.MaxPooling2D(pool_size=(2, 2), name="maxpool_1"),

            # 4. Conv2D: 64 filters, 3 x 3 kernel, ReLU activation
            layers.Conv2D(
                filters=64,
                kernel_size=(3, 3),
                activation="relu",
                name="conv2d_64",
            ),
            # 5. MaxPooling2D: 2 x 2
            layers.MaxPooling2D(pool_size=(2, 2), name="maxpool_2"),

            # 6. Conv2D: 128 filters, 3 x 3 kernel, ReLU activation
            layers.Conv2D(
                filters=128,
                kernel_size=(3, 3),
                activation="relu",
                name="conv2d_128",
            ),
            # 7. MaxPooling2D: 2 x 2
            layers.MaxPooling2D(pool_size=(2, 2), name="maxpool_3"),

            # 8. GlobalAveragePooling2D
            layers.GlobalAveragePooling2D(name="global_avg_pool"),

            # 9. Dense: 128 neurons, ReLU activation
            layers.Dense(128, activation="relu", name="dense_128"),

            # 10. Dropout: 0.5
            layers.Dropout(rate=dropout_rate, seed=seed, name="dropout_0_5"),

            # 11. Output: Dense(1, activation="sigmoid")
            # Positive class (1) = Fake, Negative class (0) = Real
            layers.Dense(1, activation="sigmoid", name="output_sigmoid"),
        ],
        name="CounterfeitMedicineCNN",
    )

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy", threshold=0.5),
            keras.metrics.Precision(name="precision", thresholds=0.5),
            keras.metrics.Recall(name="recall", thresholds=0.5),
            keras.metrics.AUC(name="auc"),
        ],
    )

    return model


def save_model_summary(
    model: keras.Model,
    output_path: Union[str, Path] = OUTPUTS_DIR / "model_summary.txt",
) -> Path:
    """
    Print model.summary() to the console and save the full text summary to a file.
    """
    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    summary_lines = []
    model.summary(print_fn=lambda line: summary_lines.append(line))
    summary_text = "\n".join(summary_lines)

    print(summary_text)

    header = (
        "AI-Powered Counterfeit Medicine Detection — Custom CNN Architecture Summary\n"
        "===========================================================================\n"
        "Input Shape : (224, 224, 3) RGB normalized to [0.0, 1.0]\n"
        "Label Map   : Real -> 0 (Negative class, sigmoid < 0.5)\n"
        "              Fake -> 1 (Positive class, sigmoid >= 0.5)\n"
        "Loss        : binary_crossentropy\n"
        "Optimizer   : Adam\n\n"
    )
    target_path.write_text(header + summary_text + "\n", encoding="utf-8")
    return target_path


if __name__ == "__main__":
    cnn = build_cnn_model()
    saved_path = save_model_summary(cnn)
    print(f"\nSaved architecture summary to: {saved_path}")
