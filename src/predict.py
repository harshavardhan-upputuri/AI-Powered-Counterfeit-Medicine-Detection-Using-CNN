"""
Single-image prediction module for the AI-Powered Counterfeit Medicine Detection CNN.

Responsibilities:
    - Load models/best_model.keras safely with caching/error handling
    - Preprocess an uploaded or local image identically to training (RGB, 224x224, [0, 1])
    - Predict the sigmoid output score and classify as Real (< 0.5) or Fake (>= 0.5)
    - Provide a clear explanation of the uncalibrated sigmoid score
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image, UnidentifiedImageError
from tensorflow import keras

from src.data_loader import INDEX_TO_CLASS, preprocess_single_image
from src.train import BEST_MODEL_PATH


@dataclass
class PredictionResult:
    """Structured result for a single medicine packaging image prediction."""
    predicted_label: str
    predicted_index: int
    sigmoid_score: float
    threshold: float
    interpretation: str
    model_path: str


def load_trained_model(model_path: Union[str, Path] = BEST_MODEL_PATH) -> keras.Model:
    """Load the saved Keras model from disk with clear error reporting if missing."""
    resolved_path = Path(model_path)
    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Trained model file not found at '{resolved_path}'. "
            "Please train the model first by running: python -m src.train"
        )
    return keras.models.load_model(str(resolved_path))


def predict_image(
    image_source: Union[str, Path, Image.Image],
    model: Optional[keras.Model] = None,
    model_path: Union[str, Path] = BEST_MODEL_PATH,
    threshold: float = 0.5,
) -> PredictionResult:
    """
    Run inference on a single medicine package image.

    Args:
        image_source: File path or PIL.Image.Image object.
        model: Optional pre-loaded Keras model instance (avoids reloading on every request).
        model_path: Path to best_model.keras if model is not passed.
        threshold: Decision threshold on the sigmoid output (default 0.5).

    Returns:
        PredictionResult with class name, class index, raw sigmoid score, and explanation.
    """
    if model is None:
        model = load_trained_model(model_path)

    try:
        preprocessed = preprocess_single_image(image_source)
    except (UnidentifiedImageError, OSError, ValueError, TypeError) as exc:
        raise ValueError(f"Invalid or corrupted image file provided: {exc}") from exc

    batch_tensor = np.expand_dims(preprocessed, axis=0)  # Shape: (1, 224, 224, 3)
    raw_output = model.predict(batch_tensor, verbose=0)
    sigmoid_score = float(np.squeeze(raw_output))

    predicted_index = 1 if sigmoid_score >= threshold else 0
    predicted_label = INDEX_TO_CLASS[predicted_index]

    if predicted_index == 1:
        interpretation = (
            f"The output neuron's sigmoid activation score is {sigmoid_score:.4f}, which is "
            f"greater than or equal to the decision threshold of {threshold:.2f}. "
            "In our binary label mapping (Real = 0, Fake = 1), scores closer to 1.0 indicate "
            "stronger visual similarity to Fake (counterfeit/manipulated) packaging patterns "
            "learned during training. Note: This raw sigmoid score is an uncalibrated network "
            "activation, not a statistically calibrated probability."
        )
    else:
        interpretation = (
            f"The output neuron's sigmoid activation score is {sigmoid_score:.4f}, which is "
            f"below the decision threshold of {threshold:.2f}. "
            "In our binary label mapping (Real = 0, Fake = 1), scores closer to 0.0 indicate "
            "stronger visual similarity to Real (genuine) medicine packaging patterns learned "
            "during training. Note: This raw sigmoid score is an uncalibrated network "
            "activation, not a statistically calibrated probability."
        )

    return PredictionResult(
        predicted_label=predicted_label,
        predicted_index=predicted_index,
        sigmoid_score=sigmoid_score,
        threshold=threshold,
        interpretation=interpretation,
        model_path=str(model_path),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict whether a medicine package image is Real (0) or Fake (1)."
    )
    parser.add_argument(
        "image_path",
        type=Path,
        help="Path to the input JPG, JPEG, or PNG medicine image.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=BEST_MODEL_PATH,
        help="Path to the trained Keras model (default: models/best_model.keras).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = predict_image(args.image_path, model_path=args.model_path)
    print("=" * 68)
    print("MEDICINE PACKAGING PREDICTION RESULT")
    print("=" * 68)
    print(f"Input Image      : {args.image_path}")
    print(f"Predicted Class  : {result.predicted_label} (Class Index = {result.predicted_index})")
    print(f"Sigmoid Score    : {result.sigmoid_score:.4f} (Threshold = {result.threshold:.2f})")
    print(f"Interpretation   : {result.interpretation}")
    print("-" * 68)
    print(
        "DISCLAIMER: This tool analyzes visual packaging patterns only and cannot "
        "chemically verify a medicine's authenticity."
    )
    print("=" * 68)
