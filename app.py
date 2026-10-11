"""
Streamlit Web Interface for AI-Powered Counterfeit Medicine Detection Using CNN.

Launch from project root:
    streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, UnidentifiedImageError
import streamlit as st

from src.data_loader import CLASS_TO_INDEX, IMAGE_SIZE, OUTPUTS_DIR
from src.evaluate import (
    CLASSIFICATION_REPORT_PATH,
    CONFUSION_MATRIX_PLOT_PATH,
    TEST_METRICS_JSON_PATH,
)
from src.predict import load_trained_model, predict_image
from src.train import (
    ACCURACY_PLOT_PATH,
    BEST_MODEL_PATH,
    LOSS_PLOT_PATH,
)

st.set_page_config(
    page_title="AI-Powered Counterfeit Medicine Detection Using CNN",
    page_icon="💊",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading trained CNN model...")
def get_cached_model(model_path_str: str):
    """Load and cache the trained Keras model so it is never reloaded or retrained per upload."""
    return load_trained_model(Path(model_path_str))


def render_disclaimer() -> None:
    """Display the mandatory academic and safety disclaimer."""
    st.warning(
        "**Important Disclaimer:** This application is an academic Deep Learning demonstration "
        "that analyzes **visual medicine packaging patterns only** (such as color, typography, "
        "layout, and surface texture). It **cannot chemically verify** a medicine's active "
        "pharmaceutical ingredients or guarantee physical authenticity. Always consult a "
        "licensed pharmacist, regulatory authority, or laboratory assay for medical verification."
    )


def main() -> None:
    st.title("AI-Powered Counterfeit Medicine Detection Using CNN")
    st.markdown(
        "Upload an image of a medicine package (**JPG, JPEG, or PNG**) to classify it as "
        "**Real** (genuine packaging) or **Fake** (counterfeit or manipulated packaging) "
        "using our custom Convolutional Neural Network."
    )

    render_disclaimer()

    # Sidebar with model & preprocessing details for faculty demonstration
    with st.sidebar:
        st.header("Model Configuration")
        st.markdown(
            f"""
            - **Architecture:** Custom 11-Layer CNN
            - **Input Resolution:** `{IMAGE_SIZE[0]} × {IMAGE_SIZE[1]} × 3` (RGB)
            - **Pixel Normalization:** `[0.0, 1.0]`
            - **Loss Function:** `binary_crossentropy`
            - **Optimizer:** `Adam`
            - **Label Mapping:**
              - `Real` $\\rightarrow$ `{CLASS_TO_INDEX['Real']}` (Negative class, score $< 0.5$)
              - `Fake` $\\rightarrow$ `{CLASS_TO_INDEX['Fake']}` (Positive class, score $\\ge 0.5$)
            - **Model Path:** `models/best_model.keras`
            """
        )

    # Check if the trained model exists before allowing predictions
    if not BEST_MODEL_PATH.exists():
        st.error("Trained CNN model not found at `models/best_model.keras`.")
        st.info(
            "### How to Train the Model First\n"
            "Please open a terminal in the project root directory, activate your virtual "
            "environment, and run the training and evaluation scripts:\n"
            "```powershell\n"
            ".\\.venv\\Scripts\\Activate.ps1\n"
            "python -m src.train\n"
            "python -m src.evaluate\n"
            "```\n"
            "Once `models/best_model.keras` is generated, refresh this page."
        )
        return

    try:
        model = get_cached_model(str(BEST_MODEL_PATH))
    except Exception as exc:
        st.error(f"Failed to load the saved model from `{BEST_MODEL_PATH}`: {exc}")
        return

    tab_predict, tab_metrics = st.tabs(
        ["🔍 Medicine Image Prediction", "📊 Model Evaluation & Training Charts"]
    )

    with tab_predict:
        uploaded_file = st.file_uploader(
            "Upload a Medicine Package Image (JPG, JPEG, or PNG)",
            type=["jpg", "jpeg", "png"],
            help="Select a clear RGB image of a medicine box, blister pack, or bottle label.",
        )

        if uploaded_file is not None:
            try:
                raw_image = Image.open(uploaded_file)
                raw_image.load()
            except (UnidentifiedImageError, OSError, ValueError) as exc:
                st.error(
                    f"Invalid or corrupted image file uploaded. Please upload a valid JPG, JPEG, or PNG image. Details: {exc}"
                )
                return

            col_img, col_res = st.columns([1, 1.2], gap="large")

            with col_img:
                st.subheader("Uploaded Medicine Image")
                st.image(
                    raw_image,
                    caption=f"Uploaded File: {uploaded_file.name} (Original size: {raw_image.size[0]}×{raw_image.size[1]}, Mode: {raw_image.mode})",
                    use_container_width=True,
                )

            with col_res:
                st.subheader("CNN Prediction Result")
                try:
                    result = predict_image(raw_image, model=model, model_path=BEST_MODEL_PATH)
                except ValueError as exc:
                    st.error(str(exc))
                    return

                if result.predicted_label == "Fake":
                    st.error(
                        f"### ⚠️ Predicted Class: **FAKE** (Counterfeit / Manipulated Packaging)\n"
                        f"**Predicted Class Index:** `{result.predicted_index}` (Positive Class)"
                    )
                else:
                    st.success(
                        f"### ✅ Predicted Class: **REAL** (Genuine Medicine Packaging)\n"
                        f"**Predicted Class Index:** `{result.predicted_index}` (Negative Class)"
                    )

                st.metric(
                    label="Raw Sigmoid Output Score (Range: 0.0000 to 1.0000)",
                    value=f"{result.sigmoid_score:.4f}",
                    delta=f"Threshold: {result.threshold:.2f} (>= 0.50 is Fake, < 0.50 is Real)",
                    delta_color="off",
                )

                st.progress(
                    min(max(float(result.sigmoid_score), 0.0), 1.0),
                    text=f"Sigmoid Activation Scale: 0.0 (Real) ⟷ 1.0 (Fake) | Current Score: {result.sigmoid_score:.4f}",
                )

                st.markdown("#### How to Interpret This Sigmoid Score")
                st.info(
                    f"{result.interpretation}\n\n"
                    "- **0.0000 – 0.4999:** Classified as **Real (0)** — visual features align more closely with genuine packaging in the training set.\n"
                    "- **0.5000 – 1.0000:** Classified as **Fake (1)** — visual features align more closely with counterfeit/manipulated packaging in the training set.\n"
                    "- **Note on Calibration:** A raw sigmoid output reflects the neural network's internal decision boundary activation and should **not** be interpreted as a calibrated statistical probability."
                )

    with tab_metrics:
        st.subheader("Test Set Performance Summary")
        if TEST_METRICS_JSON_PATH.exists():
            with open(TEST_METRICS_JSON_PATH, "r", encoding="utf-8") as f:
                metrics = json.load(f)

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Test Accuracy", f"{metrics['test_accuracy'] * 100:.2f}%")
            m2.metric("Precision (Fake=1)", f"{metrics['precision']:.4f}")
            m3.metric("Recall (Fake=1)", f"{metrics['recall']:.4f}")
            m4.metric("F1-Score (Fake=1)", f"{metrics['f1_score']:.4f}")
            m5.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")

            c1, c2, c3 = st.columns(3)
            with c1:
                if CONFUSION_MATRIX_PLOT_PATH.exists():
                    st.image(
                        str(CONFUSION_MATRIX_PLOT_PATH),
                        caption="Test Set Confusion Matrix",
                        use_container_width=True,
                    )
            with c2:
                if ACCURACY_PLOT_PATH.exists():
                    st.image(
                        str(ACCURACY_PLOT_PATH),
                        caption="Training vs. Validation Accuracy",
                        use_container_width=True,
                    )
            with c3:
                if LOSS_PLOT_PATH.exists():
                    st.image(
                        str(LOSS_PLOT_PATH),
                        caption="Training vs. Validation Loss",
                        use_container_width=True,
                    )

            if CLASSIFICATION_REPORT_PATH.exists():
                with st.expander("View Full scikit-learn Classification Report"):
                    st.code(CLASSIFICATION_REPORT_PATH.read_text(encoding="utf-8"), language="text")
        else:
            st.info(
                "Evaluation metrics not found yet. Run `python -m src.evaluate` to generate test metrics and plots."
            )


if __name__ == "__main__":
    main()
