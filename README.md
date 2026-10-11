# AI-Powered Counterfeit Medicine Detection Using CNN

A complete end-to-end Deep Learning academic project that classifies medicine package images into **Real** (genuine packaging) or **Fake** (counterfeit or manipulated packaging) using a custom 11-layer Convolutional Neural Network (CNN) built from scratch with TensorFlow and Keras.

---

## 1. Project Overview

Counterfeit and substandard medicines pose a severe threat to public health and supply-chain integrity worldwide. Counterfeit medicine packages often exhibit subtle visual discrepancies compared to genuine pharmaceutical packaging—such as altered typography, misaligned logos, inconsistent color saturation, degraded print quality, or manipulated blister/box layouts.

This project implements a supervised computer vision pipeline using a custom Convolutional Neural Network (CNN) to automatically inspect RGB images of medicine packaging and classify them as **Real (`0`)** or **Fake (`1`)**, accompanied by an interactive **Streamlit** web application for live demonstration.

---

## 2. Problem Statement

Manual visual inspection of medicine packaging by distributors, pharmacists, and consumers is error-prone and difficult to scale. Given an RGB image of a medicine package of arbitrary resolution, the goal is to design, train, and evaluate a binary CNN classifier capable of distinguishing **Fake (counterfeit/manipulated)** medicine packaging from **Real (genuine)** packaging with high accuracy and recall, while strictly avoiding cross-split data leakage.

---

## 3. Objectives

1. **Audit and Verify the Local Dataset:** Inspect the local dataset directory, identify non-image files (`desktop.ini`), detect within-split and cross-split duplicate images via MD5 hashing, and construct a leakage-free **Train / Val / Test** split without modifying the original dataset on disk.
2. **Build a Custom 11-Layer CNN from Scratch:** Implement the specified architecture using the Keras API without relying on pretrained transfer-learning backbones (such as MobileNet, ResNet, or VGG).
3. **Standardize Preprocessing & Augmentation:** Resize all images to `224 × 224 × 3` (RGB), normalize pixel intensities to `[0, 1]`, and apply moderate, realistic data augmentation strictly to the training split.
4. **Train with Regularization & Callbacks:** Train using `binary_crossentropy` loss, the `Adam` optimizer, balanced class weights, `ModelCheckpoint`, `EarlyStopping`, and `ReduceLROnPlateau`.
5. **Evaluate on an Untouched Test Set:** Compute genuine test metrics (`Accuracy`, `Precision`, `Recall`, `F1-Score`, `ROC-AUC`, `Confusion Matrix`, and per-class report) using `scikit-learn` with **Fake (`1`)** designated as the positive class.
6. **Provide an Interactive Prediction Interface:** Build a Streamlit web application (`app.py`) and CLI utility (`src/predict.py`) to classify uploaded medicine images and explain the raw sigmoid score.

---

## 4. Project Directory Structure

```text
counterfiet-detection/
├── dataset/                          # Original local dataset (untouched on disk)
│   ├── Train/
│   │   ├── Fake/
│   │   └── Real/
│   ├── Val/
│   │   ├── Fake/
│   │   └── Real/
│   └── Test/
│       ├── Fake/
│       └── Real/
├── src/
│   ├── __init__.py
│   ├── data_loader.py                # Dataset audit, MD5 deduplication, preprocessing & tf.data pipelines
│   ├── model.py                      # Custom 11-layer CNN definition & summary exporter
│   ├── train.py                      # Training script with callbacks, class weights & curve plotting
│   ├── evaluate.py                   # Test-set evaluation script using scikit-learn
│   └── predict.py                    # Single-image CLI inference & sigmoid interpretation
├── models/
│   └── best_model.keras              # Best saved CNN model checkpoint (monitored on val_loss)
├── outputs/
│   ├── dataset_audit.json            # Complete dataset verification and duplicate audit report
│   ├── model_summary.txt             # Saved Keras model.summary() table
│   ├── training_history.json         # Epoch-by-epoch training & validation metrics
│   ├── training_validation_accuracy.png  # Training vs. Validation Accuracy curve
│   ├── training_validation_loss.png      # Training vs. Validation Loss curve
│   ├── confusion_matrix.png          # Test set Confusion Matrix heatmap
│   ├── classification_report.txt     # Full scikit-learn classification report
│   └── test_metrics.json             # Structured test metrics JSON
├── app.py                            # Streamlit web application
├── requirements.txt                  # Pinned compatible Python dependencies
├── README.md                         # Project documentation & viva guide
└── .gitignore                        # Git ignore rules
```

---

## 5. Dataset Structure & Integrity Audit

Default dataset path: `C:\Final Projects\counterfiet-detection\dataset`

### Raw Folder Inspection Findings
During Phase 1 verification (`python -m src.data_loader`), the following characteristics were detected in the raw dataset folders on disk:

| Split Folder | Valid `Fake` Images (`.png`) | Valid `Real` Images (`.jpg`) | Total Valid Images | Non-Image Skipped Files | Within-Split Duplicate Files |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Train** | 240 | 421 | **661** | 1 (`Train/Real/desktop.ini`) | 15 |
| **Val** | 167 | 286 | **453** | 1 (`Val/Real/desktop.ini`) | 5 |
| **Test** | 162 | 287 | **449** | 1 (`Test/Real/desktop.ini`) | 5 |
| **Total (Raw)** | **569** | **994** | **1,563** | **3 (`desktop.ini`)** | **25** |

### Duplicate & Data Leakage Prevention
- **Within-Split Duplicates:** 25 duplicate image files were found inside the `Real` folders (e.g., `images01.jpg` and `images248.jpg` have identical MD5 hashes).
- **Cross-Split Overlap on Disk:** MD5 hash comparison revealed that all `448` unique images in `Val` and all `444` unique images in `Test` were exact byte-for-byte copies of images already present in `Train` (`646` unique images total across the dataset: `240 Fake` and `406 Real`).
- **Leakage-Free In-Memory Split (`split_mode="deduplicated"`):** Without moving, renaming, or modifying any files on disk, `src/data_loader.py` collects all **646 unique valid images** (`240 Fake`, `406 Real`) and performs a deterministic, stratified **70% Train / 15% Val / 15% Test** split (`seed=42`) with **0% hash overlap** across splits:

| Prepared Split | `Fake (1)` [Positive] | `Real (0)` [Negative] | Total Unique Images | Percentage |
| :--- | :---: | :---: | :---: | :---: |
| **Train** | 168 | 284 | **452** | 70.0% |
| **Val** | 36 | 61 | **97** | 15.0% |
| **Test** | 36 | 61 | **97** | 15.0% |
| **Total (Unique)** | **240** | **406** | **646** | **100.0%** |

- **Class Imbalance Handling:** The dataset has a moderate class imbalance of `1.692 : 1` (`Real` : `Fake`). To prevent the model from biasing toward the majority `Real` class, balanced class weights (`Real [0] = 0.7958`, `Fake [1] = 1.3452`) are computed via `scikit-learn` and applied during training.

---

## 6. CNN Architecture & College Viva Layer Guide

### Label Mapping Convention
- **`Real` $\rightarrow$ `0` (Negative Class):** Genuine medicine packaging (Sigmoid score `< 0.5`).
- **`Fake` $\rightarrow$ `1` (Positive Class):** Counterfeit / manipulated medicine packaging (Sigmoid score `>= 0.5`).

### Architecture Specification (`109,889` Trainable Parameters)

| Step | Layer Name & Type | Configuration | Output Shape | Param # |
| :---: | :--- | :--- | :---: | :---: |
| **1** | `Input` | `224 × 224 × 3` (Normalized RGB `[0, 1]`) | `(None, 224, 224, 3)` | `0` |
| **2** | `conv2d_32` (`Conv2D`) | 32 filters, `3 × 3` kernel, `ReLU` | `(None, 222, 222, 32)` | `896` |
| **3** | `maxpool_1` (`MaxPooling2D`) | `2 × 2` pool size | `(None, 111, 111, 32)` | `0` |
| **4** | `conv2d_64` (`Conv2D`) | 64 filters, `3 × 3` kernel, `ReLU` | `(None, 109, 109, 64)` | `18,496` |
| **5** | `maxpool_2` (`MaxPooling2D`) | `2 × 2` pool size | `(None, 54, 54, 64)` | `0` |
| **6** | `conv2d_128` (`Conv2D`) | 128 filters, `3 × 3` kernel, `ReLU` | `(None, 52, 52, 128)` | `73,856` |
| **7** | `maxpool_3` (`MaxPooling2D`) | `2 × 2` pool size | `(None, 26, 26, 128)` | `0` |
| **8** | `global_avg_pool` (`GlobalAveragePooling2D`) | Spatial average per feature map | `(None, 128)` | `0` |
| **9** | `dense_128` (`Dense`) | 128 neurons, `ReLU` | `(None, 128)` | `16,512` |
| **10** | `dropout_0_5` (`Dropout`) | `rate = 0.5` | `(None, 128)` | `0` |
| **11** | `output_sigmoid` (`Dense`) | 1 neuron, `sigmoid` activation | `(None, 1)` | `129` |

### Short Viva Explanation of Each Layer
1. **Input Layer (`224 × 224 × 3`):** Accepts standardized 3-channel RGB images of height 224 and width 224 with pixel values scaled to `[0.0, 1.0]`.
2. **`Conv2D` (32 filters, `3 × 3`, ReLU):** Extracts low-level visual primitives such as edges, color gradients, borders, and sharp corners using 32 learnable `3 × 3` filters. ReLU ($\max(0, x)$) introduces non-linearity and prevents vanishing gradients.
3. **`MaxPooling2D` (`2 × 2`):** Downsamples the spatial dimensions by half (`222 × 222` $\rightarrow$ `111 × 111`) by retaining the maximum activation in each `2 × 2` window, providing local translation invariance and reducing computation.
4. **`Conv2D` (64 filters, `3 × 3`, ReLU):** Combines low-level edges into mid-level packaging features such as font strokes, barcode lines, blister grid patterns, and logo shapes.
5. **`MaxPooling2D` (`2 × 2`):** Further compresses spatial dimensions (`109 × 109` $\rightarrow$ `54 × 54`) so subsequent convolutions cover a larger effective receptive field.
6. **`Conv2D` (128 filters, `3 × 3`, ReLU):** Captures high-level semantic patterns across the medicine package, such as overall layout consistency, print quality anomalies, and branding structure.
7. **`MaxPooling2D` (`2 × 2`):** Reduces the feature maps to `26 × 26 × 128`, summarizing high-level spatial activations.
8. **`GlobalAveragePooling2D`:** Computes the average value of each of the 128 `26 × 26` feature maps, producing a compact `128`-dimensional feature vector. Unlike `Flatten`, Global Average Pooling drastically reduces parameters and prevents overfitting on small/medium datasets.
9. **`Dense` (128 neurons, ReLU):** A fully connected layer that learns non-linear combinations of the 128 global visual features to form a discriminative representation.
10. **`Dropout` (`0.5`):** Randomly zeroes 50% of neuron activations during each training step, forcing the network to learn redundant, robust features instead of memorizing specific training images.
11. **`Output Dense` (`1` neuron, `sigmoid`):** Squashes the final logit $z$ into the range $[0, 1]$ via $\sigma(z) = \frac{1}{1 + e^{-z}}$. Values $\ge 0.5$ predict **Fake (`1`)**, and values $< 0.5$ predict **Real (`0`)**.

---

## 7. Data Preprocessing and Augmentation

- **RGB Conversion:** All images (including 4-channel `RGBA` PNG screenshots) are explicitly converted to 3-channel `RGB`.
- **Resizing:** Images of varying native resolutions are resized to `224 × 224` using bilinear interpolation.
- **Normalization:** Pixel values are scaled from `[0, 255]` to `float32` in `[0.0, 1.0]`.
- **Training-Only Data Augmentation:** Moderate random transformations are applied **only** to the training split (`RandomFlip("horizontal")`, `RandomRotation(0.08)`, `RandomZoom(±0.10)`, and `RandomTranslation(0.08)`). Validation and Test splits receive **no** random augmentation.

---

## 8. Genuine Training & Evaluation Results

### Training Summary (`src/train.py`)
- **Configured Epochs:** `30` (with `EarlyStopping(patience=8, restore_best_weights=True)`)
- **Epochs Completed:** `17`
- **Best Validation Epoch:** Epoch `9`
- **Best Validation Loss:** `0.4019`
- **Best Validation Accuracy:** `78.35%`

### Untouched Test Set Results (`src/evaluate.py`)
Evaluated on the **97 untouched test images** (`61 Real`, `36 Fake`) with **Fake (`1`)** as the positive class:

| Metric | Value |
| :--- | :---: |
| **Test Accuracy** | **83.51%** (`0.8351`) |
| **Test Loss (Binary Cross-Entropy)** | **0.3577** |
| **Precision (Positive Class = Fake `1`)** | **0.7941** (`79.41%`) |
| **Recall (Positive Class = Fake `1`)** | **0.7500** (`75.00%`) |
| **F1-Score (Positive Class = Fake `1`)** | **0.7714** (`77.14%`) |
| **ROC-AUC Score** | **0.9185** (`91.85%`) |

#### Per-Class Breakdown
| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Real (`0` - Genuine)** | `0.8571` | `0.8852` | `0.8710` | `61` |
| **Fake (`1` - Counterfeit)** | `0.7941` | `0.7500` | `0.7714` | `36` |
| **Macro Avg** | `0.8256` | `0.8176` | `0.8212` | `97` |
| **Weighted Avg** | `0.8338` | `0.8351` | `0.8340` | `97` |

#### Confusion Matrix
| | Predicted `Real (0)` | Predicted `Fake (1)` |
| :--- | :---: | :---: |
| **Actual `Real (0)`** | **54** (True Negatives) | **7** (False Positives) |
| **Actual `Fake (1)`** | **9** (False Negatives) | **27** (True Positives) |

---

## 9. Setup, Training, Evaluation & Streamlit Instructions

All commands below should be run from the project root (`C:\Final Projects\counterfiet-detection`) in **Windows PowerShell**.

### Step 1: Create and Activate the Python Virtual Environment
```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment in PowerShell
.\.venv\Scripts\Activate.ps1

# Install required packages
python -m pip install -r requirements.txt
```

### Step 2: Inspect Dataset & Verify CNN Architecture
```powershell
python -m src.data_loader
python -m src.model
```

### Step 3: Train the CNN Model
```powershell
python -m src.train --epochs 30 --batch-size 32
```

### Step 4: Evaluate the Saved Model on the Test Split
```powershell
python -m src.evaluate
```

### Step 5: Run Single-Image Prediction from the CLI
```powershell
python -m src.predict "C:\Final Projects\counterfiet-detection\dataset\test\Real\images01.jpg"
```

### Step 6: Launch the Streamlit Web Application
```powershell
streamlit run app.py
```
This opens the interactive web interface in your browser (typically at `http://localhost:8501`), where you can upload any `.jpg`, `.jpeg`, or `.png` medicine package image and inspect the prediction and evaluation charts.

---

## 10. Required Packages (`requirements.txt`)

- `tensorflow==2.19.0`
- `keras==3.10.0`
- `numpy>=1.26.0,<2.2.0`
- `matplotlib==3.10.3`
- `scikit-learn==1.5.1`
- `Pillow==10.3.0`
- `streamlit==1.46.1`

---

## 11. Generated Output Files

| File Path | Description |
| :--- | :--- |
| `models/best_model.keras` | Best trained CNN model saved by `ModelCheckpoint` (lowest `val_loss`). |
| `outputs/dataset_audit.json` | Detailed audit of valid images, skipped `desktop.ini` files, duplicates, and split counts. |
| `outputs/model_summary.txt` | Text export of `model.summary()` showing all 11 layers and parameter counts. |
| `outputs/training_history.json` | Epoch-by-epoch training & validation loss, accuracy, precision, recall, and AUC. |
| `outputs/training_validation_accuracy.png` | Plot of Training vs. Validation Accuracy across epochs. |
| `outputs/training_validation_loss.png` | Plot of Training vs. Validation Loss across epochs. |
| `outputs/confusion_matrix.png` | Annotated heatmap of the Test Set Confusion Matrix. |
| `outputs/classification_report.txt` | Full text report of Test Set metrics and per-class precision/recall/F1. |
| `outputs/test_metrics.json` | Machine-readable JSON containing all Test Set evaluation metrics. |

---

## 12. Limitations and Disclaimer

1. **Visual Packaging Inspection Only:** This CNN model analyzes 2D visual patterns on medicine packaging (color, print layout, typography, surface appearance). It **cannot** chemically analyze pills, capsules, or liquid formulations and cannot guarantee pharmacological authenticity.
2. **Uncalibrated Sigmoid Output:** The model's sigmoid score in `[0.0, 1.0]` represents the raw activation of the final neuron rather than a statistically calibrated probability.
3. **Dataset Domain Scope:** Because the model is trained on a specific local dataset of 646 unique images, predictions on unfamiliar medicine brands or extreme lighting conditions should be treated as academic demonstrations only.

## 13. Quick Commands (Run from the Project Root)
Run the following commands in Windows PowerShell from C:\Final Projects\counterfiet-detection.
1. Activate the virtual environment
.\.venv\Scripts\Activate.ps1

2. Verify dataset integrity and print the CNN architecture summary
python -m src.data_loader
python -m src.model

1. (Optional) Re-train the CNN model
python -m src.train --epochs 30 --batch-size 32

4. Evaluate the saved best model on the untouched Test split
python -m src.evaluate

5. Predict a single medicine image from the CLI
python -m src.predict "C:\Final Projects\counterfiet-detection\dataset\Test\Real\images01.jpg"

6. Launch the interactive Streamlit web application
streamlit run app.py
Note: Run the commands from the project root. The prediction example uses Test with a capital T, matching the dataset folder structure documented above. Replace images01.jpg with the path to an image that actually exists in your dataset.