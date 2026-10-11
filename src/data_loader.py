"""
Data loading, dataset verification, leakage-free splitting, preprocessing, and augmentation
for the AI-Powered Counterfeit Medicine Detection project.

Label Mapping Convention (Explicitly Verified):
    Real -> 0 (Negative class: genuine medicine packaging, sigmoid output < 0.5)
    Fake -> 1 (Positive class: counterfeit medicine packaging, sigmoid output >= 0.5)
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf

# ============================================================================
# Project-Wide Constants & Label Mapping
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_DIR = Path(r"C:\Final Projects\counterfiet-detection\dataset")
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
IMAGE_SIZE: Tuple[int, int] = (IMAGE_HEIGHT, IMAGE_WIDTH)
IMAGE_CHANNELS = 3
INPUT_SHAPE: Tuple[int, int, int] = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

# Positive class is explicitly configured as Fake (1), Negative class as Real (0)
CLASS_TO_INDEX: Dict[str, int] = {
    "Real": 0,
    "Fake": 1,
}
INDEX_TO_CLASS: Dict[int, str] = {
    0: "Real",
    1: "Fake",
}

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}
DEFAULT_BATCH_SIZE = 32
DEFAULT_SEED = 42


@dataclass
class SplitStats:
    """Statistics for a single dataset split (Train, Val, or Test)."""
    split_name: str
    fake_valid_count: int
    real_valid_count: int
    fake_unique_hashes: int
    real_unique_hashes: int
    within_split_duplicates: int
    skipped_files: List[str]

    @property
    def total_valid(self) -> int:
        return self.fake_valid_count + self.real_valid_count


@dataclass
class DatasetAuditReport:
    """Comprehensive audit report of the raw dataset and prepared splits."""
    dataset_path: str
    exists: bool
    missing_folders: List[str]
    raw_split_stats: Dict[str, SplitStats]
    total_raw_valid_images: int
    total_unique_images: int
    unique_fake_images: int
    unique_real_images: int
    cross_split_duplicate_hashes: int
    val_overlap_with_train_pct: float
    test_overlap_with_train_pct: float
    split_mode_used: str
    prepared_counts: Dict[str, Dict[str, int]]
    class_imbalance_ratio_real_to_fake: float


def set_global_seed(seed: int = DEFAULT_SEED) -> None:
    """Set random seeds across Python, NumPy, and TensorFlow for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)


def verify_label_mapping() -> None:
    """Explicitly assert that Fake is the positive class (1) and Real is negative (0)."""
    assert CLASS_TO_INDEX["Real"] == 0, "Real must map to 0 (negative class)."
    assert CLASS_TO_INDEX["Fake"] == 1, "Fake must map to 1 (positive class)."
    assert INDEX_TO_CLASS[0] == "Real", "Index 0 must map to Real."
    assert INDEX_TO_CLASS[1] == "Fake", "Index 1 must map to Fake."


def _resolve_split_dir(dataset_dir: Path, split_name: str) -> Optional[Path]:
    """Locate split directory case-insensitively (e.g., 'Train' or 'train')."""
    if not dataset_dir.exists():
        return None
    for child in dataset_dir.iterdir():
        if child.is_dir() and child.name.lower() == split_name.lower():
            return child
    return None


def _resolve_class_dir(split_dir: Path, class_name: str) -> Optional[Path]:
    """Locate class directory case-insensitively ('Fake' or 'Real')."""
    if not split_dir.exists():
        return None
    for child in split_dir.iterdir():
        if child.is_dir() and child.name.lower() == class_name.lower():
            return child
    return None


def is_valid_image(file_path: Path) -> Tuple[bool, str]:
    """
    Check whether a file has a supported image extension and can be decoded by Pillow.

    Returns:
        Tuple[bool, str]: (is_valid, reason_if_invalid)
    """
    if file_path.suffix.lower() not in VALID_EXTENSIONS:
        return False, f"Unsupported extension '{file_path.suffix}'"
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True, ""
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        return False, f"Corrupted or unreadable image: {exc}"


def inspect_and_prepare_splits(
    dataset_dir: Union[str, Path] = DEFAULT_DATASET_DIR,
    split_mode: str = "deduplicated",
    seed: int = DEFAULT_SEED,
    save_report: bool = True,
) -> Tuple[Dict[str, Tuple[List[Path], List[int]]], DatasetAuditReport]:
    """
    Inspect the dataset directory, verify valid images, report skipped/corrupted files,
    detect duplicates, and return file paths & binary labels for Train, Val, and Test.

    Args:
        dataset_dir: Path to the root dataset folder.
        split_mode:
            - "deduplicated" (default): Collects all unique valid images by MD5 hash across
              the dataset without modifying any files on disk, and performs a stratified
              70% Train / 15% Val / 15% Test split to prevent cross-split data leakage.
            - "folder": Preserves the exact folder assignments in Train, Val, and Test,
              removing only non-image files (e.g., desktop.ini) and within-split duplicates.
        seed: Random seed for reproducible stratified splitting when split_mode="deduplicated".
        save_report: Whether to write the audit report JSON to outputs/dataset_audit.json.

    Returns:
        splits_data: Dict mapping 'Train', 'Val', 'Test' to (file_paths, labels)
        report: DatasetAuditReport instance with full audit metrics
    """
    verify_label_mapping()
    dataset_path = Path(dataset_dir)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset directory does not exist: {dataset_path}. "
            f"Please ensure the dataset is located at '{DEFAULT_DATASET_DIR}'."
        )

    expected_splits = ["Train", "Val", "Test"]
    expected_classes = ["Fake", "Real"]
    missing_folders: List[str] = []

    split_dirs: Dict[str, Dict[str, Path]] = {}
    for split_name in expected_splits:
        s_dir = _resolve_split_dir(dataset_path, split_name)
        if s_dir is None:
            missing_folders.append(split_name)
            continue
        split_dirs[split_name] = {}
        for cls_name in expected_classes:
            c_dir = _resolve_class_dir(s_dir, cls_name)
            if c_dir is None:
                missing_folders.append(f"{split_name}/{cls_name}")
            else:
                split_dirs[split_name][cls_name] = c_dir

    if missing_folders:
        raise FileNotFoundError(
            f"Missing required dataset folders inside {dataset_path}: {missing_folders}"
        )

    raw_split_stats: Dict[str, SplitStats] = {}
    # Track unique hashes per split and globally
    split_hashes: Dict[str, Dict[str, Dict[str, Path]]] = {
        s: {"Fake": {}, "Real": {}} for s in expected_splits
    }
    global_unique_by_class: Dict[str, Dict[str, Path]] = {"Fake": {}, "Real": {}}
    hash_to_splits: Dict[str, set] = defaultdict(set)

    for split_name in expected_splits:
        skipped: List[str] = []
        valid_counts = {"Fake": 0, "Real": 0}
        within_dups = 0

        for cls_name in expected_classes:
            folder = split_dirs[split_name][cls_name]
            for file_path in sorted(folder.iterdir()):
                if not file_path.is_file():
                    continue
                valid, reason = is_valid_image(file_path)
                if not valid:
                    rel_path = f"{split_name}/{cls_name}/{file_path.name} ({reason})"
                    skipped.append(rel_path)
                    continue

                valid_counts[cls_name] += 1
                file_hash = hashlib.md5(file_path.read_bytes()).hexdigest()
                hash_to_splits[file_hash].add(split_name)

                if file_hash in split_hashes[split_name][cls_name]:
                    within_dups += 1
                else:
                    split_hashes[split_name][cls_name][file_hash] = file_path

                if file_hash not in global_unique_by_class[cls_name]:
                    global_unique_by_class[cls_name][file_hash] = file_path

        raw_split_stats[split_name] = SplitStats(
            split_name=split_name,
            fake_valid_count=valid_counts["Fake"],
            real_valid_count=valid_counts["Real"],
            fake_unique_hashes=len(split_hashes[split_name]["Fake"]),
            real_unique_hashes=len(split_hashes[split_name]["Real"]),
            within_split_duplicates=within_dups,
            skipped_files=skipped,
        )

    # Compute cross-split overlap diagnostics
    train_all_hashes = set(split_hashes["Train"]["Fake"]) | set(split_hashes["Train"]["Real"])
    val_all_hashes = set(split_hashes["Val"]["Fake"]) | set(split_hashes["Val"]["Real"])
    test_all_hashes = set(split_hashes["Test"]["Fake"]) | set(split_hashes["Test"]["Real"])

    cross_split_dup_count = sum(1 for s_set in hash_to_splits.values() if len(s_set) > 1)
    val_overlap_pct = (
        100.0 * len(val_all_hashes & train_all_hashes) / max(1, len(val_all_hashes))
    )
    test_overlap_pct = (
        100.0 * len(test_all_hashes & train_all_hashes) / max(1, len(test_all_hashes))
    )

    unique_fake = len(global_unique_by_class["Fake"])
    unique_real = len(global_unique_by_class["Real"])
    total_unique = unique_fake + unique_real
    total_raw_valid = sum(st.total_valid for st in raw_split_stats.values())

    # Prepare final splits (without modifying any files on disk)
    splits_data: Dict[str, Tuple[List[Path], List[int]]] = {}

    if split_mode == "deduplicated":
        all_paths: List[Path] = []
        all_labels: List[int] = []
        for cls_name in ["Real", "Fake"]:
            label = CLASS_TO_INDEX[cls_name]
            for _, p in sorted(global_unique_by_class[cls_name].items()):
                all_paths.append(p)
                all_labels.append(label)

        # Stratified 70% Train, 15% Val, 15% Test split
        train_paths, temp_paths, train_labels, temp_labels = train_test_split(
            all_paths,
            all_labels,
            test_size=0.30,
            stratify=all_labels,
            random_state=seed,
        )
        val_paths, test_paths, val_labels, test_labels = train_test_split(
            temp_paths,
            temp_labels,
            test_size=0.50,
            stratify=temp_labels,
            random_state=seed,
        )

        splits_data["Train"] = (train_paths, train_labels)
        splits_data["Val"] = (val_paths, val_labels)
        splits_data["Test"] = (test_paths, test_labels)

        # Verify zero leakage across prepared splits
        tr_set = {hashlib.md5(p.read_bytes()).hexdigest() for p in train_paths}
        va_set = {hashlib.md5(p.read_bytes()).hexdigest() for p in val_paths}
        te_set = {hashlib.md5(p.read_bytes()).hexdigest() for p in test_paths}
        assert len(tr_set & va_set) == 0, "Data leakage detected between Train and Val!"
        assert len(tr_set & te_set) == 0, "Data leakage detected between Train and Test!"
        assert len(va_set & te_set) == 0, "Data leakage detected between Val and Test!"

    elif split_mode == "folder":
        for split_name in expected_splits:
            paths: List[Path] = []
            labels: List[int] = []
            for cls_name in ["Real", "Fake"]:
                label = CLASS_TO_INDEX[cls_name]
                for _, p in sorted(split_hashes[split_name][cls_name].items()):
                    paths.append(p)
                    labels.append(label)
            splits_data[split_name] = (paths, labels)
    else:
        raise ValueError(f"Unsupported split_mode '{split_mode}'. Choose 'deduplicated' or 'folder'.")

    prepared_counts: Dict[str, Dict[str, int]] = {}
    for split_name, (_, labels) in splits_data.items():
        fake_c = sum(1 for lbl in labels if lbl == CLASS_TO_INDEX["Fake"])
        real_c = sum(1 for lbl in labels if lbl == CLASS_TO_INDEX["Real"])
        prepared_counts[split_name] = {
            "Fake": fake_c,
            "Real": real_c,
            "Total": fake_c + real_c,
        }

    imbalance_ratio = round(unique_real / max(1, unique_fake), 3)

    report = DatasetAuditReport(
        dataset_path=str(dataset_path),
        exists=True,
        missing_folders=missing_folders,
        raw_split_stats=raw_split_stats,
        total_raw_valid_images=total_raw_valid,
        total_unique_images=total_unique,
        unique_fake_images=unique_fake,
        unique_real_images=unique_real,
        cross_split_duplicate_hashes=cross_split_dup_count,
        val_overlap_with_train_pct=round(val_overlap_pct, 2),
        test_overlap_with_train_pct=round(test_overlap_pct, 2),
        split_mode_used=split_mode,
        prepared_counts=prepared_counts,
        class_imbalance_ratio_real_to_fake=imbalance_ratio,
    )

    if save_report:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        audit_file = OUTPUTS_DIR / "dataset_audit.json"
        with open(audit_file, "w", encoding="utf-8") as f:
            json.dump(asdict(report), f, indent=2)

    return splits_data, report


def print_audit_report(report: DatasetAuditReport) -> None:
    """Print a clean, human-readable summary of the dataset inspection report."""
    print("=" * 72)
    print("DATASET INSPECTION & INTEGRITY REPORT")
    print("=" * 72)
    print(f"Dataset Path       : {report.dataset_path}")
    print(f"Path Exists        : {report.exists}")
    print(f"Missing Folders    : {report.missing_folders if report.missing_folders else 'None'}")
    print(f"Label Mapping      : Real -> {CLASS_TO_INDEX['Real']} (Negative), Fake -> {CLASS_TO_INDEX['Fake']} (Positive)")
    print("-" * 72)
    print("RAW FOLDER COUNTS ON DISK:")
    for split_name, st in report.raw_split_stats.items():
        print(
            f"  {split_name:<5} | Valid: {st.total_valid:<4} "
            f"(Fake: {st.fake_valid_count:<3}, Real: {st.real_valid_count:<3}) | "
            f"Unique Hashes: {st.fake_unique_hashes + st.real_unique_hashes:<4} | "
            f"Within-Split Dups: {st.within_split_duplicates:<2} | "
            f"Skipped: {len(st.skipped_files)}"
        )
        for skipped_item in st.skipped_files:
            print(f"          -> Skipped non-image/invalid file: {skipped_item}")
    print("-" * 72)
    print("DUPLICATE & DATA LEAKAGE ANALYSIS:")
    print(f"  Total Raw Valid Images Across Folders : {report.total_raw_valid_images}")
    print(
        f"  Total Unique Images (MD5 Deduplicated): {report.total_unique_images} "
        f"(Fake: {report.unique_fake_images}, Real: {report.unique_real_images})"
    )
    print(f"  Cross-Split Duplicate Image Groups    : {report.cross_split_duplicate_hashes}")
    print(f"  Raw Val Overlap with Raw Train        : {report.val_overlap_with_train_pct:.1f}%")
    print(f"  Raw Test Overlap with Raw Train       : {report.test_overlap_with_train_pct:.1f}%")
    print(f"  Class Imbalance Ratio (Real : Fake)   : {report.class_imbalance_ratio_real_to_fake} : 1")
    print("-" * 72)
    print(f"ACTIVE SPLIT CONFIGURATION (Mode: '{report.split_mode_used}'):")
    for split_name, counts in report.prepared_counts.items():
        print(
            f"  {split_name:<5} | Total: {counts['Total']:<4} "
            f"(Fake [1]: {counts['Fake']:<3}, Real [0]: {counts['Real']:<3})"
        )
    print("=" * 72)


def preprocess_single_image(
    image_source: Union[str, Path, Image.Image],
    target_size: Tuple[int, int] = IMAGE_SIZE,
) -> np.ndarray:
    """
    Load and preprocess a single image for model inference or training.
    Steps:
        1. Open image via Pillow (if path provided)
        2. Convert to 3-channel RGB (strips alpha channel from RGBA PNGs)
        3. Resize to 224 x 224 using bilinear interpolation
        4. Normalize pixel values from [0, 255] to [0.0, 1.0] in float32

    Args:
        image_source: File path or PIL.Image.Image object.
        target_size: Target (width, height), defaults to (224, 224).

    Returns:
        np.ndarray of shape (224, 224, 3) with float32 values in [0.0, 1.0].
    """
    if isinstance(image_source, (str, Path)):
        with Image.open(image_source) as img:
            rgb_img = img.convert("RGB")
            resized_img = rgb_img.resize(target_size, Image.Resampling.BILINEAR)
            arr = np.asarray(resized_img, dtype=np.float32) / 255.0
    elif isinstance(image_source, Image.Image):
        rgb_img = image_source.convert("RGB")
        resized_img = rgb_img.resize(target_size, Image.Resampling.BILINEAR)
        arr = np.asarray(resized_img, dtype=np.float32) / 255.0
    else:
        raise TypeError(f"Unsupported image_source type: {type(image_source)}")

    return arr


def _load_images_to_arrays(
    file_paths: List[Path],
    labels: List[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Load a list of image paths and labels into normalized float32 NumPy arrays."""
    num_samples = len(file_paths)
    x_data = np.zeros((num_samples, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS), dtype=np.float32)
    y_data = np.asarray(labels, dtype=np.float32)

    for idx, path in enumerate(file_paths):
        x_data[idx] = preprocess_single_image(path)

    return x_data, y_data


def build_augmentation_pipeline(seed: int = DEFAULT_SEED) -> tf.keras.Sequential:
    """
    Create moderate, realistic data augmentation layers applied ONLY to training batches.
    Includes small random rotations, zoom, translations, and horizontal flips.
    """
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal", seed=seed),
            tf.keras.layers.RandomRotation(factor=0.08, fill_mode="nearest", seed=seed),
            tf.keras.layers.RandomZoom(height_factor=(-0.10, 0.10), width_factor=(-0.10, 0.10), fill_mode="nearest", seed=seed),
            tf.keras.layers.RandomTranslation(height_factor=0.08, width_factor=0.08, fill_mode="nearest", seed=seed),
        ],
        name="train_augmentation",
    )


def create_tf_datasets(
    dataset_dir: Union[str, Path] = DEFAULT_DATASET_DIR,
    batch_size: int = DEFAULT_BATCH_SIZE,
    split_mode: str = "deduplicated",
    seed: int = DEFAULT_SEED,
) -> Tuple[
    tf.data.Dataset,
    tf.data.Dataset,
    tf.data.Dataset,
    Dict[int, float],
    DatasetAuditReport,
    Dict[str, Tuple[List[Path], List[int]]],
]:
    """
    Build TensorFlow datasets for Train, Val, and Test splits, compute balanced class weights,
    and return the audit report and raw split file lists.

    Augmentation is applied ONLY to the training dataset.
    Validation and Test datasets are deterministic and unaugmented.
    """
    set_global_seed(seed)
    splits_data, report = inspect_and_prepare_splits(
        dataset_dir=dataset_dir,
        split_mode=split_mode,
        seed=seed,
        save_report=True,
    )

    train_paths, train_labels = splits_data["Train"]
    val_paths, val_labels = splits_data["Val"]
    test_paths, test_labels = splits_data["Test"]

    x_train, y_train = _load_images_to_arrays(train_paths, train_labels)
    x_val, y_val = _load_images_to_arrays(val_paths, val_labels)
    x_test, y_test = _load_images_to_arrays(test_paths, test_labels)

    # Compute balanced class weights from training labels to counter class imbalance
    classes_present = np.array([0, 1], dtype=np.int32)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes_present,
        y=np.asarray(train_labels, dtype=np.int32),
    )
    class_weights: Dict[int, float] = {
        0: float(weights[0]),
        1: float(weights[1]),
    }

    augmenter = build_augmentation_pipeline(seed=seed)

    # Create Train tf.data.Dataset (shuffled + moderate augmentation)
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.shuffle(buffer_size=len(x_train), seed=seed, reshuffle_each_iteration=True)
    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.map(
        lambda x, y: (tf.clip_by_value(augmenter(x, training=True), 0.0, 1.0), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    # Create Val tf.data.Dataset (no shuffle, no augmentation)
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Create Test tf.data.Dataset (no shuffle, no augmentation)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds, class_weights, report, splits_data


if __name__ == "__main__":
    _, audit_report = inspect_and_prepare_splits()
    print_audit_report(audit_report)
