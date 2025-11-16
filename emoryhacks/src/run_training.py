from pathlib import Path
from typing import List, Tuple

import json
import numpy as np
import pandas as pd

from .generate_splits import stratified_subject_folds, save_folds
from .ml_train import train_rf_cv
from .ensemble_train import train_ensemble_cv


def _validate_subject_labels(df: pd.DataFrame) -> None:
    grouped = df.groupby("subject_id")["label"].nunique()
    bad = grouped[grouped > 1]
    if not bad.empty:
        raise ValueError(f"Inconsistent labels within subjects: {bad.index.tolist()}")


def _build_folds_Xy(
    df: pd.DataFrame, folds: List[tuple[list[str], list[str]]], feature_cols: List[str]
) -> List[Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]]:
    X_folds = []
    for train_subj, val_subj in folds:
        train_df = df[df["subject_id"].isin(train_subj)]
        val_df = df[df["subject_id"].isin(val_subj)]
        X_train = train_df[feature_cols].to_numpy(dtype=float)
        y_train = train_df["label"].to_numpy(dtype=int)
        X_val = val_df[feature_cols].to_numpy(dtype=float)
        y_val = val_df["label"].to_numpy(dtype=int)
        X_folds.append(((X_train, y_train), (X_val, y_val)))
    return X_folds


def run_all(
    project_root: Path,
    features_csv: Path,
    splits_dir: Path,
    rf_out: Path,
    ensemble_out: Path,
    k_folds: int = 5,
    use_gpu: bool = False,
) -> None:
    """
    Load features CSV, generate subject-wise stratified folds, and train RF + ensembles.
    Writes metrics and models to output directories.
    
    Args:
        project_root: Project root directory
        features_csv: Path to features CSV
        splits_dir: Directory to save CV splits
        rf_out: Output directory for RF models
        ensemble_out: Output directory for ensemble models
        k_folds: Number of CV folds
        use_gpu: Whether to use GPU acceleration (auto-detects if False)
    """
    from .gpu_utils import check_cuda_available
    
    # Auto-detect GPU if not specified
    if use_gpu is None:
        use_gpu = check_cuda_available()
    
    if use_gpu:
        print("GPU acceleration enabled for training")
    else:
        print("Using CPU for training (all 8 logical processors)")
    
    df = pd.read_csv(features_csv)
    for col in ("subject_id", "label"):
        if col not in df.columns:
            raise ValueError(f"Missing column in features CSV: {col}")
    _validate_subject_labels(df)
    # Feature columns: exclude identifiers
    exclude = {"subject_id", "label", "filepath"}
    feature_cols = [c for c in df.columns if c not in exclude]
    # Subject->label map
    subj_lbl = df.groupby("subject_id")["label"].first().to_dict()
    folds = stratified_subject_folds(subj_lbl, k=k_folds, seed=42)
    save_folds(folds, splits_dir)
    X_folds = _build_folds_Xy(df, folds, feature_cols)
    rf_out.mkdir(parents=True, exist_ok=True)
    ensemble_out.mkdir(parents=True, exist_ok=True)
    
    # Use GPU-accelerated training if available
    if use_gpu:
        try:
            from .ml_train_gpu import train_rf_cv_gpu
            train_rf_cv_gpu(X_folds, rf_out, use_gpu=True)
        except ImportError:
            print("Warning: GPU training not available, falling back to CPU")
            train_rf_cv(X_folds, rf_out)
    else:
        train_rf_cv(X_folds, rf_out)
    
    train_ensemble_cv(X_folds, ensemble_out, use_gpu=use_gpu)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    run_all(
        project_root=root,
        features_csv=root / "data" / "processed" / "features.csv",
        splits_dir=root / "data" / "splits",
        rf_out=root / "reports" / "metrics" / "rf",
        ensemble_out=root / "reports" / "metrics" / "ensemble",
        k_folds=5,
    )


