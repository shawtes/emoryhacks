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
) -> None:
    """
    Load features CSV, generate subject-wise stratified folds, and train RF + ensembles.
    Writes metrics and models to output directories.
    """
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
    train_rf_cv(X_folds, rf_out)
    train_ensemble_cv(X_folds, ensemble_out)


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


