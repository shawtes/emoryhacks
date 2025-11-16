"""
GPU-accelerated model training using cuML (RAPIDS).
Optimized for GeForce GTX 1660 Super.
"""
from pathlib import Path
from typing import List, Tuple, Optional

import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

try:
    import cuml
    from cuml.ensemble import RandomForestClassifier as cuRF
    from cuml.preprocessing import StandardScaler as cuStandardScaler
    CUML_AVAILABLE = True
except ImportError:
    CUML_AVAILABLE = False
    cuRF = None
    cuStandardScaler = None

from .gpu_utils import check_cuda_available, to_gpu, to_cpu, clear_gpu_cache


def train_rf_cv_gpu(
    X_folds: List[Tuple[np.ndarray, np.ndarray]],
    out_dir: Path,
    n_estimators: int = 300,
    max_depth: int = None,
    random_state: int = 42,
    use_gpu: Optional[bool] = None,
) -> None:
    """
    Train Random Forest across folds with GPU acceleration.
    Falls back to CPU sklearn if GPU is unavailable.
    """
    if use_gpu is None:
        use_gpu = CUML_AVAILABLE and check_cuda_available()
    
    if not use_gpu or not CUML_AVAILABLE:
        # Fallback to CPU version
        from .ml_train import train_rf_cv
        return train_rf_cv(X_folds, out_dir, n_estimators, max_depth, random_state)
    
    print("Using GPU-accelerated Random Forest (cuML)")
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = []
    
    for i, (Xy_train, Xy_val) in enumerate(X_folds, start=1):
        X_train, y_train = Xy_train
        X_val, y_val = Xy_val
        
        # Transfer to GPU
        X_train_gpu = to_gpu(X_train.astype(np.float32))
        y_train_gpu = to_gpu(y_train.astype(np.int32))
        X_val_gpu = to_gpu(X_val.astype(np.float32))
        
        # Train on GPU
        clf = cuRF(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_streams=1,  # Single stream for GTX 1660 Super
        )
        clf.fit(X_train_gpu, y_train_gpu)
        
        # Predict on GPU
        y_pred_gpu = clf.predict(X_val_gpu)
        y_pred = to_cpu(y_pred_gpu)
        
        # Compute metrics on CPU
        acc = accuracy_score(y_val, y_pred)
        p, r, f1, _ = precision_recall_fscore_support(y_val, y_pred, average="binary", zero_division=0)
        cm = confusion_matrix(y_val, y_pred).tolist()
        
        metrics.append({
            "fold": i,
            "accuracy": acc,
            "precision": p,
            "recall": r,
            "f1": f1,
            "confusion_matrix": cm
        })
        
        # Save model (cuML models can be pickled)
        import joblib
        joblib.dump(clf, out_dir / f"rf_fold_{i}.joblib")
        
        # Clear GPU cache between folds
        clear_gpu_cache()
        
        print(f"Completed fold {i}/{len(X_folds)}")
    
    pd.DataFrame(metrics).to_csv(out_dir / "rf_cv_metrics.csv", index=False)
    (out_dir / "rf_cv_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"GPU training complete. Metrics saved to {out_dir}")



