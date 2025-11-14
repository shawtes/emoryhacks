from pathlib import Path
from typing import List, Tuple

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


def train_rf_cv(
    X_folds: List[Tuple[np.ndarray, np.ndarray]],
    out_dir: Path,
    n_estimators: int = 300,
    max_depth: int = None,
    random_state: int = 42,
) -> None:
    """
    Train RF across folds; save per-fold metrics and models.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = []
    for i, (Xy_train, Xy_val) in enumerate(X_folds, start=1):
        X_train, y_train = Xy_train
        X_val, y_val = Xy_val
        clf = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, n_jobs=-1
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        p, r, f1, _ = precision_recall_fscore_support(y_val, y_pred, average="binary", zero_division=0)
        cm = confusion_matrix(y_val, y_pred).tolist()
        metrics.append({"fold": i, "accuracy": acc, "precision": p, "recall": r, "f1": f1, "confusion_matrix": cm})
        joblib.dump(clf, out_dir / f"rf_fold_{i}.joblib")
    pd.DataFrame(metrics).to_csv(out_dir / "rf_cv_metrics.csv", index=False)
    (out_dir / "rf_cv_metrics.json").write_text(json.dumps(metrics, indent=2))


