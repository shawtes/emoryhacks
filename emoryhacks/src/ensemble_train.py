from pathlib import Path
from typing import List, Tuple, Dict, Any

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def build_base_estimators(random_state: int = 42, use_gpu: bool = False) -> Dict[str, Any]:
    """
    Define a diverse set of base learners. Keep to sklearn to avoid heavy deps.
    Optionally use GPU-accelerated models via cuML.
    """
    if use_gpu:
        try:
            import cuml
            from cuml.ensemble import RandomForestClassifier as cuRF
            rf = cuRF(
                n_estimators=400,
                max_depth=None,
                min_samples_leaf=1,
                random_state=random_state,
                n_streams=1,
            )
        except ImportError:
            use_gpu = False
    
    if not use_gpu:
        rf = RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=1,
            n_jobs=-1,  # Use all 8 logical processors
            random_state=random_state,
        )
    svc = Pipeline(
        [
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("svc", SVC(C=2.0, kernel="rbf", gamma="scale", probability=True, random_state=random_state)),
        ]
    )
    gbc = GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=3, random_state=random_state
    )
    lr = Pipeline(
        [
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("lr", LogisticRegression(max_iter=2000, solver="lbfgs", C=1.0, random_state=random_state)),
        ]
    )
    return {"rf": rf, "svc": svc, "gbc": gbc, "lr": lr}


def soft_voting_classifier(base_estimators: Dict[str, Any]) -> VotingClassifier:
    """
    Soft voting ensemble over base estimators.
    """
    estimators = [(k, v) for k, v in base_estimators.items()]
    return VotingClassifier(estimators=estimators, voting="soft", weights=None, n_jobs=None, flatten_transform=True)


def stacking_classifier(base_estimators: Dict[str, Any], random_state: int = 42) -> StackingClassifier:
    """
    Stacking ensemble with logistic regression meta-learner.
    """
    estimators = [(k, v) for k, v in base_estimators.items()]
    final_est = LogisticRegression(max_iter=2000, solver="lbfgs", C=1.0, random_state=random_state)
    # Use passthrough=True to give meta-learner both base predictions and original features
    return StackingClassifier(estimators=estimators, final_estimator=final_est, passthrough=True, stack_method="auto", n_jobs=None)


def _metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """
    Compute standard binary metrics; y_prob are positive-class probabilities.
    """
    y_pred = (y_prob >= 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float("nan")
    cm = confusion_matrix(y_true, y_pred).tolist()
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1, "roc_auc": auc, "confusion_matrix": cm}


def train_ensemble_cv(
    X_folds: List[Tuple[np.ndarray, np.ndarray]],
    out_dir: Path,
    random_state: int = 42,
    use_gpu: bool = False,
) -> None:
    """
    Train base models and ensembles across CV folds.
    X_folds: list over folds of ((X_train, y_train), (X_val, y_val))
    Saves per-fold models and metrics; returns nothing (writes to disk).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    all_metrics: List[Dict[str, Any]] = []

    for i, (Xy_train, Xy_val) in enumerate(X_folds, start=1):
        X_train, y_train = Xy_train
        X_val, y_val = Xy_val

        base = build_base_estimators(random_state=random_state + i, use_gpu=use_gpu)
        fold_dir = out_dir / f"fold_{i}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        fold_summary: Dict[str, Any] = {"fold": i, "models": {}}

        # Train base models
        for name, model in base.items():
            # Transfer to GPU if using cuML models
            if use_gpu and hasattr(model, 'fit'):
                try:
                    from .gpu_utils import to_gpu, to_cpu
                    X_train_gpu = to_gpu(X_train.astype(np.float32))
                    y_train_gpu = to_gpu(y_train.astype(np.int32))
                    X_val_gpu = to_gpu(X_val.astype(np.float32))
                    model.fit(X_train_gpu, y_train_gpu)
                    # Predict on GPU if model supports it
                    if hasattr(model, 'predict_proba'):
                        y_prob_gpu = model.predict_proba(X_val_gpu)
                        y_prob = to_cpu(y_prob_gpu)[:, 1] if y_prob_gpu.ndim > 1 else to_cpu(y_prob_gpu)
                    elif hasattr(model, 'predict'):
                        y_pred_gpu = model.predict(X_val_gpu)
                        y_prob = to_cpu(y_pred_gpu).astype(float)
                    else:
                        y_prob = model.predict(X_val).astype(float)
                except Exception as e:
                    # Fallback to CPU
                    print(f"Warning: GPU training failed for {name}, using CPU: {e}")
                    model.fit(X_train, y_train)
                    if hasattr(model, "predict_proba"):
                        y_prob = model.predict_proba(X_val)[:, 1]
                    elif hasattr(model, "decision_function"):
                        scores = model.decision_function(X_val)
                        y_prob = 1.0 / (1.0 + np.exp(-scores))
                    else:
                        y_prob = model.predict(X_val).astype(float)
            else:
                model.fit(X_train, y_train)
                if hasattr(model, "predict_proba"):
                    y_prob = model.predict_proba(X_val)[:, 1]
                elif hasattr(model, "decision_function"):
                    scores = model.decision_function(X_val)
                    y_prob = 1.0 / (1.0 + np.exp(-scores))
                else:
                    y_prob = model.predict(X_val).astype(float)
            
            m = _metrics(y_val, y_prob)
            fold_summary["models"][name] = m
            joblib.dump(model, fold_dir / f"{name}.joblib")

        # Soft voting
        vote = soft_voting_classifier(base)
        vote.fit(X_train, y_train)
        y_prob_v = vote.predict_proba(X_val)[:, 1]
        fold_summary["models"]["voting_soft"] = _metrics(y_val, y_prob_v)
        joblib.dump(vote, fold_dir / "voting_soft.joblib")

        # Stacking
        stack = stacking_classifier(base, random_state=random_state + 202)
        stack.fit(X_train, y_train)
        # StackingClassifier exposes predict_proba if available
        if hasattr(stack, "predict_proba"):
            y_prob_s = stack.predict_proba(X_val)[:, 1]
        else:
            scores = stack.decision_function(X_val)
            y_prob_s = 1.0 / (1.0 + np.exp(-scores))
        fold_summary["models"]["stacking"] = _metrics(y_val, y_prob_s)
        joblib.dump(stack, fold_dir / "stacking.joblib")

        all_metrics.append(fold_summary)

    # Persist metrics
    (out_dir / "ensemble_cv_metrics.json").write_text(json.dumps(all_metrics, indent=2))
    # Flatten to CSV
    rows = []
    for fs in all_metrics:
        fold = fs["fold"]
        for model_name, m in fs["models"].items():
            rows.append(
                {
                    "fold": fold,
                    "model": model_name,
                    "accuracy": m["accuracy"],
                    "precision": m["precision"],
                    "recall": m["recall"],
                    "f1": m["f1"],
                    "roc_auc": m.get("roc_auc", np.nan),
                    "confusion_matrix": json.dumps(m["confusion_matrix"]),
                }
            )
    pd.DataFrame(rows).to_csv(out_dir / "ensemble_cv_metrics.csv", index=False)


