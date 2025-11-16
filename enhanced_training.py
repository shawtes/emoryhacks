#!/usr/bin/env python3
"""
Enhanced model training with techniques to improve performance.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib


def enhanced_model_training(features_csv: Path, output_dir: Path):
    """
    Enhanced model training with class balancing, feature selection, and hyperparameter tuning.
    """
    print("=== ENHANCED MODEL TRAINING ===")
    
    # Load data
    df = pd.read_csv(features_csv)
    metadata_cols = ['subject_id', 'label', 'filepath']
    feature_cols = [col for col in df.columns if col not in metadata_cols]
    
    X = df[feature_cols].values
    y = df['label'].values
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class distribution: {np.bincount(y)} (0: non-dementia, 1: dementia)")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Enhanced cross-validation with stratification
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    results = []
    fold_predictions = []
    
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), 1):
        print(f"\n--- Fold {fold} ---")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Create enhanced pipeline
        pipeline = create_enhanced_pipeline()
        
        # Train model
        pipeline.fit(X_train, y_train)
        
        # Predictions
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
        
        # Store results
        results.append({
            'fold': fold,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'train_size': len(y_train),
            'test_size': len(y_test),
            'dementia_in_test': np.sum(y_test)
        })
        
        # Store predictions for analysis
        fold_predictions.extend(zip(test_idx, y_test, y_pred, y_pred_proba))
        
        print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}")
        
        # Save fold model
        joblib.dump(pipeline, output_dir / f"enhanced_model_fold_{fold}.joblib")
    
    # Overall results
    results_df = pd.DataFrame(results)
    
    print(f"\n=== ENHANCED RESULTS ===")
    print(f"Mean Accuracy: {results_df['accuracy'].mean():.4f} ± {results_df['accuracy'].std():.4f}")
    print(f"Mean F1-Score: {results_df['f1'].mean():.4f} ± {results_df['f1'].std():.4f}")
    print(f"Mean Precision: {results_df['precision'].mean():.4f} ± {results_df['precision'].std():.4f}")
    print(f"Mean Recall: {results_df['recall'].mean():.4f} ± {results_df['recall'].std():.4f}")
    
    # Save results
    results_df.to_csv(output_dir / "enhanced_results.csv", index=False)
    
    # Detailed analysis
    analyze_predictions(fold_predictions, output_dir)
    
    return output_dir


def create_enhanced_pipeline():
    """
    Create an enhanced ML pipeline with multiple improvement techniques.
    """
    # Pipeline with SMOTE, feature selection, scaling, and tuned Random Forest
    pipeline = ImbPipeline([
        ('scaler', StandardScaler()),
        ('feature_selection', SelectKBest(f_classif, k=50)),  # Select top 50 features
        ('smote', SMOTE(random_state=42, k_neighbors=3)),  # Handle class imbalance
        ('classifier', RandomForestClassifier(
            n_estimators=500,           # More trees
            max_depth=10,              # Limit depth to prevent overfitting
            min_samples_split=5,       # Require more samples to split
            min_samples_leaf=3,        # Require more samples in leaves
            class_weight='balanced',   # Handle class imbalance
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    return pipeline


def analyze_predictions(fold_predictions, output_dir):
    """
    Analyze predictions across all folds.
    """
    # Convert to arrays
    indices, y_true, y_pred, y_proba = zip(*fold_predictions)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_proba = np.array(y_proba)
    
    # Overall confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"\n=== OVERALL CONFUSION MATRIX ===")
    print(f"True Negatives: {cm[0,0]}")
    print(f"False Positives: {cm[0,1]}")
    print(f"False Negatives: {cm[1,0]}")
    print(f"True Positives: {cm[1,1]}")
    
    # Classification report
    report = classification_report(y_true, y_pred, target_names=['Non-Dementia', 'Dementia'])
    print(f"\n=== CLASSIFICATION REPORT ===")
    print(report)
    
    # Save detailed results
    detailed_results = pd.DataFrame({
        'sample_index': indices,
        'true_label': y_true,
        'predicted_label': y_pred,
        'prediction_probability': y_proba
    })
    detailed_results.to_csv(output_dir / "detailed_predictions.csv", index=False)
    
    return detailed_results


def hyperparameter_tuning(features_csv: Path, output_dir: Path):
    """
    Perform hyperparameter tuning to find optimal model settings.
    """
    print("=== HYPERPARAMETER TUNING ===")
    
    # Load data
    df = pd.read_csv(features_csv)
    metadata_cols = ['subject_id', 'label', 'filepath']
    feature_cols = [col for col in df.columns if col not in metadata_cols]
    
    X = df[feature_cols].values
    y = df['label'].values
    
    # Create base pipeline
    base_pipeline = ImbPipeline([
        ('scaler', StandardScaler()),
        ('feature_selection', SelectKBest(f_classif)),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))
    ])
    
    # Parameter grid
    param_grid = {
        'feature_selection__k': [30, 50, 75, 100],
        'classifier__n_estimators': [300, 500, 700],
        'classifier__max_depth': [8, 10, 12, None],
        'classifier__min_samples_split': [3, 5, 7],
        'classifier__class_weight': ['balanced', 'balanced_subsample']
    }
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(
        base_pipeline,
        param_grid,
        cv=3,  # Use 3-fold for faster tuning
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    print("Starting hyperparameter search...")
    grid_search.fit(X, y)
    
    print(f"Best F1 score: {grid_search.best_score_:.4f}")
    print(f"Best parameters: {grid_search.best_params_}")
    
    # Save best model and results
    joblib.dump(grid_search.best_estimator_, output_dir / "best_tuned_model.joblib")
    
    # Save tuning results
    tuning_results = pd.DataFrame(grid_search.cv_results_)
    tuning_results.to_csv(output_dir / "hyperparameter_tuning_results.csv", index=False)
    
    return grid_search.best_estimator_, grid_search.best_params_


def main():
    """
    Run enhanced model training.
    """
    project_root = Path(".")
    features_csv = project_root / "data" / "processed" / "features_clean.csv"
    output_dir = project_root / "reports" / "enhanced_models"
    
    if not features_csv.exists():
        print(f"Features file not found: {features_csv}")
        return
    
    # Run enhanced training
    enhanced_model_training(features_csv, output_dir)
    
    # Run hyperparameter tuning
    print(f"\n" + "="*60)
    hyperparameter_tuning(features_csv, output_dir)
    
    print(f"\n=== ENHANCED TRAINING COMPLETE ===")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
