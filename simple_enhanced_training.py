#!/usr/bin/env python3
"""
Enhanced model training with built-in sklearn techniques to improve performance.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
import joblib
import warnings
warnings.filterwarnings('ignore')


def simple_enhanced_training(features_csv: Path, output_dir: Path):
    """
    Enhanced model training using only built-in sklearn techniques.
    """
    print("=== ENHANCED MODEL TRAINING (No External Dependencies) ===")
    
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
    
    # Test multiple improved models
    models = {
        'Balanced_RF': create_balanced_rf(),
        'Feature_Selected_RF': create_feature_selected_rf(),
        'Tuned_GB': create_tuned_gb(),
        'Scaled_SVM': create_scaled_svm(),
        'Ensemble_Voting': create_voting_ensemble()
    }
    
    # Cross-validation results
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    
    print("\n=== MODEL COMPARISON ===")
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Cross-validation scores
        cv_scores = cross_validate_model(model, X, y, cv)
        results[name] = cv_scores
        
        # Print results
        print(f"{name} Results:")
        print(f"  Accuracy: {cv_scores['accuracy_mean']:.4f} ± {cv_scores['accuracy_std']:.4f}")
        print(f"  F1-Score: {cv_scores['f1_mean']:.4f} ± {cv_scores['f1_std']:.4f}")
        print(f"  Precision: {cv_scores['precision_mean']:.4f} ± {cv_scores['precision_std']:.4f}")
        print(f"  Recall: {cv_scores['recall_mean']:.4f} ± {cv_scores['recall_std']:.4f}")
        
        # Train full model and save
        model.fit(X, y)
        joblib.dump(model, output_dir / f"{name}_model.joblib")
    
    # Find best model
    best_model_name = max(results.keys(), key=lambda k: results[k]['f1_mean'])
    best_score = results[best_model_name]['f1_mean']
    
    print(f"\n🏆 BEST MODEL: {best_model_name}")
    print(f"   F1-Score: {best_score:.4f}")
    print(f"   Improvement over baseline: {((best_score - 0.3507) / 0.3507 * 100):.1f}%")
    
    # Save results
    results_df = pd.DataFrame(results).T
    results_df.to_csv(output_dir / "model_comparison.csv")
    
    # Detailed analysis of best model
    detailed_analysis(models[best_model_name], X, y, cv, output_dir, best_model_name)
    
    return results


def create_balanced_rf():
    """Random Forest with balanced class weights and optimized parameters."""
    return RandomForestClassifier(
        n_estimators=750,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight='balanced_subsample',
        random_state=42,
        n_jobs=-1
    )


def create_feature_selected_rf():
    """Random Forest with feature selection pipeline."""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('feature_selection', SelectKBest(f_classif, k=75)),
        ('classifier', RandomForestClassifier(
            n_estimators=600,
            max_depth=10,
            min_samples_split=5,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ))
    ])


def create_tuned_gb():
    """Gradient Boosting with optimized parameters."""
    return GradientBoostingClassifier(
        n_estimators=400,
        learning_rate=0.03,
        max_depth=4,
        min_samples_split=6,
        min_samples_leaf=3,
        subsample=0.8,
        random_state=42
    )


def create_scaled_svm():
    """SVM with scaling and class weights."""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('feature_selection', SelectKBest(f_classif, k=50)),
        ('classifier', SVC(
            C=10.0,
            kernel='rbf',
            gamma='scale',
            class_weight='balanced',
            probability=True,
            random_state=42
        ))
    ])


def create_voting_ensemble():
    """Voting ensemble of multiple classifiers."""
    from sklearn.ensemble import VotingClassifier
    
    # Individual classifiers
    rf = RandomForestClassifier(
        n_estimators=400, max_depth=10, class_weight='balanced', 
        random_state=42, n_jobs=-1
    )
    
    gb = GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=3, 
        random_state=42
    )
    
    svm_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(C=1.0, kernel='rbf', class_weight='balanced', 
                   probability=True, random_state=42))
    ])
    
    # Voting classifier
    voting_clf = VotingClassifier(
        estimators=[
            ('rf', rf),
            ('gb', gb), 
            ('svm', svm_pipeline)
        ],
        voting='soft'
    )
    
    return voting_clf


def cross_validate_model(model, X, y, cv):
    """Perform cross-validation and return detailed metrics."""
    
    accuracy_scores = []
    f1_scores = []
    precision_scores = []
    recall_scores = []
    
    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Fit model
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy_scores.append(accuracy_score(y_test, y_pred))
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)
        f1_scores.append(f1)
        precision_scores.append(prec)
        recall_scores.append(rec)
    
    return {
        'accuracy_mean': np.mean(accuracy_scores),
        'accuracy_std': np.std(accuracy_scores),
        'f1_mean': np.mean(f1_scores),
        'f1_std': np.std(f1_scores),
        'precision_mean': np.mean(precision_scores),
        'precision_std': np.std(precision_scores),
        'recall_mean': np.mean(recall_scores),
        'recall_std': np.std(recall_scores)
    }


def detailed_analysis(best_model, X, y, cv, output_dir, model_name):
    """Perform detailed analysis of the best model."""
    print(f"\n=== DETAILED ANALYSIS: {model_name} ===")
    
    fold_results = []
    all_predictions = []
    
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Fit model
        best_model.fit(X_train, y_train)
        
        # Predictions
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, 'predict_proba') else None
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
        
        fold_results.append({
            'fold': fold,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'test_size': len(y_test),
            'dementia_cases': np.sum(y_test),
            'predicted_dementia': np.sum(y_pred)
        })
        
        # Store predictions
        for i, (true_label, pred_label) in enumerate(zip(y_test, y_pred)):
            prob = y_pred_proba[i] if y_pred_proba is not None else None
            all_predictions.append({
                'fold': fold,
                'sample_idx': test_idx[i],
                'true_label': true_label,
                'predicted_label': pred_label,
                'probability': prob
            })
    
    # Save detailed results
    fold_df = pd.DataFrame(fold_results)
    pred_df = pd.DataFrame(all_predictions)
    
    fold_df.to_csv(output_dir / f"{model_name}_fold_results.csv", index=False)
    pred_df.to_csv(output_dir / f"{model_name}_predictions.csv", index=False)
    
    # Overall confusion matrix
    y_true = pred_df['true_label'].values
    y_pred = pred_df['predicted_label'].values
    
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"Overall Confusion Matrix:")
    print(f"  True Negatives (Correctly identified non-dementia): {cm[0,0]}")
    print(f"  False Positives (Non-dementia classified as dementia): {cm[0,1]}")
    print(f"  False Negatives (Dementia classified as non-dementia): {cm[1,0]}")
    print(f"  True Positives (Correctly identified dementia): {cm[1,1]}")
    
    # Clinical interpretation
    sensitivity = cm[1,1] / (cm[1,1] + cm[1,0])  # Recall
    specificity = cm[0,0] / (cm[0,0] + cm[0,1])
    
    print(f"\nClinical Metrics:")
    print(f"  Sensitivity (Recall): {sensitivity:.4f} - Ability to detect dementia")
    print(f"  Specificity: {specificity:.4f} - Ability to correctly identify non-dementia")
    
    return fold_df, pred_df


def compare_with_baseline():
    """Compare results with baseline model."""
    print("\n=== COMPARISON WITH BASELINE ===")
    print("Baseline Random Forest:")
    print("  Accuracy: 65.36%")
    print("  F1-Score: 35.07%") 
    print("  Precision: 57.07%")
    print("  Recall: 25.70%")
    print("\nKey Issues with Baseline:")
    print("  - Low recall (missing 74% of dementia cases)")
    print("  - Class imbalance not addressed")
    print("  - No feature selection")
    print("  - Basic hyperparameters")


def main():
    """Run enhanced model training."""
    project_root = Path(".")
    features_csv = project_root / "data" / "processed" / "features_clean.csv"
    output_dir = project_root / "reports" / "enhanced_models_simple"
    
    if not features_csv.exists():
        print(f"Features file not found: {features_csv}")
        return
    
    # Show baseline comparison
    compare_with_baseline()
    
    # Run enhanced training
    results = simple_enhanced_training(features_csv, output_dir)
    
    print(f"\n=== TRAINING COMPLETE ===")
    print(f"Results saved to: {output_dir}")
    print("\nNext steps to further improve:")
    print("  1. Collect more data (current: 355 samples)")
    print("  2. Engineer domain-specific features")
    print("  3. Use deep learning approaches")
    print("  4. Apply data augmentation techniques")


if __name__ == "__main__":
    main()
