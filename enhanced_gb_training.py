#!/usr/bin/env python3
"""
Enhanced Gradient Boosting Training with Advanced Features (2024 Research)
Single-fold training for speed with latest voice biomarkers
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif, RFE
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class AdvancedGradientBoostingClassifier:
    """
    Enhanced Gradient Boosting with advanced feature selection and preprocessing
    """
    
    def __init__(self, use_feature_selection=True, n_features=50):
        self.use_feature_selection = use_feature_selection
        self.n_features = n_features
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.model = None
        self.feature_names = None
        
    def _create_optimized_gb(self):
        """Create optimized Gradient Boosting classifier based on best results"""
        return GradientBoostingClassifier(
            n_estimators=300,  # Increased for better performance
            learning_rate=0.1,
            max_depth=8,  # Deeper trees for complex patterns
            min_samples_split=3,  # More sensitive to patterns
            min_samples_leaf=1,
            subsample=0.8,  # Add randomness for robustness
            max_features='sqrt',
            random_state=42,
            verbose=0
        )
    
    def _select_features(self, X, y, feature_names):
        """Advanced feature selection using multiple methods"""
        
        print(f"🔍 Feature Selection: {X.shape[1]} → {self.n_features} features")
        
        # Method 1: Statistical selection (F-test)
        selector_stats = SelectKBest(score_func=f_classif, k=min(self.n_features*2, X.shape[1]))
        X_stats = selector_stats.fit_transform(X, y)
        stats_scores = selector_stats.scores_
        
        # Method 2: Recursive Feature Elimination with GB
        temp_model = GradientBoostingClassifier(n_estimators=50, random_state=42)
        selector_rfe = RFE(estimator=temp_model, n_features_to_select=self.n_features)
        selector_rfe.fit(X, y)
        
        # Combine both methods properly
        # Get indices of features selected by statistical method
        stats_indices = selector_stats.get_support()
        rfe_indices = selector_rfe.support_
        
        # Create combined score: statistical score (normalized) + RFE ranking (normalized)
        combined_scores = np.zeros(X.shape[1])
        
        # Add normalized statistical scores
        if np.max(stats_scores) > 0:
            combined_scores += stats_scores / np.max(stats_scores)
        
        # Add normalized RFE scores (lower ranking = better, so invert)
        rfe_ranking_norm = (np.max(selector_rfe.ranking_) - selector_rfe.ranking_) / np.max(selector_rfe.ranking_)
        combined_scores += rfe_ranking_norm
        
        # Select top features based on combined score
        selected_indices = np.argsort(combined_scores)[-self.n_features:]
        
        # Create final selector
        self.feature_selector = selected_indices
        self.selected_feature_names = [feature_names[i] for i in selected_indices]
        
        print("🎯 Top Selected Features:")
        for i, (idx, name) in enumerate(zip(selected_indices[-10:], self.selected_feature_names[-10:])):
            print(f"  {i+1:2d}. {name:<30} (score: {combined_scores[idx]:.3f})")
        
        return X[:, selected_indices]
    
    def fit(self, X, y, feature_names=None):
        """Train the enhanced model"""
        
        self.feature_names = feature_names if feature_names is not None else [f'feature_{i}' for i in range(X.shape[1])]
        
        # Feature selection
        if self.use_feature_selection:
            X_selected = self._select_features(X, y, self.feature_names)
        else:
            X_selected = X
            self.selected_feature_names = self.feature_names
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_selected)
        
        # Train model
        self.model = self._create_optimized_gb()
        self.model.fit(X_scaled, y)
        
        # Feature importance analysis
        feature_importance = self.model.feature_importances_
        self.feature_importance_df = pd.DataFrame({
            'feature': self.selected_feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        X_selected = X[:, self.feature_selector] if self.use_feature_selection else X
        X_scaled = self.scaler.transform(X_selected)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        X_selected = X[:, self.feature_selector] if self.use_feature_selection else X
        X_scaled = self.scaler.transform(X_selected)
        return self.model.predict_proba(X_scaled)
    
    def get_feature_importance(self):
        """Get feature importance ranking"""
        return self.feature_importance_df

def evaluate_with_single_fold(X, y, feature_names, test_size=0.3):
    """
    Evaluate model using single train-test split for speed
    """
    print("🚀 SINGLE-FOLD EVALUATION (Fast Training)")
    
    # Single train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Training distribution: {np.bincount(y_train)}")
    print(f"Test distribution: {np.bincount(y_test)}")
    
    # Train enhanced model
    print("\n🔧 Training Enhanced Gradient Boosting...")
    model = AdvancedGradientBoostingClassifier(use_feature_selection=True, n_features=50)
    model.fit(X_train, y_train, feature_names)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'model': model
    }
    
    return results

def compare_feature_sets(advanced_features_file, basic_features_file):
    """
    Compare performance between advanced and basic features
    """
    print("🆚 FEATURE SET COMPARISON")
    
    results = {}
    
    # Test with advanced features
    if advanced_features_file.exists():
        print("\n📊 Testing Advanced Features (2024 Research)...")
        df_advanced = pd.read_csv(advanced_features_file)
        
        # Prepare data
        metadata_cols = ['subject_id', 'label', 'filepath']
        feature_cols = [col for col in df_advanced.columns if col not in metadata_cols]
        
        X_adv = df_advanced[feature_cols].values
        y_adv = df_advanced['label'].values
        
        # Handle any remaining NaN/inf values
        X_adv = np.nan_to_num(X_adv, nan=0.0, posinf=0.0, neginf=0.0)
        
        print(f"Advanced features: {X_adv.shape[1]} features, {len(X_adv)} samples")
        
        results['Advanced_Features'] = evaluate_with_single_fold(X_adv, y_adv, feature_cols)
    
    # Test with basic features
    if basic_features_file.exists():
        print("\n📊 Testing Basic Features (Baseline)...")
        df_basic = pd.read_csv(basic_features_file)
        
        # Prepare data
        metadata_cols = ['subject_id', 'label', 'filepath']
        feature_cols = [col for col in df_basic.columns if col not in metadata_cols]
        
        X_basic = df_basic[feature_cols].values
        y_basic = df_basic['label'].values
        
        # Handle any remaining NaN/inf values
        X_basic = np.nan_to_num(X_basic, nan=0.0, posinf=0.0, neginf=0.0)
        
        print(f"Basic features: {X_basic.shape[1]} features, {len(X_basic)} samples")
        
        results['Basic_Features'] = evaluate_with_single_fold(X_basic, y_basic, feature_cols)
    
    return results

def analyze_results(results):
    """
    Analyze and visualize results
    """
    print("\n" + "="*60)
    print("🎯 ENHANCED GRADIENT BOOSTING RESULTS")
    print("="*60)
    
    # Performance comparison
    comparison_data = []
    
    for feature_type, result in results.items():
        comparison_data.append({
            'Feature_Set': feature_type,
            'F1_Score': result['f1'],
            'Accuracy': result['accuracy'],
            'Precision': result['precision'],
            'Recall': result['recall']
        })
        
        print(f"\n📊 {feature_type}:")
        print(f"   Accuracy:  {result['accuracy']:.4f}")
        print(f"   F1-Score:  {result['f1']:.4f}")
        print(f"   Precision: {result['precision']:.4f}")
        print(f"   Recall:    {result['recall']:.4f}")
        
        # Classification report
        print(f"\n   Detailed Classification Report:")
        print(classification_report(result['y_test'], result['y_pred'], 
                                  target_names=['Control', 'Dementia']))
    
    # Find best result
    if comparison_data:
        best_result = max(comparison_data, key=lambda x: x['F1_Score'])
        
        print(f"\n🏆 BEST PERFORMANCE:")
        print(f"   Feature Set: {best_result['Feature_Set']}")
        print(f"   F1-Score: {best_result['F1_Score']:.4f}")
        print(f"   Accuracy: {best_result['Accuracy']:.4f}")
        
        # Compare with baseline
        baseline_f1 = 0.4338  # Previous best Tuned_GB
        improvement = best_result['F1_Score'] - baseline_f1
        improvement_pct = (improvement / baseline_f1) * 100
        
        print(f"\n📈 IMPROVEMENT ANALYSIS:")
        print(f"   Previous best F1: {baseline_f1:.4f}")
        print(f"   New best F1: {best_result['F1_Score']:.4f}")
        print(f"   Improvement: {improvement:+.4f} ({improvement_pct:+.2f}%)")
        
        if improvement > 0:
            print("✅ NEW BEST MODEL ACHIEVED!")
        else:
            print("⚠️  Advanced features did not improve performance")
    
    # Save comparison
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv('reports/enhanced_gb_comparison.csv', index=False)
    
    return comparison_data

def save_best_model(results, output_dir: Path):
    """
    Save the best performing model
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find best model
    best_feature_set = max(results.keys(), key=lambda k: results[k]['f1'])
    best_model = results[best_feature_set]['model']
    
    # Save model
    model_path = output_dir / f"enhanced_gb_{best_feature_set.lower()}.joblib"
    joblib.dump(best_model, model_path)
    
    # Save feature importance
    feature_importance = best_model.get_feature_importance()
    importance_path = output_dir / f"feature_importance_{best_feature_set.lower()}.csv"
    feature_importance.to_csv(importance_path, index=False)
    
    print(f"\n💾 BEST MODEL SAVED:")
    print(f"   Model: {model_path}")
    print(f"   Feature importance: {importance_path}")
    
    # Display top features
    print(f"\n🔝 TOP 10 FEATURES ({best_feature_set}):")
    for i, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
        print(f"   {i:2d}. {row['feature']:<35} {row['importance']:.4f}")

def main():
    """
    Main function for enhanced gradient boosting training
    """
    print("🚀 ENHANCED GRADIENT BOOSTING WITH COMBINED FEATURES")
    print("Existing 142 features + 11 new 2024 voice biomarkers = 153 total features")
    
    # File paths
    combined_features_file = Path("data/processed/combined_features.csv")
    basic_features_file = Path("data/processed/features_clean.csv")
    output_dir = Path("reports/enhanced_models")
    
    # Check if combined features exist
    if not combined_features_file.exists():
        print("❌ Combined features not found! Please run advanced_features_extractor.py first")
        return
    
    # Test combined features vs basic features
    results = {}
    
    # Test combined features
    print("\n📊 Testing Combined Features (Basic + Advanced)...")
    df_combined = pd.read_csv(combined_features_file)
    
    # Handle duplicates by removing them
    print(f"Before deduplication: {len(df_combined)} samples")
    df_combined = df_combined.drop_duplicates(subset=['subject_id'], keep='first')
    print(f"After deduplication: {len(df_combined)} samples")
    
    # Prepare data
    metadata_cols = ['subject_id', 'label', 'filepath']
    feature_cols = [col for col in df_combined.columns if col not in metadata_cols]
    
    X_combined = df_combined[feature_cols].values
    y_combined = df_combined['label'].values
    
    # Handle any remaining NaN/inf values
    X_combined = np.nan_to_num(X_combined, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"Combined features: {X_combined.shape[1]} features, {len(X_combined)} samples")
    print(f"Class distribution: {np.bincount(y_combined)}")
    
    results['Combined_Features'] = evaluate_with_single_fold(X_combined, y_combined, feature_cols)
    
    # Test basic features for comparison
    print("\n📊 Testing Basic Features (Baseline)...")
    df_basic = pd.read_csv(basic_features_file)
    
    # Prepare data
    feature_cols_basic = [col for col in df_basic.columns if col not in metadata_cols]
    
    X_basic = df_basic[feature_cols_basic].values
    y_basic = df_basic['label'].values
    
    # Handle any remaining NaN/inf values
    X_basic = np.nan_to_num(X_basic, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"Basic features: {X_basic.shape[1]} features, {len(X_basic)} samples")
    
    results['Basic_Features'] = evaluate_with_single_fold(X_basic, y_basic, feature_cols_basic)
    
    # Analyze results
    comparison_data = analyze_results(results)
    
    # Save best model
    save_best_model(results, output_dir)
    
    print("\n✅ ENHANCED TRAINING COMPLETE!")
    print("🎯 Single-fold training with combined feature set")
    print("📊 Total features: 153 (142 basic + 11 advanced voice biomarkers)")
    print("� New features: Sound objects, voice quality, prosody, formants")

if __name__ == "__main__":
    main()
