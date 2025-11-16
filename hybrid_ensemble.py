#!/usr/bin/env python3
"""
Hybrid Stacking Ensemble combining Neural Networks with Traditional ML.
Single-fold training for efficiency. Integrates advanced voice biomarkers.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.pipeline import Pipeline
import joblib
import warnings
warnings.filterwarnings('ignore')

from neural_network_training import CNNModel, LSTMModel, CNNLSTMModel, AudioDataset, train_model
from advanced_features import extract_advanced_features_batch


class HybridNeuralWrapper:
    """
    Wrapper to make PyTorch neural networks compatible with scikit-learn pipelines.
    Optimized for single-fold training.
    """
    
    def __init__(self, model_class, model_params, sequence_length=10, epochs=20):
        self.model_class = model_class
        self.model_params = model_params
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.model = None
        self.scaler = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def fit(self, X, y):
        """Train the neural network."""
        print(f"  Training {self.model_class.__name__} with {len(X)} samples...")
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Prepare sequences for neural networks
        X_reshaped = self._prepare_sequences(X_scaled)
        
        # Split for training/validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_reshaped, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create datasets and loaders
        train_dataset = AudioDataset(X_train, y_train, self.sequence_length)
        val_dataset = AudioDataset(X_val, y_val, self.sequence_length)
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
        
        # Initialize and train model
        self.model = self.model_class(**self.model_params)
        self.model, history = train_model(self.model, train_loader, val_loader, epochs=self.epochs)
        
        print(f"    Best validation accuracy: {history['best_val_acc']:.4f}")
        return self
    
    def predict(self, X):
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained yet.")
            
        # Scale and reshape features
        X_scaled = self.scaler.transform(X)
        X_reshaped = self._prepare_sequences(X_scaled)
        
        # Create dataset and loader
        dataset = AudioDataset(X_reshaped, np.zeros(len(X)), self.sequence_length)
        loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=2)
        
        # Predict
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for batch_features, _ in loader:
                batch_features = batch_features.to(self.device)
                outputs = self.model(batch_features)
                _, predicted = torch.max(outputs.data, 1)
                predictions.extend(predicted.cpu().numpy())
        
        return np.array(predictions)
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        if self.model is None:
            raise ValueError("Model not trained yet.")
            
        # Scale and reshape features
        X_scaled = self.scaler.transform(X)
        X_reshaped = self._prepare_sequences(X_scaled)
        
        # Create dataset and loader
        dataset = AudioDataset(X_reshaped, np.zeros(len(X)), self.sequence_length)
        loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=2)
        
        # Predict probabilities
        self.model.eval()
        probabilities = []
        
        with torch.no_grad():
            for batch_features, _ in loader:
                batch_features = batch_features.to(self.device)
                outputs = self.model(batch_features)
                probs = torch.softmax(outputs, dim=1)
                probabilities.extend(probs.cpu().numpy())
        
        return np.array(probabilities)
    
    def _prepare_sequences(self, X):
        """Prepare data for sequence modeling."""
        n_samples, n_features = X.shape
        features_per_seq = n_features // self.sequence_length
        
        # Reshape to (samples, sequence_length, features_per_sequence)
        X_reshaped = X[:, :features_per_seq*self.sequence_length].reshape(
            n_samples, self.sequence_length, features_per_seq
        )
        
        return X_reshaped


def create_hybrid_base_models(input_dim):
    """Create a combination of traditional ML and neural network base models."""
    
    # Traditional ML models - these perform well on small datasets
    traditional_models = {
        'Enhanced_RF': RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=3,
            min_samples_leaf=1,
            class_weight='balanced_subsample',
            random_state=42,
            n_jobs=-1
        ),
        'Tuned_GB': GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.08,
            max_depth=6,
            min_samples_split=3,
            min_samples_leaf=1,
            subsample=0.8,
            random_state=42
        ),
        'Optimized_SVM': SVC(
            C=2.0,
            kernel='rbf',
            gamma='scale',
            class_weight='balanced',
            probability=True,
            random_state=42
        )
    }
    
    # Neural network models - lightweight versions for better performance
    sequence_length = 10
    features_per_seq = input_dim // sequence_length
    
    neural_models = {
        'Compact_LSTM': HybridNeuralWrapper(
            LSTMModel,
            {'input_dim': features_per_seq, 'hidden_dim': 64, 'num_layers': 1},  # Simplified
            epochs=15
        ),
        'Lightweight_CNN': HybridNeuralWrapper(
            CNNModel,
            {'input_dim': input_dim},
            epochs=15
        )
    }
    
    # Combine all models
    all_models = {**traditional_models, **neural_models}
    
    print(f"Created {len(traditional_models)} traditional ML + {len(neural_models)} neural network base models")
    return all_models


def create_advanced_stacking_ensemble(base_models, X, y):
    """Create an advanced stacking ensemble with neural network integration."""
    
    print("=== CREATING ADVANCED STACKING ENSEMBLE ===")
    
    # Prepare base estimators
    estimators = [(name, model) for name, model in base_models.items()]
    
    # Advanced meta-learner with regularization
    meta_learner = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
    
    # Create stacking classifier
    stacking_clf = StackingClassifier(
        estimators=estimators,
        final_estimator=meta_learner,
        cv=3,  # Use 3-fold for meta-learning
        n_jobs=1,  # Avoid conflicts with neural network training
        passthrough=False  # Don't include original features to avoid overfitting
    )
    
    print("Advanced stacking ensemble created with:")
    print(f"  • {len(estimators)} base models")
    print(f"  • {meta_learner.__class__.__name__} meta-learner")
    print(f"  • 3-fold cross-validation for meta-features")
    
    return stacking_clf


def train_hybrid_ensemble(features_csv, output_dir):
    """Train hybrid ensemble combining traditional ML and neural networks."""
    print("=== HYBRID NEURAL-ML ENSEMBLE TRAINING ===")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load features
    print("Loading feature data...")
    df = pd.read_csv(features_csv)
    
    # Separate metadata from features
    metadata_cols = ['subject_id', 'label', 'filepath']
    if 'filepath' not in df.columns:
        metadata_cols.remove('filepath')
    
    feature_cols = [col for col in df.columns if col not in metadata_cols]
    
    X = df[feature_cols].values
    y = df['label'].values
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class distribution: {np.bincount(y)} (0=Normal, 1=Dementia)")
    
    # Single train/test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Create base models
    print("\n=== CREATING BASE MODELS ===")
    base_models = create_hybrid_base_models(X.shape[1])
    
    # Train individual base models and evaluate
    print("\n=== TRAINING BASE MODELS ===")
    base_results = {}
    
    for model_name, model in base_models.items():
        print(f"\nTraining {model_name}...")
        
        try:
            # Train model
            model.fit(X_train, y_train)
            
            # Predict on test set
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            acc = accuracy_score(y_test, y_pred)
            prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
            
            base_results[model_name] = {
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1': f1
            }
            
            print(f"  Accuracy: {acc:.4f}, F1: {f1:.4f}")
            
            # Save individual model
            model_path = output_dir / f"{model_name}_model.joblib"
            joblib.dump(model, model_path)
            
        except Exception as e:
            print(f"  ERROR training {model_name}: {e}")
            base_results[model_name] = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}
    
    # Create and train stacking ensemble
    print("\n=== TRAINING STACKING ENSEMBLE ===")
    
    # Select best performing base models for stacking
    valid_models = {name: model for name, model in base_models.items() 
                   if base_results[name]['f1'] > 0.1}  # Only include models that learned something
    
    if len(valid_models) >= 2:
        print(f"Using {len(valid_models)} valid base models for stacking")
        
        stacking_ensemble = create_advanced_stacking_ensemble(valid_models, X_train, y_train)
        
        # Train stacking ensemble
        print("Training stacking ensemble...")
        stacking_ensemble.fit(X_train, y_train)
        
        # Evaluate stacking ensemble
        y_pred_stack = stacking_ensemble.predict(X_test)
        
        acc_stack = accuracy_score(y_test, y_pred_stack)
        prec_stack, rec_stack, f1_stack, _ = precision_recall_fscore_support(y_test, y_pred_stack, average='binary')
        
        stacking_results = {
            'accuracy': acc_stack,
            'precision': prec_stack,
            'recall': rec_stack,
            'f1': f1_stack
        }
        
        print(f"Stacking Ensemble Results:")
        print(f"  Accuracy: {acc_stack:.4f}")
        print(f"  F1-Score: {f1_stack:.4f}")
        print(f"  Precision: {prec_stack:.4f}")
        print(f"  Recall: {rec_stack:.4f}")
        
        # Save stacking ensemble
        ensemble_path = output_dir / "hybrid_stacking_ensemble.joblib"
        joblib.dump(stacking_ensemble, ensemble_path)
        print(f"Stacking ensemble saved to: {ensemble_path}")
        
    else:
        print("Not enough valid base models for stacking ensemble")
        stacking_results = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}
    
    # Compile results
    all_results = {**base_results, 'Hybrid_Stacking_Ensemble': stacking_results}
    
    # Save results
    results_df = pd.DataFrame(all_results).T
    results_csv = output_dir / "hybrid_ensemble_results.csv"
    results_df.to_csv(results_csv)
    
    # Find best model
    best_model_name = max(all_results.keys(), key=lambda k: all_results[k]['f1'])
    best_f1 = all_results[best_model_name]['f1']
    
    print(f"\n🏆 BEST HYBRID MODEL: {best_model_name}")
    print(f"   F1-Score: {best_f1:.4f}")
    print(f"   Accuracy: {all_results[best_model_name]['accuracy']:.4f}")
    
    # Detailed classification report for best model
    if best_model_name == 'Hybrid_Stacking_Ensemble' and len(valid_models) >= 2:
        print(f"\n=== DETAILED CLASSIFICATION REPORT (Stacking Ensemble) ===")
        print(classification_report(y_test, y_pred_stack, target_names=['Normal', 'Dementia']))
    
    print(f"\nResults saved to: {results_csv}")
    return all_results


def main():
    """Main training function for hybrid ensemble."""
    
    # Check if advanced features exist, if not extract them
    advanced_features_csv = Path("data/processed/advanced_features.csv")
    
    if not advanced_features_csv.exists():
        print("Advanced features not found. Extracting advanced voice biomarkers...")
        
        # Extract advanced features from audio data
        audio_dir = Path("data/processed")
        if audio_dir.exists():
            extract_advanced_features_batch(audio_dir, advanced_features_csv)
        else:
            print("Audio directory not found. Using existing features.")
            advanced_features_csv = Path("data/processed/features_clean.csv")
    
    # Check hardware capabilities
    if torch.cuda.is_available():
        print(f"🚀 GPU Available: {torch.cuda.get_device_name(0)}")
    print(f"🖥️  CPU Cores: {torch.get_num_threads()}")
    
    # Train hybrid ensemble
    output_dir = Path("reports/hybrid_ensemble")
    results = train_hybrid_ensemble(advanced_features_csv, output_dir)
    
    # Compare with previous benchmarks
    print(f"\n=== PERFORMANCE COMPARISON ===")
    baseline_f1 = 0.4338  # Previous best (Tuned_GB)
    
    best_hybrid_f1 = max(results.values(), key=lambda x: x['f1'])['f1']
    improvement = ((best_hybrid_f1 - baseline_f1) / baseline_f1) * 100
    
    print(f"Previous best F1-score: {baseline_f1:.4f}")
    print(f"New hybrid best F1-score: {best_hybrid_f1:.4f}")
    print(f"Improvement: {improvement:+.2f}%")
    
    print(f"\n=== HYBRID ENSEMBLE TRAINING COMPLETE ===")


if __name__ == "__main__":
    main()
