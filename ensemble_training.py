#!/usr/bin/env python3
"""
Enhanced ensemble training combining traditional ML with neural networks.
Implements stacking and voting ensembles for improved performance.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

from neural_network_training import (
    CNNModel, LSTMModel, CNNLSTMModel, TransformerModel, 
    AudioDataset, prepare_neural_data, train_model, evaluate_model
)
from torch.utils.data import DataLoader


class NeuralNetworkWrapper:
    """Wrapper to make PyTorch models compatible with scikit-learn."""
    
    def __init__(self, model_class, model_params, sequence_length=10, epochs=30):
        self.model_class = model_class
        self.model_params = model_params
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.model = None
        self.scaler = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def fit(self, X, y):
        """Train the neural network."""
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Prepare sequences
        X_reshaped = self._prepare_sequences(X_scaled)
        
        # Create dataset and loader
        dataset = AudioDataset(X_reshaped, y, self.sequence_length)
        train_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2)
        val_loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=2)
        
        # Initialize model
        self.model = self.model_class(**self.model_params)
        
        # Train
        self.model, _ = train_model(self.model, train_loader, val_loader, epochs=self.epochs)
        
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


def create_ensemble_models(input_dim):
    """Create ensemble of traditional ML and neural network models."""
    
    # Traditional ML models
    traditional_models = {
        'Random_Forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        'Gradient_Boosting': GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        ),
        'SVM': SVC(
            C=1.0,
            kernel='rbf',
            gamma='scale',
            class_weight='balanced',
            probability=True,
            random_state=42
        )
    }
    
    # Calculate features per sequence for neural networks
    sequence_length = 10
    features_per_seq = input_dim // sequence_length
    
    # Neural network models
    neural_models = {
        'CNN_Wrapper': NeuralNetworkWrapper(
            CNNModel, 
            {'input_dim': input_dim},
            epochs=20
        ),
        'LSTM_Wrapper': NeuralNetworkWrapper(
            LSTMModel,
            {'input_dim': features_per_seq, 'hidden_dim': 64, 'num_layers': 2},
            epochs=20
        ),
        'CNN_LSTM_Wrapper': NeuralNetworkWrapper(
            CNNLSTMModel,
            {'input_dim': features_per_seq, 'cnn_channels': 32, 'lstm_hidden': 64},
            epochs=20
        )
    }
    
    return traditional_models, neural_models


def create_stacking_ensemble(base_models, X, y):
    """Create a stacking ensemble with logistic regression as meta-learner."""
    from sklearn.ensemble import StackingClassifier
    
    # Prepare base estimators
    estimators = [(name, model) for name, model in base_models.items()]
    
    # Meta-learner
    meta_learner = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    )
    
    # Create stacking classifier
    stacking_clf = StackingClassifier(
        estimators=estimators,
        final_estimator=meta_learner,
        cv=3,  # Use fewer folds for speed
        n_jobs=-1,
        passthrough=True  # Include original features
    )
    
    return stacking_clf


def create_voting_ensemble(base_models):
    """Create a voting ensemble."""
    estimators = [(name, model) for name, model in base_models.items()]
    
    voting_clf = VotingClassifier(
        estimators=estimators,
        voting='soft',  # Use probability averaging
        n_jobs=-1
    )
    
    return voting_clf


def evaluate_ensemble_models(features_csv: Path, output_dir: Path):
    """Evaluate ensemble models combining traditional ML and neural networks."""
    print("=== ENSEMBLE MODEL EVALUATION ===")
    
    # Load and prepare data
    df = pd.read_csv(features_csv)
    metadata_cols = ['subject_id', 'label', 'filepath']
    feature_cols = [col for col in df.columns if col not in metadata_cols]
    
    X = df[feature_cols].values
    y = df['label'].values
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Create models
    traditional_models, neural_models = create_ensemble_models(X.shape[1])
    all_models = {**traditional_models, **neural_models}
    
    # Cross-validation setup
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Evaluate individual models
    print("\n=== Individual Model Performance ===")
    for model_name, model in all_models.items():
        print(f"\nEvaluating {model_name}...")
        
        try:
            # Cross-validation scores
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring='f1', n_jobs=1)
            
            results[model_name] = {
                'f1_mean': cv_scores.mean(),
                'f1_std': cv_scores.std(),
                'cv_scores': cv_scores
            }
            
            print(f"{model_name} - F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            
        except Exception as e:
            print(f"Error training {model_name}: {str(e)}")
            continue
    
    # Create and evaluate ensemble models
    print("\n=== Ensemble Model Performance ===")
    
    # Select best performing models for ensemble
    best_models = {}
    for model_name, result in sorted(results.items(), key=lambda x: x[1]['f1_mean'], reverse=True)[:5]:
        best_models[model_name] = all_models[model_name]
        print(f"Selected for ensemble: {model_name} (F1: {result['f1_mean']:.4f})")
    
    # Voting ensemble
    print(f"\nEvaluating Voting Ensemble...")
    try:
        voting_ensemble = create_voting_ensemble(best_models)
        voting_scores = cross_val_score(voting_ensemble, X, y, cv=cv, scoring='f1', n_jobs=1)
        
        results['Voting_Ensemble'] = {
            'f1_mean': voting_scores.mean(),
            'f1_std': voting_scores.std(),
            'cv_scores': voting_scores
        }
        
        print(f"Voting Ensemble - F1: {voting_scores.mean():.4f} ± {voting_scores.std():.4f}")
        
    except Exception as e:
        print(f"Error with voting ensemble: {str(e)}")
    
    # Stacking ensemble (with traditional models only for speed)
    print(f"\nEvaluating Stacking Ensemble...")
    try:
        stacking_models = {name: model for name, model in traditional_models.items()}
        stacking_ensemble = create_stacking_ensemble(stacking_models, X, y)
        stacking_scores = cross_val_score(stacking_ensemble, X, y, cv=cv, scoring='f1', n_jobs=1)
        
        results['Stacking_Ensemble'] = {
            'f1_mean': stacking_scores.mean(),
            'f1_std': stacking_scores.std(),
            'cv_scores': stacking_scores
        }
        
        print(f"Stacking Ensemble - F1: {stacking_scores.mean():.4f} ± {stacking_scores.std():.4f}")
        
    except Exception as e:
        print(f"Error with stacking ensemble: {str(e)}")
    
    # Save results
    results_df = pd.DataFrame({
        name: [result['f1_mean'], result['f1_std']] 
        for name, result in results.items()
    }, index=['F1_Mean', 'F1_Std']).T
    
    results_df.to_csv(output_dir / "ensemble_results.csv")
    
    # Train and save best model
    best_model_name = max(results.keys(), key=lambda k: results[k]['f1_mean'])
    best_f1 = results[best_model_name]['f1_mean']
    
    print(f"\n🏆 BEST MODEL: {best_model_name}")
    print(f"   F1-Score: {best_f1:.4f} ± {results[best_model_name]['f1_std']:.4f}")
    
    # Train final model on all data
    if best_model_name in all_models:
        print(f"\nTraining final {best_model_name} on full dataset...")
        final_model = all_models[best_model_name]
        final_model.fit(X, y)
        
        # Save model
        model_path = output_dir / f"best_model_{best_model_name}.joblib"
        joblib.dump(final_model, model_path)
        print(f"Best model saved to: {model_path}")
    
    return results


def compare_with_baseline(ensemble_results: dict):
    """Compare ensemble results with baseline models."""
    print("\n=== PERFORMANCE COMPARISON ===")
    
    # Baseline results (from previous training)
    baseline_results = {
        'Random_Forest_Baseline': 0.3507,
        'Tuned_GB_Baseline': 0.4338,
        'Enhanced_RF_Baseline': 0.3538
    }
    
    # Combine results
    all_results = {**baseline_results, **ensemble_results}
    
    # Sort by F1 score
    sorted_results = sorted(all_results.items(), key=lambda x: x[1]['f1_mean'] if isinstance(x[1], dict) else x[1], reverse=True)
    
    print("Model Performance Ranking:")
    print("-" * 50)
    
    for i, (model_name, result) in enumerate(sorted_results, 1):
        if isinstance(result, dict):
            f1_score = result['f1_mean']
            f1_std = result.get('f1_std', 0)
            print(f"{i:2d}. {model_name:<25} F1: {f1_score:.4f} ± {f1_std:.4f}")
        else:
            print(f"{i:2d}. {model_name:<25} F1: {result:.4f}")
    
    # Calculate improvements
    best_ensemble = max(ensemble_results.keys(), key=lambda k: ensemble_results[k]['f1_mean'])
    best_ensemble_f1 = ensemble_results[best_ensemble]['f1_mean']
    baseline_best = max(baseline_results.values())
    
    improvement = ((best_ensemble_f1 - baseline_best) / baseline_best) * 100
    
    print(f"\n📈 IMPROVEMENT ANALYSIS:")
    print(f"   Previous best: {baseline_best:.4f}")
    print(f"   New best: {best_ensemble_f1:.4f}")
    print(f"   Improvement: {improvement:+.2f}%")


def main():
    """Main ensemble training function."""
    project_root = Path(".")
    features_csv = project_root / "data" / "processed" / "features_clean.csv"
    output_dir = project_root / "reports" / "ensemble_models"
    
    if not features_csv.exists():
        print(f"Features file not found: {features_csv}")
        return
    
    # Check hardware
    if torch.cuda.is_available():
        print(f"🚀 GPU Available: {torch.cuda.get_device_name(0)}")
    
    print(f"🖥️  CPU Cores: {torch.get_num_threads()}")
    
    # Evaluate ensemble models
    ensemble_results = evaluate_ensemble_models(features_csv, output_dir)
    
    # Compare with baseline
    compare_with_baseline(ensemble_results)
    
    print(f"\n=== ENSEMBLE TRAINING COMPLETE ===")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
