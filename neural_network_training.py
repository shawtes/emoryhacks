#!/usr/bin/env python3
"""
Neural Network models for dementia detection with CNN, LSTM, and hybrid architectures.
Uses multi-core processing for efficient training.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set device for GPU acceleration if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class AudioDataset(Dataset):
    """Custom dataset for audio features."""
    
    def __init__(self, features, labels, sequence_length=10):
        self.features = features
        self.labels = labels
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        # Reshape features for sequence modeling
        feature = self.features[idx]
        
        # Create sequences by reshaping features
        if len(feature.shape) == 1:
            # Reshape 1D features into sequence format
            feature_seq = feature.reshape(self.sequence_length, -1)
        else:
            feature_seq = feature
            
        return torch.FloatTensor(feature_seq), torch.LongTensor([self.labels[idx]])


class CNNModel(nn.Module):
    """1D CNN for audio feature processing."""
    
    def __init__(self, input_dim, num_classes=2):
        super(CNNModel, self).__init__()
        self.input_dim = input_dim
        
        # 1D Convolutional layers
        self.conv1 = nn.Conv1d(1, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        
        # Pooling and normalization
        self.pool = nn.MaxPool1d(2)
        self.batch_norm1 = nn.BatchNorm1d(64)
        self.batch_norm2 = nn.BatchNorm1d(128)
        self.batch_norm3 = nn.BatchNorm1d(256)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
        
        # Activation functions
        self.relu = nn.ReLU()
        
        # Calculate flattened size
        self.flatten_size = self._calculate_flatten_size()
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flatten_size, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
    def _calculate_flatten_size(self):
        # Calculate size after convolutions and pooling
        x = torch.randn(1, 1, self.input_dim)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        return x.view(1, -1).size(1)
    
    def forward(self, x):
        # Reshape for 1D convolution
        if len(x.shape) == 3:
            batch_size, seq_len, features = x.shape
            x = x.view(batch_size, 1, seq_len * features)
        else:
            x = x.unsqueeze(1)  # Add channel dimension
        
        # Convolutional layers
        x = self.pool(self.relu(self.batch_norm1(self.conv1(x))))
        x = self.dropout(x)
        
        x = self.pool(self.relu(self.batch_norm2(self.conv2(x))))
        x = self.dropout(x)
        
        x = self.pool(self.relu(self.batch_norm3(self.conv3(x))))
        x = self.dropout(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


class LSTMModel(nn.Module):
    """LSTM for temporal sequence modeling."""
    
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, num_classes=2):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=0.2, bidirectional=True)
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_dim * 2, 1)
        
        # Output layers
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(hidden_dim * 2, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, num_classes)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Attention mechanism
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Fully connected layers
        x = self.dropout(context_vector)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


class CNNLSTMModel(nn.Module):
    """Hybrid CNN-LSTM model combining spatial and temporal features."""
    
    def __init__(self, input_dim, cnn_channels=64, lstm_hidden=128, num_classes=2):
        super(CNNLSTMModel, self).__init__()
        
        # CNN for feature extraction
        self.conv1 = nn.Conv1d(1, cnn_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(cnn_channels, cnn_channels*2, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.batch_norm1 = nn.BatchNorm1d(cnn_channels)
        self.batch_norm2 = nn.BatchNorm1d(cnn_channels*2)
        
        # Activation function
        self.relu = nn.ReLU()
        
        # Calculate CNN output size
        self.cnn_output_size = self._calculate_cnn_output_size(input_dim, cnn_channels)
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(self.cnn_output_size, lstm_hidden, batch_first=True, 
                           bidirectional=True, dropout=0.2)
        
        # Output layers
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(lstm_hidden * 2, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
    def _calculate_cnn_output_size(self, input_dim, cnn_channels):
        x = torch.randn(1, 1, input_dim)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        return x.size(-1) * cnn_channels * 2
    
    def forward(self, x):
        batch_size, seq_len, features = x.shape
        
        # Reshape for CNN processing
        x = x.view(batch_size * seq_len, 1, features)
        
        # CNN feature extraction
        x = self.pool(self.relu(self.batch_norm1(self.conv1(x))))
        x = self.pool(self.relu(self.batch_norm2(self.conv2(x))))
        
        # Flatten CNN output
        x = x.view(batch_size, seq_len, -1)
        
        # LSTM temporal modeling
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use last hidden state
        x = lstm_out[:, -1, :]
        
        # Output layers
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class TransformerModel(nn.Module):
    """Transformer model for sequence modeling."""
    
    def __init__(self, input_dim, d_model=256, nhead=8, num_layers=4, num_classes=2):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = nn.Parameter(torch.randn(1000, d_model))
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Output layers
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(d_model, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        seq_len = x.size(1)
        
        # Project input to model dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        x += self.pos_encoder[:seq_len, :].unsqueeze(0)
        
        # Apply transformer
        x = self.transformer_encoder(x)
        
        # Global average pooling
        x = torch.mean(x, dim=1)
        
        # Output layers
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


def prepare_neural_data(features_csv: Path, sequence_length=10):
    """Prepare data for neural network training."""
    print("Preparing neural network data...")
    
    # Load features
    df = pd.read_csv(features_csv)
    metadata_cols = ['subject_id', 'label', 'filepath']
    feature_cols = [col for col in df.columns if col not in metadata_cols]
    
    X = df[feature_cols].values
    y = df['label'].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Reshape for sequence modeling
    n_samples, n_features = X_scaled.shape
    features_per_seq = n_features // sequence_length
    
    # Reshape to (samples, sequence_length, features_per_sequence)
    X_reshaped = X_scaled[:, :features_per_seq*sequence_length].reshape(
        n_samples, sequence_length, features_per_seq
    )
    
    print(f"Data shape: {X_reshaped.shape}")
    print(f"Sequence length: {sequence_length}, Features per timestep: {features_per_seq}")
    
    return X_reshaped, y, scaler, features_per_seq


def train_model(model, train_loader, val_loader, epochs=50, lr=0.001):
    """Train a neural network model."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    best_val_acc = 0.0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.squeeze().to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.squeeze().to(device)
                
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_acc = correct / total
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    return model, {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'best_val_acc': best_val_acc
    }


def evaluate_model(model, test_loader):
    """Evaluate model performance."""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.squeeze().to(device)
            
            outputs = model(batch_features)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
    
    # Calculate metrics
    acc = accuracy_score(all_labels, all_predictions)
    prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='binary')
    
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'predictions': all_predictions,
        'labels': all_labels
    }


def train_neural_networks(features_csv: Path, output_dir: Path):
    """Train multiple neural network architectures."""
    print("=== NEURAL NETWORK TRAINING ===")
    
    # Prepare data
    X, y, scaler, features_per_seq = prepare_neural_data(features_csv)
    
    # Models to train
    models = {
        'CNN': lambda: CNNModel(X.shape[1] * X.shape[2]),
        'LSTM': lambda: LSTMModel(features_per_seq),
        'CNN_LSTM': lambda: CNNLSTMModel(features_per_seq),
        'Transformer': lambda: TransformerModel(features_per_seq)
    }
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for model_name, model_fn in models.items():
        print(f"\n=== Training {model_name} ===")
        
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
            print(f"Fold {fold}/5")
            
            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Create data loaders
            train_dataset = AudioDataset(X_train, y_train)
            val_dataset = AudioDataset(X_val, y_val)
            
            train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
            val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
            
            # Train model
            model = model_fn()
            trained_model, training_history = train_model(model, train_loader, val_loader, epochs=30)
            
            # Evaluate
            eval_results = evaluate_model(trained_model, val_loader)
            fold_results.append(eval_results)
            
            # Save model
            torch.save(trained_model.state_dict(), output_dir / f"{model_name}_fold_{fold}.pth")
            
            print(f"Fold {fold} - Acc: {eval_results['accuracy']:.4f}, F1: {eval_results['f1']:.4f}")
        
        # Aggregate results
        avg_results = {
            'accuracy_mean': np.mean([r['accuracy'] for r in fold_results]),
            'accuracy_std': np.std([r['accuracy'] for r in fold_results]),
            'f1_mean': np.mean([r['f1'] for r in fold_results]),
            'f1_std': np.std([r['f1'] for r in fold_results]),
            'precision_mean': np.mean([r['precision'] for r in fold_results]),
            'precision_std': np.std([r['precision'] for r in fold_results]),
            'recall_mean': np.mean([r['recall'] for r in fold_results]),
            'recall_std': np.std([r['recall'] for r in fold_results])
        }
        
        results[model_name] = avg_results
        
        print(f"{model_name} Results:")
        print(f"  Accuracy: {avg_results['accuracy_mean']:.4f} ± {avg_results['accuracy_std']:.4f}")
        print(f"  F1-Score: {avg_results['f1_mean']:.4f} ± {avg_results['f1_std']:.4f}")
        print(f"  Precision: {avg_results['precision_mean']:.4f} ± {avg_results['precision_std']:.4f}")
        print(f"  Recall: {avg_results['recall_mean']:.4f} ± {avg_results['recall_std']:.4f}")
    
    # Save results
    results_df = pd.DataFrame(results).T
    results_df.to_csv(output_dir / "neural_network_results.csv")
    
    # Find best model
    best_model = max(results.keys(), key=lambda k: results[k]['f1_mean'])
    best_f1 = results[best_model]['f1_mean']
    
    print(f"\n🏆 BEST NEURAL NETWORK: {best_model}")
    print(f"   F1-Score: {best_f1:.4f}")
    
    return results


def main():
    """Main training function."""
    project_root = Path(".")
    features_csv = project_root / "data" / "processed" / "features_clean.csv"
    output_dir = project_root / "reports" / "neural_networks"
    
    if not features_csv.exists():
        print(f"Features file not found: {features_csv}")
        return
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"🚀 GPU Training Available: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("💻 Using CPU training (consider GPU for faster training)")
    
    # Set number of workers for multi-core data loading
    torch.set_num_threads(min(torch.get_num_threads(), 8))
    
    # Train neural networks
    results = train_neural_networks(features_csv, output_dir)
    
    print(f"\n=== NEURAL NETWORK TRAINING COMPLETE ===")
    print(f"Models saved to: {output_dir}")
    print("\nCompare with previous best (Tuned_GB): F1-Score 0.4338")


if __name__ == "__main__":
    main()
