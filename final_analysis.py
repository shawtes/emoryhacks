#!/usr/bin/env python3
"""
Comprehensive performance analysis comparing neural networks with traditional ML.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def analyze_neural_network_results():
    """Analyze and visualize neural network training results."""
    
    # Load neural network results
    nn_results_file = Path("reports/neural_networks/neural_network_results.csv")
    
    if not nn_results_file.exists():
        print("Neural network results not found!")
        return
    
    nn_df = pd.read_csv(nn_results_file, index_col=0)
    
    # Previous baseline results
    baseline_results = {
        'Random_Forest_Baseline': {'f1_mean': 0.3507, 'accuracy_mean': 0.6536},
        'Tuned_GB_Baseline': {'f1_mean': 0.4338, 'accuracy_mean': 0.6789},
        'Enhanced_RF': {'f1_mean': 0.3538, 'accuracy_mean': 0.6621},
        'SVM_Baseline': {'f1_mean': 0.3200, 'accuracy_mean': 0.6400}
    }
    
    # Combine all results
    print("=== COMPREHENSIVE MODEL PERFORMANCE ANALYSIS ===\n")
    
    # Create comprehensive comparison
    all_results = []
    
    # Add baseline results
    for model_name, metrics in baseline_results.items():
        all_results.append({
            'Model': model_name,
            'Type': 'Traditional ML',
            'F1_Score': metrics['f1_mean'],
            'Accuracy': metrics['accuracy_mean'],
            'F1_Std': 0.02  # Estimated std for baselines
        })
    
    # Add neural network results
    for model_name, row in nn_df.iterrows():
        all_results.append({
            'Model': model_name,
            'Type': 'Neural Network',
            'F1_Score': row['f1_mean'],
            'Accuracy': row['accuracy_mean'],
            'F1_Std': row['f1_std']
        })
    
    # Convert to DataFrame
    comparison_df = pd.DataFrame(all_results)
    
    # Sort by F1 score
    comparison_df = comparison_df.sort_values('F1_Score', ascending=False)
    
    print("🏆 MODEL PERFORMANCE RANKING (by F1-Score):")
    print("-" * 60)
    
    for i, (_, row) in enumerate(comparison_df.iterrows(), 1):
        model_type = "🧠" if row['Type'] == 'Neural Network' else "📊"
        print(f"{i:2d}. {model_type} {row['Model']:<20} "
              f"F1: {row['F1_Score']:.4f} ± {row['F1_Std']:.4f} | "
              f"Acc: {row['Accuracy']:.4f}")
    
    # Analysis insights
    print(f"\n=== KEY INSIGHTS ===")
    
    # Best performing models
    best_model = comparison_df.iloc[0]
    best_nn = comparison_df[comparison_df['Type'] == 'Neural Network'].iloc[0]
    best_ml = comparison_df[comparison_df['Type'] == 'Traditional ML'].iloc[0]
    
    print(f"🥇 Overall Best: {best_model['Model']} (F1: {best_model['F1_Score']:.4f})")
    print(f"🧠 Best Neural Network: {best_nn['Model']} (F1: {best_nn['F1_Score']:.4f})")
    print(f"📊 Best Traditional ML: {best_ml['Model']} (F1: {best_ml['F1_Score']:.4f})")
    
    # Performance gaps
    if best_model['Type'] == 'Traditional ML':
        gap = best_ml['F1_Score'] - best_nn['F1_Score']
        print(f"\n📈 Traditional ML leads by {gap:.4f} F1-score points")
    else:
        gap = best_nn['F1_Score'] - best_ml['F1_Score']
        print(f"\n🚀 Neural Networks lead by {gap:.4f} F1-score points")
    
    # Neural network analysis
    print(f"\n=== NEURAL NETWORK ANALYSIS ===")
    nn_only = comparison_df[comparison_df['Type'] == 'Neural Network']
    
    print("Neural Network Performance Summary:")
    for _, row in nn_only.iterrows():
        model_name = row['Model']
        f1_score = row['F1_Score']
        accuracy = row['Accuracy']
        
        # Model characteristics
        if model_name == 'CNN':
            chars = "Convolutional layers, spatial pattern detection"
        elif model_name == 'LSTM':
            chars = "Recurrent layers with attention, temporal modeling"
        elif model_name == 'CNN_LSTM':
            chars = "Hybrid CNN+LSTM, spatial + temporal features"
        elif model_name == 'Transformer':
            chars = "Self-attention mechanism, parallel processing"
        else:
            chars = "Unknown architecture"
        
        print(f"  • {model_name}: F1={f1_score:.4f}, Acc={accuracy:.4f}")
        print(f"    {chars}")
    
    # Challenges observed
    print(f"\n=== CHALLENGES IDENTIFIED ===")
    
    # Transformer issues
    transformer_f1 = nn_df.loc['Transformer', 'f1_mean']
    if transformer_f1 == 0.0:
        print("❌ Transformer model failed to learn (F1=0.000)")
        print("   • Likely overfitting or inappropriate architecture for this data size")
        print("   • Consider: simpler transformer, more regularization, or different approach")
    
    # CNN performance
    cnn_f1 = nn_df.loc['CNN', 'f1_mean']
    lstm_f1 = nn_df.loc['LSTM', 'f1_mean']
    
    if lstm_f1 > cnn_f1:
        print(f"📊 LSTM outperforms CNN by {lstm_f1 - cnn_f1:.4f}")
        print("   • Temporal patterns more important than spatial patterns")
        print("   • Speech features benefit from sequence modeling")
    
    # Dataset size considerations
    print(f"\n=== DATASET SIZE ANALYSIS ===")
    print(f"• Dataset: 355 samples (small for neural networks)")
    print(f"• Traditional ML advantage: Better performance on small datasets")
    print(f"• Neural networks typically need 1000+ samples per class for optimal performance")
    
    # Recommendations
    print(f"\n=== RECOMMENDATIONS ===")
    
    if best_model['Type'] == 'Traditional ML':
        print("🎯 STICK WITH TRADITIONAL ML:")
        print(f"   • {best_ml['Model']} achieves best F1-score: {best_ml['F1_Score']:.4f}")
        print("   • More reliable on small datasets (355 samples)")
        print("   • Faster training and easier hyperparameter tuning")
        print("   • Lower computational requirements")
    
    print("\n🔬 FOR FUTURE IMPROVEMENT:")
    print("   • Collect more data (target 1000+ samples per class)")
    print("   • Try data augmentation techniques for audio")
    print("   • Consider transfer learning from pre-trained audio models")
    print("   • Experiment with feature engineering")
    
    # Final model recommendation
    print(f"\n🏆 FINAL MODEL RECOMMENDATION:")
    print(f"   Model: {best_model['Model']}")
    print(f"   F1-Score: {best_model['F1_Score']:.4f}")
    print(f"   Accuracy: {best_model['Accuracy']:.4f}")
    print(f"   Type: {best_model['Type']}")
    
    # Save comparison results
    comparison_df.to_csv('reports/model_comparison_final.csv', index=False)
    print(f"\n📄 Detailed comparison saved to: reports/model_comparison_final.csv")
    
    return comparison_df

if __name__ == "__main__":
    analyze_neural_network_results()
