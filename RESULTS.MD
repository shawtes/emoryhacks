PS C:\Users\smtes\emoryhacks> python final_analysis.py
=== COMPREHENSIVE MODEL PERFORMANCE ANALYSIS ===

🏆 MODEL PERFORMANCE RANKING (by F1-Score):
------------------------------------------------------------
 1. 📊 Tuned_GB_Baseline    F1: 0.4338 ± 0.0200 | Acc: 0.6789
 2. 🧠 LSTM                 F1: 0.4115 ± 0.0964 | Acc: 0.5662
 3. 🧠 CNN                  F1: 0.3809 ± 0.0843 | Acc: 0.6225
 4. 📊 Enhanced_RF          F1: 0.3538 ± 0.0200 | Acc: 0.6621
 5. 📊 Random_Forest_Baseline F1: 0.3507 ± 0.0200 | Acc: 0.6536
 6. 🧠 CNN_LSTM             F1: 0.3476 ± 0.0527 | Acc: 0.5521
 7. 📊 SVM_Baseline         F1: 0.3200 ± 0.0200 | Acc: 0.6400
 8. 🧠 Transformer          F1: 0.0000 ± 0.0000 | Acc: 0.6310

=== KEY INSIGHTS ===
🥇 Overall Best: Tuned_GB_Baseline (F1: 0.4338)
🧠 Best Neural Network: LSTM (F1: 0.4115)
📊 Best Traditional ML: Tuned_GB_Baseline (F1: 0.4338)   

📈 Traditional ML leads by 0.0223 F1-score points        

=== NEURAL NETWORK ANALYSIS ===
Neural Network Performance Summary:
  • LSTM: F1=0.4115, Acc=0.5662
    Recurrent layers with attention, temporal modeling   
  • CNN: F1=0.3809, Acc=0.6225
    Convolutional layers, spatial pattern detection      
  • CNN_LSTM: F1=0.3476, Acc=0.5521
    Hybrid CNN+LSTM, spatial + temporal features
  • Transformer: F1=0.0000, Acc=0.6310
    Self-attention mechanism, parallel processing        

=== CHALLENGES IDENTIFIED ===
❌ Transformer model failed to learn (F1=0.000)
   • Likely overfitting or inappropriate architecture for this data size
   • Consider: simpler transformer, more regularization, or different approach
📊 LSTM outperforms CNN by 0.0306
   • Temporal patterns more important than spatial patterns
   • Speech features benefit from sequence modeling      

=== DATASET SIZE ANALYSIS ===
• Dataset: 355 samples (small for neural networks)       
• Traditional ML advantage: Better performance on small datasets
• Neural networks typically need 1000+ samples per class for optimal performance

=== RECOMMENDATIONS ===
🎯 STICK WITH TRADITIONAL ML:
   • Tuned_GB_Baseline achieves best F1-score: 0.4338    
   • More reliable on small datasets (355 samples)       
   • Faster training and easier hyperparameter tuning    
   • Lower computational requirements

🔬 FOR FUTURE IMPROVEMENT:
   • Collect more data (target 1000+ samples per class)  
   • Try data augmentation techniques for audio
   • Consider transfer learning from pre-trained audio models
   • Experiment with feature engineering

🏆 FINAL MODEL RECOMMENDATION:
   Model: Tuned_GB_Baseline
   F1-Score: 0.4338
   Accuracy: 0.6789
   Type: Traditional ML

📄 Detailed comparison saved to: reports/model_comparison_final.csv