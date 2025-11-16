#!/usr/bin/env python3
"""
Data cleaning and preprocessing utilities.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def clean_features_data(features_csv: Path, output_csv: Path = None, scale_features: bool = True) -> Path:
    """
    Clean the features dataset by handling NaN values and infinite values.
    
    Args:
        features_csv: Path to the features CSV file
        output_csv: Output path (defaults to same file with _clean suffix)
        scale_features: Whether to apply feature scaling
        
    Returns:
        Path to cleaned CSV file
    """
    if output_csv is None:
        output_csv = features_csv.parent / f"{features_csv.stem}_clean.csv"
    
    print(f"Loading features from: {features_csv}")
    df = pd.read_csv(features_csv)
    
    print(f"Original dataset shape: {df.shape}")
    
    # Identify feature columns (exclude metadata columns)
    metadata_cols = ['subject_id', 'label', 'filepath']
    feature_cols = [col for col in df.columns if col not in metadata_cols]
    
    print(f"Processing {len(feature_cols)} feature columns")
    
    # Check for issues in features
    X = df[feature_cols]
    
    # Count NaN values
    nan_counts = X.isnull().sum()
    cols_with_nan = nan_counts[nan_counts > 0]
    if len(cols_with_nan) > 0:
        print(f"Found NaN values in {len(cols_with_nan)} columns:")
        for col, count in cols_with_nan.head(10).items():
            print(f"  {col}: {count} NaNs")
        if len(cols_with_nan) > 10:
            print(f"  ... and {len(cols_with_nan) - 10} more columns")
    
    # Count infinite values
    inf_counts = np.isinf(X.select_dtypes(include=[np.number])).sum()
    cols_with_inf = inf_counts[inf_counts > 0]
    if len(cols_with_inf) > 0:
        print(f"Found infinite values in {len(cols_with_inf)} columns:")
        for col, count in cols_with_inf.head(10).items():
            print(f"  {col}: {count} infs")
    
    # Replace infinite values with NaN first
    X_clean = X.replace([np.inf, -np.inf], np.nan)
    
    # Remove columns that are entirely NaN
    entirely_nan_cols = X_clean.columns[X_clean.isnull().all()]
    if len(entirely_nan_cols) > 0:
        print(f"Removing {len(entirely_nan_cols)} entirely NaN columns: {list(entirely_nan_cols)}")
        X_clean = X_clean.drop(columns=entirely_nan_cols)
        feature_cols = [col for col in feature_cols if col not in entirely_nan_cols]
    
    # Handle remaining NaN values using median imputation
    if X_clean.isnull().sum().sum() > 0:
        print("Imputing missing values with median...")
        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(X_clean)
        X_clean = pd.DataFrame(X_imputed, columns=X_clean.columns, index=X.index)
    
    # Remove columns with zero variance (constant features)
    constant_cols = X_clean.columns[X_clean.var() == 0]
    if len(constant_cols) > 0:
        print(f"Removing {len(constant_cols)} constant columns")
        X_clean = X_clean.drop(columns=constant_cols)
        feature_cols = [col for col in feature_cols if col not in constant_cols]
    
    # Apply feature scaling if requested
    if scale_features:
        print("Applying feature scaling (standardization)...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clean)
        X_clean = pd.DataFrame(X_scaled, columns=X_clean.columns, index=X.index)
        
        # Save scaler for later use
        scaler_path = output_csv.parent / f"{output_csv.stem}_scaler.joblib"
        import joblib
        joblib.dump(scaler, scaler_path)
        print(f"Scaler saved to: {scaler_path}")
    
    # Combine cleaned features with metadata
    df_clean = pd.concat([df[metadata_cols], X_clean], axis=1)
    
    # Final check
    final_nan_count = df_clean[X_clean.columns].isnull().sum().sum()
    final_inf_count = np.isinf(df_clean[X_clean.columns].select_dtypes(include=[np.number])).sum().sum()
    
    print(f"Cleaned dataset shape: {df_clean.shape}")
    print(f"Final NaN count: {final_nan_count}")
    print(f"Final infinite count: {final_inf_count}")
    
    if scale_features:
        print(f"Features scaled - mean: {X_clean.mean().mean():.6f}, std: {X_clean.std().mean():.6f}")
    
    # Save cleaned data
    df_clean.to_csv(output_csv, index=False)
    print(f"Cleaned features saved to: {output_csv}")
    
    return output_csv


def get_dataset_summary(features_csv: Path):
    """
    Print a summary of the dataset.
    """
    df = pd.read_csv(features_csv)
    
    print(f"\n=== Dataset Summary ===")
    print(f"Total samples: {len(df)}")
    print(f"Features: {df.shape[1] - 3}")  # Exclude metadata columns
    
    # Class distribution
    class_counts = df['label'].value_counts()
    print(f"Class distribution:")
    print(f"  Non-dementia (0): {class_counts.get(0, 0)}")
    print(f"  Dementia (1): {class_counts.get(1, 0)}")
    
    # Subject distribution
    print(f"Unique subjects: {df['subject_id'].nunique()}")
    
    # Feature statistics
    metadata_cols = ['subject_id', 'label', 'filepath']
    feature_cols = [col for col in df.columns if col not in metadata_cols]
    X = df[feature_cols]
    
    print(f"Feature statistics:")
    print(f"  Mean: {X.mean().mean():.4f}")
    print(f"  Std: {X.std().mean():.4f}")
    print(f"  Min: {X.min().min():.4f}")
    print(f"  Max: {X.max().max():.4f}")


if __name__ == "__main__":
    # Example usage
    features_path = Path("data/processed/features.csv")
    if features_path.exists():
        # Clean the data
        clean_path = clean_features_data(features_path)
        
        # Show summary
        get_dataset_summary(clean_path)
    else:
        print(f"Features file not found: {features_path}")
