#!/usr/bin/env python3
"""
Process audio data and labels, then train a model for dementia detection.
"""

from pathlib import Path
from src.data_ingest import create_metadata_from_audio_dirs, organize_raw_data, build_features_parallel
from src.run_training import run_all
from optimize_config import get_optimal_settings, set_optimization_flags, configure_librosa_for_speed


def main():
    # Optimize system settings
    print("=== SYSTEM OPTIMIZATION ===")
    settings = get_optimal_settings()
    set_optimization_flags()
    configure_librosa_for_speed()
    
    # Define paths
    project_root = Path(__file__).parent
    data_dir = project_root / "data"
    
    print(f"\n💻 Using {settings['n_workers']} workers with chunk size {settings['chunk_size']}")
    
    print("\n=== STEP 1: Organizing Raw Audio Data ===")
    try:
        organize_raw_data(data_dir)
        print("✅ Raw data organized successfully")
    except Exception as e:
        print(f"⚠️  Raw data already organized or error: {e}")
    
    print("\n=== STEP 2: Creating Metadata CSV ===")
    metadata_path = create_metadata_from_audio_dirs(data_dir)
    print("✅ Metadata CSV created")
    
    print("\n=== STEP 3: Extracting Audio Features (Parallel Processing) ===")
    features_csv = data_dir / "processed" / "features.csv"
    try:
        build_features_parallel(
            metadata_csv=metadata_path, 
            project_root=project_root, 
            output_csv=features_csv,
            n_workers=settings['n_workers'],
            chunk_size=settings['chunk_size']
        )
        print(f"✅ Features extracted and saved to {features_csv}")
    except Exception as e:
        print(f"❌ Error extracting features: {e}")
        print("Make sure required packages are installed (librosa, soundfile, etc.)")
        return
    
    print("\n=== STEP 3.5: Cleaning Features Data ===")
    try:
        from data_cleaning import clean_features_data, get_dataset_summary
        clean_features_csv = clean_features_data(features_csv)
        get_dataset_summary(clean_features_csv)
        features_csv = clean_features_csv  # Use cleaned version for training
        print("✅ Features data cleaned successfully")
    except Exception as e:
        print(f"❌ Error cleaning features: {e}")
        return
    
    print("\n=== STEP 4: Training Models ===")
    try:
        run_all(
            project_root=project_root,
            features_csv=features_csv,
            splits_dir=data_dir / "splits",
            rf_out=project_root / "reports" / "metrics" / "rf", 
            ensemble_out=project_root / "reports" / "metrics" / "ensemble",
            k_folds=5
        )
        print("✅ Models trained successfully")
    except Exception as e:
        print(f"❌ Error training models: {e}")
        return
    
    print("\n=== TRAINING COMPLETE ===")
    print("Check the following directories for results:")
    print(f"  - Features: {features_csv}")
    print(f"  - Train/Val splits: {data_dir / 'splits'}")
    print(f"  - Random Forest metrics: {project_root / 'reports' / 'metrics' / 'rf'}")
    print(f"  - Ensemble metrics: {project_root / 'reports' / 'metrics' / 'ensemble'}")


if __name__ == "__main__":
    main()
