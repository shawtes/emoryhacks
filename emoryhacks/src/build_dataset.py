from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
import numpy as np

from .preprocess import load_audio
from .features import extract_frame_features, extract_high_level_features
from .features_agg import aggregate_all
from .gpu_utils import check_cuda_available, get_optimal_workers, get_optimal_chunk_size


def _extract_features_worker(args):
    """Worker function for multiprocessing feature extraction."""
    row_dict, project_root, use_gpu = args
    try:
        path_str = row_dict["filepath"]
        path = Path(path_str)
        raw_root = project_root / "data" / "raw"
        
        if not path.is_file():
            rel = raw_root / path_str
            if rel.is_file():
                path = rel
            else:
                raise FileNotFoundError(f"Audio not found: {path_str}")
        
        audio, sr = load_audio(path)
        
        # Use GPU-accelerated features if available
        if use_gpu and check_cuda_available():
            try:
                from .features_gpu import extract_frame_features_gpu, extract_high_level_features_gpu
                frames = extract_frame_features_gpu(audio, sr, use_gpu=True)
                globals_f = extract_high_level_features_gpu(audio, sr, use_gpu=True)
            except Exception as e:
                # Fallback to CPU if GPU fails
                print(f"Warning: GPU feature extraction failed, using CPU: {e}")
                frames = extract_frame_features(audio, sr)
                globals_f = extract_high_level_features(audio, sr)
        else:
            frames = extract_frame_features(audio, sr)
            globals_f = extract_high_level_features(audio, sr)
        
        feats = aggregate_all(frames, globals_f)
        record = {
            "subject_id": row_dict["subject_id"],
            "label": int(row_dict["label"]),
            "filepath": str(path)
        }
        record.update(feats)
        return (True, record, None)
    except Exception as e:
        return (False, row_dict, str(e))


def build_features_from_metadata(
    metadata_csv: Path,
    project_root: Path,
    output_csv: Path,
    n_workers: Optional[int] = None,
    use_gpu: Optional[bool] = None,
) -> Path:
    """
    Build aggregated features from a metadata CSV with multiprocessing and GPU acceleration.
    Expected CSV columns:
      - filepath: path to audio file (absolute or relative to project_root/data/raw)
      - subject_id: subject identifier
      - label: 0 (HC) or 1 (dementia)
    Produces a wide CSV of features including subject_id, label, filepath.
    
    Args:
        metadata_csv: Path to metadata CSV file
        project_root: Project root directory
        output_csv: Output CSV path
        n_workers: Number of worker processes (default: auto-detect optimal)
        use_gpu: Whether to use GPU acceleration (default: auto-detect)
    """
    from multiprocessing import Pool, cpu_count
    
    df = pd.read_csv(metadata_csv)
    required = {"filepath", "subject_id", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in metadata: {missing}")
    
    # Determine GPU usage
    if use_gpu is None:
        use_gpu = check_cuda_available()
    
    if use_gpu:
        print("GPU acceleration enabled")
    else:
        print("Using CPU processing")
    
    # Get optimal worker count
    if n_workers is None:
        _, n_threads = get_optimal_workers()
        n_workers = n_threads  # Use all 8 logical processors
    else:
        n_workers = min(n_workers, cpu_count())
    
    print(f"Processing {len(df)} files with {n_workers} workers")
    
    # Prepare worker arguments
    worker_args = []
    for _, row in df.iterrows():
        row_dict = row.to_dict()
        worker_args.append((row_dict, project_root, use_gpu))
    
    # Process in chunks
    chunk_size = get_optimal_chunk_size(len(worker_args), n_workers, use_gpu)
    rows: list[Dict[str, Any]] = []
    errors = []
    processed = 0
    
    with Pool(processes=n_workers) as pool:
        # Process in chunks for progress reporting
        for i in range(0, len(worker_args), chunk_size):
            chunk = worker_args[i:i + chunk_size]
            results = pool.map(_extract_features_worker, chunk)
            
            for success, record, error in results:
                processed += 1
                if success:
                    rows.append(record)
                else:
                    errors.append((record.get("filepath", "unknown"), error))
                
                if processed % 10 == 0 or processed == len(df):
                    print(f"Processed {processed}/{len(df)} files")
    
    if errors:
        print(f"\nEncountered {len(errors)} errors:")
        for path, error in errors[:10]:
            print(f"  {path}: {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
    
    if not rows:
        raise RuntimeError("No features were successfully extracted")
    
    out_df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    print(f"Successfully extracted features from {len(rows)}/{len(df)} files")
    print(f"Features saved to {output_csv}")
    return output_csv


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Build aggregated audio features CSV from metadata.")
    parser.add_argument(
        "--project-root",
        type=str,
        default=str(Path(__file__).resolve().parents[1]),
        help="Project root (defaults to package root).",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default=None,
        help="Path to metadata CSV (default: <project_root>/data/raw/metadata.csv).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output features CSV (default: <project_root>/data/processed/features.csv).",
    )
    args = parser.parse_args()
    project_root = Path(args.project_root)
    metadata = Path(args.metadata) if args.metadata else project_root / "data" / "raw" / "metadata.csv"
    output = Path(args.output) if args.output else project_root / "data" / "processed" / "features.csv"
    try:
        out_path = build_features_from_metadata(metadata, project_root, output)
        print(str(out_path))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


