from pathlib import Path
from typing import Dict, Any

import pandas as pd
import numpy as np

from .preprocess import load_audio
from .features import extract_frame_features, extract_high_level_features
from .features_agg import aggregate_all


def build_features_from_metadata(
    metadata_csv: Path,
    project_root: Path,
    output_csv: Path,
) -> Path:
    """
    Build aggregated features from a metadata CSV.
    Expected CSV columns:
      - filepath: path to audio file (absolute or relative to project_root/data/raw)
      - subject_id: subject identifier
      - label: 0 (HC) or 1 (dementia)
    Produces a wide CSV of features including subject_id, label, filepath.
    """
    df = pd.read_csv(metadata_csv)
    required = {"filepath", "subject_id", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in metadata: {missing}")
    rows: list[Dict[str, Any]] = []
    raw_root = project_root / "data" / "raw"

    for _, row in df.iterrows():
        path_str = str(row["filepath"])
        path = Path(path_str)
        if not path.is_file():
            rel = raw_root / path_str
            if rel.is_file():
                path = rel
            else:
                raise FileNotFoundError(f"Audio not found: {path_str}")
        audio, sr = load_audio(path)
        frames = extract_frame_features(audio, sr)
        globals_f = extract_high_level_features(audio, sr)
        feats = aggregate_all(frames, globals_f)
        record = {"subject_id": row["subject_id"], "label": int(row["label"]), "filepath": str(path)}
        record.update(feats)
        rows.append(record)

    out_df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)
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


