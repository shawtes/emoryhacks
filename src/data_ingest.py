from pathlib import Path
from typing import List
import shutil
import zipfile
import pandas as pd
from multiprocessing import Pool, cpu_count
from functools import partial
import numpy as np


def list_audio_files(directory: Path, extensions: List[str] = None) -> List[Path]:
    """
    Recursively list audio files under directory.
    """
    if extensions is None:
        extensions = [".wav", ".mp3", ".flac", ".m4a", ".aac", ".aiff", ".au"]
    exts = set(e.lower() for e in extensions)
    files: List[Path] = []
    for p in directory.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    return sorted(files)


def ensure_dir(path: Path) -> None:
    """
    Create directory if it does not exist.
    """
    path.mkdir(parents=True, exist_ok=True)


def copy_file_to_data_dir(source_path: Path, data_dir: Path, new_name: str = None) -> Path:
    """
    Copy a file from source to data directory with optional renaming.

    Args:
        source_path: Path to the source file
        data_dir: Destination data directory
        new_name: Optional new name for the file

    Returns:
        Path to the copied file
    """
    ensure_dir(data_dir)

    if new_name:
        dest_path = data_dir / new_name
    else:
        dest_path = data_dir / source_path.name

    shutil.copy2(source_path, dest_path)
    print(f"Copied {source_path} to {dest_path}")
    return dest_path


def extract_zip_to_data_dir(zip_path: Path, data_dir: Path, extract_subdir: str = None) -> Path:
    """
    Extract a zip file to the data directory.

    Args:
        zip_path: Path to the zip file
        data_dir: Destination data directory
        extract_subdir: Optional subdirectory name for extraction

    Returns:
        Path to the extraction directory
    """
    ensure_dir(data_dir)

    if extract_subdir:
        extract_path = data_dir / extract_subdir
    else:
        extract_path = data_dir / zip_path.stem

    ensure_dir(extract_path)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    print(f"Extracted {zip_path} to {extract_path}")
    return extract_path


def find_files_in_downloads(pattern: str, downloads_dir: Path = None) -> List[Path]:
    """
    Search for files matching a pattern in the Downloads directory.

    Args:
        pattern: Pattern to search for (can include wildcards)
        downloads_dir: Downloads directory path (defaults to user's Downloads)

    Returns:
        List of matching file paths
    """
    if downloads_dir is None:
        downloads_dir = Path.home() / "Downloads"

    try:
        matching_files = list(downloads_dir.glob(pattern))
        return sorted(matching_files)
    except Exception as e:
        print(f"Error searching downloads: {e}")
        return []


def list_data_files(directory: Path, extensions: List[str] = None) -> List[Path]:
    """
    Recursively list data files under directory.
    """
    if extensions is None:
        extensions = [".csv", ".json", ".txt", ".parquet", ".h5", ".pkl", ".zip", ".tar.gz"]
    exts = set(e.lower() for e in extensions)
    files: List[Path] = []
    for p in directory.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    return sorted(files)


def process_audio_file_chunk(file_info_chunk, project_root):
    """
    Process a chunk of audio files for feature extraction.
    This function will be called by multiprocessing workers.
    
    Args:
        file_info_chunk: List of tuples (filepath, subject_id, label)
        project_root: Path to project root directory
        
    Returns:
        List of dictionaries containing features for each file
    """
    from .preprocess import load_audio
    from .features import extract_frame_features, extract_high_level_features
    from .features_agg import aggregate_all
    
    results = []
    raw_root = Path(project_root) / "data" / "raw"
    
    for filepath, subject_id, label in file_info_chunk:
        try:
            # Resolve file path
            path = Path(filepath)
            if not path.is_file():
                path = raw_root / filepath
                if not path.is_file():
                    print(f"Warning: Audio file not found: {filepath}")
                    continue
            
            # Extract features
            audio, sr = load_audio(path)
            frames = extract_frame_features(audio, sr)
            globals_f = extract_high_level_features(audio, sr)
            feats = aggregate_all(frames, globals_f)
            
            # Create record
            record = {
                "subject_id": subject_id, 
                "label": int(label), 
                "filepath": str(path)
            }
            record.update(feats)
            results.append(record)
            
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            continue
    
    return results


def create_chunks(items, chunk_size):
    """Split items into chunks of specified size."""
    for i in range(0, len(items), chunk_size):
        yield items[i:i + chunk_size]


def build_features_parallel(
    metadata_csv: Path, 
    project_root: Path, 
    output_csv: Path, 
    n_workers: int = None,
    chunk_size: int = 10
) -> Path:
    """
    Build aggregated features from metadata CSV using parallel processing.
    
    Args:
        metadata_csv: Path to metadata CSV
        project_root: Project root directory
        output_csv: Output path for features CSV
        n_workers: Number of worker processes (defaults to CPU count)
        chunk_size: Number of files per chunk
        
    Returns:
        Path to output CSV
    """
    print("Loading metadata...")
    df = pd.read_csv(metadata_csv)
    required = {"filepath", "subject_id", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in metadata: {missing}")
    
    # Prepare file info for processing
    file_info = list(zip(df['filepath'], df['subject_id'], df['label']))
    
    # Set up parallel processing
    if n_workers is None:
        n_workers = min(cpu_count(), 8)  # Cap at 8 to avoid overwhelming system
    
    print(f"Processing {len(file_info)} files using {n_workers} workers with chunks of {chunk_size}...")
    
    # Create chunks
    chunks = list(create_chunks(file_info, chunk_size))
    print(f"Created {len(chunks)} chunks")
    
    # Process chunks in parallel
    all_results = []
    
    if n_workers == 1:
        # Single-threaded for debugging
        for i, chunk in enumerate(chunks, 1):
            print(f"Processing chunk {i}/{len(chunks)}")
            results = process_audio_file_chunk(chunk, project_root)
            all_results.extend(results)
    else:
        # Multi-threaded processing
        process_chunk_partial = partial(process_audio_file_chunk, project_root=project_root)
        
        with Pool(n_workers) as pool:
            chunk_results = []
            for i, chunk in enumerate(chunks, 1):
                print(f"Submitting chunk {i}/{len(chunks)}")
                result = pool.apply_async(process_chunk_partial, (chunk,))
                chunk_results.append(result)
            
            # Collect results
            for i, result in enumerate(chunk_results, 1):
                print(f"Collecting results from chunk {i}/{len(chunk_results)}")
                try:
                    chunk_data = result.get(timeout=300)  # 5 minute timeout per chunk
                    all_results.extend(chunk_data)
                except Exception as e:
                    print(f"Error processing chunk {i}: {e}")
                    continue
    
    # Create output DataFrame
    if all_results:
        out_df = pd.DataFrame(all_results)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(output_csv, index=False)
        print(f"Successfully processed {len(all_results)} files")
        print(f"Features saved to: {output_csv}")
    else:
        raise RuntimeError("No files were successfully processed")
    
    return output_csv


def create_metadata_from_audio_dirs(data_dir: Path) -> Path:
    """
    Create metadata CSV from the organized audio directories.
    Based on the actual structure:
    - data/raw/%24RQQUVBT (1)/*.wav (dementia files - label=1)
    - data/raw/nodementia/*.wav (non-dementia files - label=0)
    
    Returns:
        Path to the created metadata.csv
    """
    ensure_dir(data_dir / "raw")
    metadata_path = data_dir / "raw" / "metadata.csv"
    raw_dir = data_dir / "raw"
    
    records = []
    
    # Process dementia files (label=1) - files in %24RQQUVBT directory
    dementia_dir = raw_dir / "%24RQQUVBT (1)"
    if dementia_dir.exists():
        audio_files = list_audio_files(dementia_dir)
        for audio_file in audio_files:
            # Extract subject ID from filename (remove extension and any suffix)
            subject_id = audio_file.stem.split('_')[0]  # Get part before first underscore
            records.append({
                'filepath': str(audio_file.relative_to(raw_dir)),  # Relative to raw directory
                'subject_id': f"dementia_{subject_id}",
                'label': 1  # dementia
            })
    
    # Process non-dementia files (label=0) - files in nodementia directory
    nodementia_dir = raw_dir / "nodementia"
    if nodementia_dir.exists():
        audio_files = list_audio_files(nodementia_dir)
        for audio_file in audio_files:
            # Extract subject ID from filename
            subject_id = audio_file.stem.split('_')[0]  # Get part before first underscore
            records.append({
                'filepath': str(audio_file.relative_to(raw_dir)),  # Relative to raw directory
                'subject_id': f"nodementia_{subject_id}",
                'label': 0  # no dementia
            })
    
    # Create DataFrame and save
    if records:
        df = pd.DataFrame(records)
        df.to_csv(metadata_path, index=False)
        
        print(f"Created metadata CSV with {len(df)} audio files:")
        print(f"  - Dementia samples: {len(df[df['label'] == 1])}")
        print(f"  - Non-dementia samples: {len(df[df['label'] == 0])}")
        print(f"  - Total subjects: {df['subject_id'].nunique()}")
        print(f"Saved to: {metadata_path}")
    else:
        print("No audio files found in the expected directories!")
        print(f"Checked: {dementia_dir} and {nodementia_dir}")
    
    return metadata_path


def organize_raw_data(data_dir: Path) -> None:
    """
    Move extracted audio files to proper raw data structure.
    """
    raw_dir = data_dir / "raw"
    ensure_dir(raw_dir)
    
    # Find extracted directories
    extracted_dirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name != "raw"]
    
    for ext_dir in extracted_dirs:
        print(f"Processing directory: {ext_dir.name}")
        
        # Move contents to raw directory with appropriate naming
        if 'dementia' in ext_dir.name.lower() and 'no' not in ext_dir.name.lower():
            target_dir = raw_dir / "dementia"
        elif 'nodementia' in ext_dir.name.lower():
            target_dir = raw_dir / "nodementia"
        else:
            target_dir = raw_dir / ext_dir.name
        
        ensure_dir(target_dir)
        
        # Move all audio files
        audio_files = list_audio_files(ext_dir)
        for audio_file in audio_files:
            shutil.move(str(audio_file), str(target_dir / audio_file.name))
        
        print(f"Moved {len(audio_files)} audio files to {target_dir}")
        
        # Remove empty directory structure
        shutil.rmtree(ext_dir)
    
    print(f"Organized data structure in: {raw_dir}")


