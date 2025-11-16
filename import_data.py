#!/usr/bin/env python3
"""
Script to import data files from Downloads directory to project data directory.
"""

from pathlib import Path
from src.data_ingest import (
    find_files_in_downloads, 
    copy_file_to_data_dir, 
    extract_zip_to_data_dir,
    list_audio_files,
    list_data_files
)

def main():
    # Define paths
    project_root = Path(__file__).parent
    data_dir = project_root / "data"
    downloads_dir = Path.home() / "Downloads"
    
    print(f"Project data directory: {data_dir}")
    print(f"Downloads directory: {downloads_dir}")
    
    # Search for potential data files
    print("\n=== Searching for potential data files ===")
    
    # Look for audio files
    audio_patterns = ["*.wav", "*.mp3", "*.flac", "*.m4a", "*.aac"]
    for pattern in audio_patterns:
        files = find_files_in_downloads(pattern, downloads_dir)
        if files:
            print(f"Found audio files ({pattern}): {len(files)} files")
            for f in files[:5]:  # Show first 5
                print(f"  - {f.name}")
            if len(files) > 5:
                print(f"  ... and {len(files) - 5} more")
    
    # Look for data files
    data_patterns = ["*.zip", "*.csv", "*.json", "*.tar.gz", "*.h5", "*.parquet"]
    for pattern in data_patterns:
        files = find_files_in_downloads(pattern, downloads_dir)
        if files:
            print(f"Found data files ({pattern}): {len(files)} files")
            for f in files[:5]:  # Show first 5
                print(f"  - {f.name} ({f.stat().st_size // 1024 // 1024} MB)")
            if len(files) > 5:
                print(f"  ... and {len(files) - 5} more")
    
    # Look for any crdownload files
    crdownload_files = find_files_in_downloads("*.crdownload", downloads_dir)
    if crdownload_files:
        print(f"\nFound incomplete downloads: {len(crdownload_files)} files")
        for f in crdownload_files:
            print(f"  - {f.name} ({f.stat().st_size // 1024 // 1024} MB)")
    
    # Look for large files that might be datasets
    print("\n=== Large files (>100MB) that might be datasets ===")
    try:
        all_files = list(downloads_dir.iterdir())
        large_files = [f for f in all_files if f.is_file() and f.stat().st_size > 100 * 1024 * 1024]
        for f in sorted(large_files, key=lambda x: x.stat().st_size, reverse=True):
            size_mb = f.stat().st_size // 1024 // 1024
            print(f"  - {f.name} ({size_mb} MB)")
    except Exception as e:
        print(f"Error listing large files: {e}")
    
    # Show current data directory contents
    print(f"\n=== Current data directory contents ===")
    if data_dir.exists():
        audio_files = list_audio_files(data_dir)
        data_files = list_data_files(data_dir)
        
        print(f"Audio files: {len(audio_files)}")
        for f in audio_files[:5]:
            print(f"  - {f.name}")
        
        print(f"Data files: {len(data_files)}")
        for f in data_files[:5]:
            print(f"  - {f.name}")
    else:
        print("Data directory does not exist yet")


def copy_specific_file(filename: str):
    """Copy a specific file from Downloads to data directory"""
    project_root = Path(__file__).parent
    data_dir = project_root / "data"
    downloads_dir = Path.home() / "Downloads"
    
    source_file = downloads_dir / filename
    if source_file.exists():
        if source_file.suffix.lower() == '.zip':
            extract_zip_to_data_dir(source_file, data_dir)
        else:
            copy_file_to_data_dir(source_file, data_dir)
        print(f"Successfully processed: {filename}")
    else:
        print(f"File not found: {filename}")
        # Search for similar names
        similar_files = find_files_in_downloads(f"*{filename.split('.')[0]}*", downloads_dir)
        if similar_files:
            print("Similar files found:")
            for f in similar_files:
                print(f"  - {f.name}")


if __name__ == "__main__":
    main()
