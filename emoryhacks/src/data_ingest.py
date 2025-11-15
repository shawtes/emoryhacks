from pathlib import Path
from typing import List


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


