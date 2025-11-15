from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import soundfile as sf

from .data_ingest import list_audio_files


def load_audio(path: Path) -> Tuple[np.ndarray, int]:
    """
    Load audio as mono float32, return (waveform, sample_rate).
    """
    data, sr = sf.read(str(path), always_2d=False, dtype="float32")
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    return data, sr


def normalize_peak(audio: np.ndarray, peak: float = 0.95) -> np.ndarray:
    """
    Peak normalize waveform to the specified absolute peak.
    """
    max_val = np.max(np.abs(audio)) if audio.size > 0 else 0.0
    if max_val == 0:
        return audio
    scale = peak / max_val
    return audio * scale


def spectral_denoise(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Basic spectral noise reduction. Requires noisereduce.
    Falls back to passthrough if library not available.
    """
    try:
        import noisereduce as nr  # type: ignore
    except Exception:
        return audio
    return nr.reduce_noise(y=audio, sr=sr)


def compress_long_silences(
    audio: np.ndarray,
    sr: int,
    max_silence_s: float = 0.75,
    frame_ms: float = 30.0,
    vad_aggressiveness: int = 2,
) -> np.ndarray:
    """
    Compress long silences to a maximum duration using webrtcvad.
    """
    try:
        import webrtcvad  # type: ignore
    except Exception:
        return audio

    vad = webrtcvad.Vad(vad_aggressiveness)
    frame_size = int(sr * frame_ms / 1000.0)
    if frame_size % 2 == 1:
        frame_size += 1
    # webrtcvad expects 16-bit PCM bytes; create a helper
    pcm = np.clip(audio, -1.0, 1.0)
    pcm16 = (pcm * 32767).astype(np.int16)
    def is_voiced(frame: np.ndarray) -> bool:
        return vad.is_speech(frame.tobytes(), sr)
    voiced_mask = []
    for start in range(0, len(pcm16), frame_size):
        frame = pcm16[start : start + frame_size]
        if len(frame) < frame_size:
            frame = np.pad(frame, (0, frame_size - len(frame)), mode="constant")
        voiced_mask.append(is_voiced(frame))
    # Reconstruct with limits on consecutive unvoiced frames
    max_unvoiced = int(max_silence_s * 1000.0 / frame_ms)
    out_chunks = []
    unvoiced_run = 0
    for i, v in enumerate(voiced_mask):
        start = i * frame_size
        end = min(start + frame_size, len(audio))
        segment = audio[start:end]
        if v:
            unvoiced_run = 0
            out_chunks.append(segment)
        else:
            if unvoiced_run < max_unvoiced:
                out_chunks.append(segment)
            unvoiced_run += 1
    if not out_chunks:
        return audio
    return np.concatenate(out_chunks, axis=0)


def save_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    """
    Save audio as 16-bit PCM WAV.
    """
    audio = np.clip(audio, -1.0, 1.0)
    sf.write(str(path), audio, sr, subtype="PCM_16")


def preprocess_file(
    in_path: Path,
    out_path: Path,
    target_sr: Optional[int] = None,
    apply_denoise: bool = True,
    compress_silence: bool = False,
) -> Tuple[Path, int]:
    """
    Load -> (optional resample) -> denoise -> normalize -> (optional silence compression) -> save
    Returns (out_path, sr).
    """
    import librosa  # lazy import for speed if unused

    audio, sr = load_audio(in_path)
    if target_sr and target_sr != sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    if apply_denoise:
        audio = spectral_denoise(audio, sr)
    audio = normalize_peak(audio, peak=0.95)
    if compress_silence:
        audio = compress_long_silences(audio, sr)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_wav(out_path, audio, sr)
    return out_path, sr


def _preprocess_worker(args):
    """Worker function for multiprocessing."""
    in_path, out_path, target_sr, apply_denoise, compress_silence = args
    try:
        preprocess_file(
            in_path=in_path,
            out_path=out_path,
            target_sr=target_sr,
            apply_denoise=apply_denoise,
            compress_silence=compress_silence,
        )
        return (True, in_path, None)
    except Exception as e:
        return (False, in_path, str(e))


def preprocess_directory(
    raw_dir: Path,
    interim_dir: Path,
    target_sr: Optional[int] = None,
    apply_denoise: bool = True,
    compress_silence: bool = False,
    n_workers: Optional[int] = None,
) -> None:
    """
    Preprocess all audio files in raw_dir and save to interim_dir, maintaining directory structure.
    Uses multiprocessing with optimal worker count (4 cores, 8 threads).
    """
    from multiprocessing import Pool, cpu_count
    from .gpu_utils import get_optimal_workers
    
    audio_files = list_audio_files(raw_dir)
    if not audio_files:
        print(f"No audio files found in {raw_dir}")
        return
    
    # Get optimal worker count: use 8 logical processors (hyperthreading)
    if n_workers is None:
        _, n_threads = get_optimal_workers()
        n_workers = n_threads  # Use all 8 logical processors
    else:
        n_workers = min(n_workers, cpu_count())
    
    print(f"Found {len(audio_files)} audio files to process")
    print(f"Using {n_workers} workers for parallel processing")
    
    # Prepare arguments for workers
    worker_args = []
    for in_path in audio_files:
        rel_path = in_path.relative_to(raw_dir)
        out_path = interim_dir / rel_path
        worker_args.append((in_path, out_path, target_sr, apply_denoise, compress_silence))
    
    # Process in chunks for better progress reporting
    chunk_size = max(1, len(worker_args) // (n_workers * 4))
    processed = 0
    errors = []
    
    with Pool(processes=n_workers) as pool:
        # Process in chunks to show progress
        for i in range(0, len(worker_args), chunk_size):
            chunk = worker_args[i:i + chunk_size]
            results = pool.map(_preprocess_worker, chunk)
            
            for success, path, error in results:
                processed += 1
                if not success:
                    errors.append((path, error))
                if processed % 10 == 0 or processed == len(audio_files):
                    print(f"Processed {processed}/{len(audio_files)} files")
    
    if errors:
        print(f"\nEncountered {len(errors)} errors:")
        for path, error in errors[:10]:  # Show first 10 errors
            print(f"  {path}: {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
    
    print(f"Preprocessing complete. Processed {processed - len(errors)}/{len(audio_files)} files successfully")
    print(f"Processed files saved to {interim_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess audio files from data/raw to data/interim")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="Directory containing raw audio files (default: <project_root>/data/raw)",
    )
    parser.add_argument(
        "--interim-dir",
        type=Path,
        default=None,
        help="Directory to save preprocessed audio (default: <project_root>/data/interim)",
    )
    parser.add_argument(
        "--target-sr",
        type=int,
        default=None,
        help="Target sample rate (default: keep original)",
    )
    parser.add_argument(
        "--no-denoise",
        action="store_true",
        help="Skip noise reduction",
    )
    parser.add_argument(
        "--compress-silence",
        action="store_true",
        help="Compress long silences",
    )
    
    args = parser.parse_args()
    
    # Determine project root (parent of src/)
    project_root = Path(__file__).resolve().parents[1]
    raw_dir = args.raw_dir or (project_root / "data" / "raw")
    interim_dir = args.interim_dir or (project_root / "data" / "interim")
    
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")
    
    preprocess_directory(
        raw_dir=raw_dir,
        interim_dir=interim_dir,
        target_sr=args.target_sr,
        apply_denoise=not args.no_denoise,
        compress_silence=args.compress_silence,
    )


