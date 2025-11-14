from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import soundfile as sf


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


