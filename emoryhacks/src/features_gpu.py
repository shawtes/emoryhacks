"""
GPU-accelerated feature extraction using CuPy and optimized operations.
Optimized for GeForce GTX 1660 Super.
"""
from typing import Dict, Optional
import numpy as np

try:
    import cupy as cp
    CUPY_AVAILABLE = cp.cuda.is_available()
except ImportError:
    CUPY_AVAILABLE = False
    cp = None


def extract_frame_features_gpu(
    audio: np.ndarray,
    sr: int,
    win_ms: float = 25.0,
    hop_ms: float = 12.0,
    use_gpu: Optional[bool] = None,
) -> Dict[str, np.ndarray]:
    """
    GPU-accelerated frame-level feature extraction.
    Falls back to CPU if GPU is unavailable.
    """
    if use_gpu is None:
        use_gpu = CUPY_AVAILABLE
    
    if not use_gpu or not CUPY_AVAILABLE:
        # Fallback to CPU version
        from .features import extract_frame_features
        return extract_frame_features(audio, sr, win_ms, hop_ms)
    
    # Transfer audio to GPU
    audio_gpu = cp.asarray(audio.astype(np.float32))
    
    n_fft = int(sr * win_ms / 1000.0)
    hop_length = int(sr * hop_ms / 1000.0)
    
    # Use librosa on CPU for complex operations, but accelerate with GPU where possible
    # For now, we'll use a hybrid approach: librosa for complex features, GPU for simple math
    import librosa
    
    # Compute STFT on CPU (librosa is optimized)
    S = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))
    S_gpu = cp.asarray(S)
    
    # MFCC computation - use librosa but transfer to GPU for delta computation
    mfcc_cpu = librosa.feature.mfcc(
        y=audio, sr=sr, n_mfcc=14, n_fft=n_fft, hop_length=hop_length
    ).T
    mfcc_gpu = cp.asarray(mfcc_cpu)
    
    # Delta computation on GPU (faster)
    mfcc_delta_gpu = cp.diff(mfcc_gpu, axis=0)
    mfcc_delta_gpu = cp.pad(mfcc_delta_gpu, ((1, 0), (0, 0)), mode='edge')
    mfcc_delta = cp.asnumpy(mfcc_delta_gpu)
    
    # Log-energy via RMS on GPU
    rms_cpu = librosa.feature.rms(y=audio, frame_length=n_fft, hop_length=hop_length).T
    rms_gpu = cp.asarray(rms_cpu)
    log_energy_gpu = cp.log(cp.maximum(rms_gpu, 1e-8))
    log_energy = cp.asnumpy(log_energy_gpu)
    
    # F0 on CPU (librosa.yin is complex)
    f0 = librosa.yin(audio, fmin=50, fmax=400, sr=sr, frame_length=n_fft, hop_length=hop_length)
    f0 = f0.reshape(-1, 1)
    
    # GTCC
    gtcc_cpu = librosa.feature.mfcc(
        y=audio, sr=sr, n_mfcc=14, n_fft=n_fft, hop_length=hop_length, htk=True
    ).T
    gtcc_gpu = cp.asarray(gtcc_cpu)
    gtcc_delta_gpu = cp.diff(gtcc_gpu, axis=0)
    gtcc_delta_gpu = cp.pad(gtcc_delta_gpu, ((1, 0), (0, 0)), mode='edge')
    gtcc_delta = cp.asnumpy(gtcc_delta_gpu)
    
    # Spectral features on GPU
    centroid_cpu = librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length).T
    bandwidth_cpu = librosa.feature.spectral_bandwidth(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length).T
    
    centroid_gpu = cp.asarray(centroid_cpu)
    bandwidth_gpu = cp.asarray(bandwidth_cpu)
    
    # Formants computation on GPU
    formants_proxy_gpu = cp.concatenate([centroid_gpu, bandwidth_gpu], axis=1)
    formants_extra_gpu = cp.concatenate(
        [cp.roll(centroid_gpu, 1, axis=0), cp.roll(bandwidth_gpu, 1, axis=0)], axis=1
    )
    formants_gpu = cp.concatenate([formants_proxy_gpu, formants_extra_gpu], axis=1)
    formants = cp.asnumpy(formants_gpu)
    
    # Clean up GPU memory
    del audio_gpu, S_gpu, mfcc_gpu, mfcc_delta_gpu, rms_gpu, log_energy_gpu
    del gtcc_gpu, gtcc_delta_gpu, centroid_gpu, bandwidth_gpu, formants_proxy_gpu, formants_extra_gpu, formants_gpu
    cp.get_default_memory_pool().free_all_blocks()
    
    return {
        "mfcc": mfcc_cpu,
        "mfcc_delta": mfcc_delta,
        "gtcc": gtcc_cpu,
        "gtcc_delta": gtcc_delta,
        "log_energy": log_energy,
        "f0": f0,
        "formants": formants,
    }


def extract_high_level_features_gpu(
    audio: np.ndarray,
    sr: int,
    use_gpu: Optional[bool] = None,
) -> Dict[str, float]:
    """
    GPU-accelerated high-level feature extraction.
    Falls back to CPU if GPU is unavailable.
    """
    if use_gpu is None:
        use_gpu = CUPY_AVAILABLE
    
    if not use_gpu or not CUPY_AVAILABLE:
        # Fallback to CPU version
        from .features import extract_high_level_features
        return extract_high_level_features(audio, sr)
    
    # Most high-level features still use CPU libraries (librosa, scipy, parselmouth)
    # GPU acceleration is limited here, but we can optimize array operations
    from .features import (
        compute_pause_and_rate_features,
        compute_hnr_cpp,
        compute_spectral_flux_rolloff,
    )
    
    feats = {}
    feats.update(compute_pause_and_rate_features(audio, sr))
    feats.update(compute_hnr_cpp(audio, sr))
    feats.update(compute_spectral_flux_rolloff(audio, sr))
    
    return feats


