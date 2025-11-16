from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def extract_frame_features(
    audio: np.ndarray,
    sr: int,
    win_ms: float = 25.0,
    hop_ms: float = 12.0,
) -> Dict[str, np.ndarray]:
    """
    Extract frame-level features aligned with plan:
    - MFCC (14) + delta (14)
    - GTCC (14) + delta (14)
    - Log-energy (1)
    - Formants F1-F4 (4) [approximate per frame]
    - Fundamental frequency F0 (1)
    Returns dict of feature_name -> (T, D) arrays (or (T,) for 1-D).
    """
    import librosa

    n_fft = int(sr * win_ms / 1000.0)
    hop_length = int(sr * hop_ms / 1000.0)
    # MFCC
    mfcc = librosa.feature.mfcc(
        y=audio, sr=sr, n_mfcc=14, n_fft=n_fft, hop_length=hop_length
    ).T  # (T, 14)
    mfcc_delta = librosa.feature.delta(mfcc.T).T
    # Log-energy via RMS
    rms = librosa.feature.rms(y=audio, frame_length=n_fft, hop_length=hop_length).T
    log_energy = np.log(np.maximum(rms, 1e-8))
    # F0 (fundamental frequency)
    f0 = librosa.yin(audio, fmin=50, fmax=400, sr=sr, frame_length=n_fft, hop_length=hop_length)
    f0 = f0.reshape(-1, 1)
    # GTCC using librosa's gammatone-esque filter bank approximation via cqt fallback
    # Note: For exact GTCC, consider external implementation. Here we approximate with mfcc on gammatone-like mel.
    gtcc = librosa.feature.mfcc(
        y=audio, sr=sr, n_mfcc=14, n_fft=n_fft, hop_length=hop_length, htk=True
    ).T
    gtcc_delta = librosa.feature.delta(gtcc.T).T
    # Formants: coarse proxy with spectral peaks in LPC domain per frame is complex.
    # Placeholder: use spectral centroid/bandwidth as proxies (4 dims total).
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length).T
    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length).T
    formants_proxy = np.concatenate([centroid, bandwidth], axis=1)  # (T, 2)
    # To meet 4 dims, duplicate with roll statistics (simple proxy).
    formants_extra = np.concatenate(
        [np.roll(centroid, 1, axis=0), np.roll(bandwidth, 1, axis=0)], axis=1
    )
    formants = np.concatenate([formants_proxy, formants_extra], axis=1)  # (T, 4)
    return {
        "mfcc": mfcc,
        "mfcc_delta": mfcc_delta,
        "gtcc": gtcc,
        "gtcc_delta": gtcc_delta,
        "log_energy": log_energy,
        "f0": f0,
        "formants": formants,
    }


def stack_features_62(features: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Stack features into (T, 62) as per plan/paper:
    14 mfcc + 14 dmfcc + 14 gtcc + 14 dgtcc + 1 logE + 4 formants + 1 f0 = 62
    """
    arrs = [
        features["mfcc"],
        features["mfcc_delta"],
        features["gtcc"],
        features["gtcc_delta"],
        features["log_energy"],
        features["formants"],
        features["f0"],
    ]
    # Pad/truncate all to min T
    T = min(a.shape[0] for a in arrs)
    arrs = [a[:T] for a in arrs]
    X = np.concatenate(arrs, axis=1)
    return X


def _vad_mask(
    audio: np.ndarray,
    sr: int,
    frame_ms: float = 30.0,
    aggressiveness: int = 2,
) -> Tuple[np.ndarray, float]:
    """
    Return (voiced_mask[T], frame_duration_s).
    """
    try:
        import webrtcvad  # type: ignore
    except Exception:
        # Fallback: energy threshold on RMS
        frame_len = int(sr * frame_ms / 1000.0)
        if frame_len <= 0:
            frame_len = 1
        pad = (-len(audio)) % frame_len
        x = np.pad(audio, (0, pad))
        frames = x.reshape(-1, frame_len)
        rms = np.sqrt(np.mean(frames**2, axis=1))
        thr = np.median(rms) * 0.8
        return (rms > thr), frame_len / sr
    vad = webrtcvad.Vad(aggressiveness)
    frame_len = int(sr * frame_ms / 1000.0)
    if frame_len % 2 == 1:
        frame_len += 1
    pcm16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    T = int(np.ceil(len(pcm16) / frame_len))
    voiced = []
    for i in range(T):
        start = i * frame_len
        end = min(start + frame_len, len(pcm16))
        frame = pcm16[start:end]
        if len(frame) < frame_len:
            frame = np.pad(frame, (0, frame_len - len(frame)), mode="constant")
        voiced.append(vad.is_speech(frame.tobytes(), sr))
    return np.array(voiced, dtype=bool), frame_len / sr


def _segment_lengths(mask: np.ndarray, frame_s: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute contiguous voiced and unvoiced segment lengths (in seconds).
    """
    if len(mask) == 0:
        return np.array([]), np.array([])
    seg_lens = []
    seg_types = []
    current = mask[0]
    run = 1
    for b in mask[1:]:
        if b == current:
            run += 1
        else:
            seg_lens.append(run * frame_s)
            seg_types.append(current)
            current = b
            run = 1
    seg_lens.append(run * frame_s)
    seg_types.append(current)
    seg_lens = np.array(seg_lens, dtype=float)
    seg_types = np.array(seg_types, dtype=bool)
    voiced_lens = seg_lens[seg_types]
    unvoiced_lens = seg_lens[~seg_types]
    return voiced_lens, unvoiced_lens


def compute_pause_and_rate_features(audio: np.ndarray, sr: int) -> Dict[str, float]:
    """
    Pause statistics and speaking/articulation rate using syllable nuclei proxy.
    """
    import librosa
    from scipy.signal import find_peaks  # type: ignore

    mask, frame_s = _vad_mask(audio, sr, frame_ms=30.0, aggressiveness=2)
    voiced_lens, unvoiced_lens = _segment_lengths(mask, frame_s)
    total_dur = len(audio) / sr
    voiced_time = float(voiced_lens.sum()) if voiced_lens.size else 0.0
    unvoiced_time = float(unvoiced_lens.sum()) if unvoiced_lens.size else total_dur - voiced_time
    long_pause_thresh = 0.75
    long_pause_count = int((unvoiced_lens > long_pause_thresh).sum()) if unvoiced_lens.size else 0
    pause_count = int(unvoiced_lens.size) if unvoiced_lens.size else 0
    pause_mean = float(np.mean(unvoiced_lens)) if unvoiced_lens.size else 0.0
    pause_p90 = float(np.percentile(unvoiced_lens, 90)) if unvoiced_lens.size else 0.0
    pause_max = float(np.max(unvoiced_lens)) if unvoiced_lens.size else 0.0
    speech_pause_ratio = voiced_time / (unvoiced_time + 1e-6)
    unvoiced_fraction = unvoiced_time / (total_dur + 1e-6)
    # Syllable nuclei via RMS peaks
    hop_length = int(sr * 0.01)
    frame_length = int(sr * 0.025)
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length).flatten()
    thr = max(1e-8, float(np.percentile(rms, 60)))
    min_dist = int(0.12 * sr / hop_length)  # ~120ms
    peaks, _ = find_peaks(rms, height=thr, distance=min_dist)
    syllables = int(len(peaks))
    speaking_rate = syllables / (total_dur + 1e-6)
    articulation_rate = syllables / (voiced_time + 1e-6) if voiced_time > 0 else 0.0
    mean_run = float(np.mean(voiced_lens)) if voiced_lens.size else 0.0
    return {
        "total_duration_s": total_dur,
        "voiced_duration_s": voiced_time,
        "unvoiced_duration_s": unvoiced_time,
        "unvoiced_fraction": unvoiced_fraction,
        "speech_pause_ratio": speech_pause_ratio,
        "pause_count": pause_count,
        "long_pause_count": long_pause_count,
        "pause_mean_s": pause_mean,
        "pause_p90_s": pause_p90,
        "pause_max_s": pause_max,
        "syllable_count": syllables,
        "speaking_rate_hz": speaking_rate,
        "articulation_rate_hz": articulation_rate,
        "mean_length_of_run_s": mean_run,
    }


def compute_hnr_cpp(audio: np.ndarray, sr: int) -> Dict[str, float]:
    """
    Compute Harmonics-to-Noise Ratio (HNR) and Cepstral Peak Prominence (CPP) via parselmouth if available.
    """
    try:
        import parselmouth  # type: ignore
        snd = parselmouth.Sound(audio, sr)
        harm = parselmouth.praat.call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr = float(parselmouth.praat.call(harm, "Get mean", 0, 0))
        # CPP approximation
        cep = parselmouth.praat.call(snd, "To PowerCepstrogram", 0.01, 50, 5000)
        time = parselmouth.praat.call(cep, "Get time from index", 1)
        quefrency = parselmouth.praat.call(cep, "Get quefrency from index", 1)
        cpp = float(parselmouth.praat.call(cep, "Get peak prominence", time, quefrency))
        return {"hnr_db": hnr, "cpp_db": cpp}
    except Exception:
        return {"hnr_db": np.nan, "cpp_db": np.nan}


def compute_spectral_flux_rolloff(audio: np.ndarray, sr: int) -> Dict[str, float]:
    """
    Spectral flux and rolloff statistics.
    """
    import librosa
    n_fft = int(sr * 0.025)
    hop_length = int(sr * 0.012)
    S = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))
    dS = np.diff(S, axis=1)
    pos = np.clip(dS, a_min=0.0, a_max=None)
    flux = np.sqrt((pos**2).sum(axis=0)) / (S.shape[0] + 1e-6)
    rolloff85 = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85).flatten()
    return {
        "spectral_flux_mean": float(np.mean(flux)) if flux.size else 0.0,
        "spectral_flux_std": float(np.std(flux)) if flux.size else 0.0,
        "rolloff85_mean": float(np.mean(rolloff85)) if rolloff85.size else 0.0,
        "rolloff85_std": float(np.std(rolloff85)) if rolloff85.size else 0.0,
    }


def extract_high_level_features(audio: np.ndarray, sr: int) -> Dict[str, float]:
    """
    Bundle quick-win engineered features.
    """
    feats = {}
    feats.update(compute_pause_and_rate_features(audio, sr))
    feats.update(compute_hnr_cpp(audio, sr))
    feats.update(compute_spectral_flux_rolloff(audio, sr))
    return feats
