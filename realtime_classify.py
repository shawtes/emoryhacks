#!/usr/bin/env python3
"""
Realtime audio classification using the saved Enhanced Gradient Boosting model.

- Captures audio from microphone in rolling windows
- Extracts advanced voice biomarkers + high-level features
- Aligns features to the model's expected feature_names (fills missing with 0.0)
- Streams prediction probabilities in the console
"""
import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import sys as _sys
# Compatibility shim for models pickled with NumPy 2.x expecting 'numpy._core'
try:
    import numpy._core as _np_core  # type: ignore
except Exception:
    try:
        import numpy.core as _np_core  # type: ignore
        _sys.modules['numpy._core'] = _np_core  # alias for unpickling
    except Exception:
        pass
import joblib

# Audio capture
try:
    import sounddevice as sd  # type: ignore
except Exception as e:
    sd = None  # Will check at runtime

# Audio utils
import librosa

# Make pickled class resolvable when it was saved from __main__
from enhanced_gb_training import AdvancedGradientBoostingClassifier  # noqa: F401

# Local feature extractors
from advanced_features_extractor import (
    extract_sound_object_features,
    extract_voice_quality_features,
    extract_prosodic_features,
    extract_formant_features,
    extract_advanced_spectral_features,
)
from src.features import extract_high_level_features


def extract_realtime_feature_dict(y: np.ndarray, sr: int) -> Dict[str, float]:
    """
    Compute a dictionary of scalar features from an audio buffer.
    Combines advanced biomarkers and engineered high-level features.
    """
    feats: Dict[str, float] = {}

    # Advanced biomarker features
    try:
        feats.update(extract_sound_object_features(y, sr))
    except Exception:
        pass
    try:
        feats.update(extract_voice_quality_features(y, sr))
    except Exception:
        pass
    try:
        feats.update(extract_prosodic_features(y, sr))
    except Exception:
        pass
    try:
        feats.update(extract_formant_features(y, sr))
    except Exception:
        pass
    try:
        feats.update(extract_advanced_spectral_features(y, sr))
    except Exception:
        pass

    # High-level engineered (pause/rate/HNR/CPP/flux/rolloff)
    try:
        feats.update(extract_high_level_features(y, sr))
    except Exception:
        pass

    # Ensure finite
    cleaned: Dict[str, float] = {}
    for k, v in feats.items():
        try:
            val = float(v)
        except Exception:
            val = 0.0
        if not np.isfinite(val):
            val = 0.0
        cleaned[k] = val
    return cleaned


def features_to_model_vector(
    feature_names: List[str],
    feats: Dict[str, float],
) -> np.ndarray:
    """
    Build the input vector in the exact order the model expects.
    Missing features are filled with 0.0.
    """
    values = [feats.get(name, 0.0) for name in feature_names]
    return np.asarray(values, dtype=np.float32).reshape(1, -1)


def load_model(model_path: Path):
    model = joblib.load(model_path)
    # Validate the required attributes exist
    missing = []
    if not hasattr(model, "predict_proba"):
        missing.append("predict_proba")
    if not hasattr(model, "feature_names"):
        missing.append("feature_names")
    if missing:
        raise RuntimeError(
            f"Loaded model is missing required attributes: {', '.join(missing)}"
        )
    return model


def stream_microphone(
    model_path: Path,
    device: Optional[int],
    sample_rate: int,
    window_seconds: float,
    step_seconds: float,
    gain: float,
    verbose: bool,
    threshold: float,
    labels_only: bool,
    max_seconds: Optional[float] = None,
) -> None:
    if sd is None:
        raise RuntimeError(
            "sounddevice is not installed. Please install it first: pip install sounddevice"
        )

    model = load_model(model_path)
    feature_names: List[str] = list(getattr(model, "feature_names"))

    channels = 1
    blocksize = int(sample_rate * step_seconds)
    window_size = int(sample_rate * window_seconds)
    ring = np.zeros(window_size, dtype=np.float32)
    write_idx = 0
    started = False

    if verbose:
        print(f"Loaded model: {model_path}")
        print(f"Expected features: {len(feature_names)}")
        print(f"Audio: device={device}, sr={sample_rate}, window={window_seconds}s, step={step_seconds}s")

    def audio_callback(indata, frames, time_info, status):
        nonlocal ring, write_idx, started
        if status:
            # status may include overruns/underruns; we continue
            if verbose:
                print(f"[audio] {status}", file=sys.stderr)
        x = indata[:, 0] * gain  # mono
        n = len(x)
        end = write_idx + n
        if end <= len(ring):
            ring[write_idx:end] = x
        else:
            first = len(ring) - write_idx
            ring[write_idx:] = x[:first]
            ring[: end - len(ring)] = x[first:]
        write_idx = (write_idx + n) % len(ring)
        started = True

    stream = sd.InputStream(
        device=device,
        channels=channels,
        samplerate=sample_rate,
        blocksize=blocksize,
        dtype="float32",
        callback=audio_callback,
    )

    with stream:
        if verbose:
            print("Mic stream started. Press Ctrl+C to stop.")
        try:
            last_run = 0.0
            start_time = time.time()
            while True:
                time.sleep(step_seconds)
                if not started:
                    continue
                # Assemble contiguous window ending at current write_idx
                end = write_idx
                start = (end - window_size) % window_size
                if start < end:
                    y = ring[start:end].copy()
                else:
                    y = np.concatenate([ring[start:], ring[:end]])
                # Basic conditioning: ensure float64 for librosa
                y64 = y.astype(np.float64, copy=False)
                # Feature extraction
                feats = extract_realtime_feature_dict(y64, sample_rate)
                X = features_to_model_vector(feature_names, feats)
                # Predict
                try:
                    proba = model.predict_proba(X)[0]
                    if proba.shape[0] == 2:
                        p_dementia = float(proba[1])
                    else:
                        # Fallback: if single-proba style or different ordering
                        p_dementia = float(proba[-1])
                except Exception:
                    # Some sklearn models might not implement predict_proba; fallback to decision
                    y_hat = model.predict(X)[0]
                    p_dementia = float(y_hat)
                # Print compact line (label + optional probability)
                label = "Dementia" if p_dementia >= threshold else "All clear"
                if labels_only:
                    print(label)
                else:
                    print(f"P(dementia)={p_dementia:0.3f} -> {label}")
                if max_seconds is not None and (time.time() - start_time) >= max_seconds:
                    if verbose:
                        print(f"Reached max_seconds={max_seconds:.1f}. Stopping...")
                    break
        except KeyboardInterrupt:
            if verbose:
                print("\nStopping stream...")
        finally:
            if verbose:
                print("Done.")


def predict_file(
    model_path: Path,
    audio_path: Path,
    sample_rate: int,
    threshold: float,
    labels_only: bool,
    verbose: bool,
) -> None:
    """
    One-shot prediction on an audio file (useful when mic is unavailable).
    """
    model = load_model(model_path)
    feature_names: List[str] = list(getattr(model, "feature_names"))
    y, sr = librosa.load(str(audio_path), sr=sample_rate)
    if verbose:
        print(f"Loaded model: {model_path}")
        print(f"Loaded audio: {audio_path} (sr={sr}, len={len(y)/sr:0.2f}s)")
        print(f"Expected features: {len(feature_names)}")
    feats = extract_realtime_feature_dict(y.astype(np.float64, copy=False), sr)
    X = features_to_model_vector(feature_names, feats)
    try:
        proba = model.predict_proba(X)[0]
        p_dementia = float(proba[1]) if proba.shape[0] == 2 else float(proba[-1])
    except Exception:
        p_dementia = float(model.predict(X)[0])
    label = "Dementia" if p_dementia >= threshold else "All clear"
    if labels_only:
        print(label)
    else:
        print(f"P(dementia)={p_dementia:0.3f} -> {label}")


def main():
    parser = argparse.ArgumentParser(description="Realtime audio classification with GB model")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/Users/sineshawmesfintesfaye/Downloads/enhanced_gb_combined_features.joblib"),
        help="Path to saved GB model (.joblib).",
    )
    parser.add_argument("--device", type=int, default=1, help="Input device index for microphone.")
    parser.add_argument("--sr", type=int, default=22050, help="Sample rate for capture.")
    parser.add_argument("--window-seconds", type=float, default=4.0, help="Analysis window size in seconds.")
    parser.add_argument("--step-seconds", type=float, default=1.0, help="Step size between inferences in seconds.")
    parser.add_argument("--gain", type=float, default=1.0, help="Input gain multiplier.")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for Dementia label.")
    parser.add_argument("--labels-only", action="store_true", default=True, help="Print only 'Dementia' or 'All clear'.")
    parser.add_argument("--max-seconds", type=float, default=60.0, help="Stop streaming after N seconds.")
    parser.add_argument("--file", type=Path, default=None, help="Path to an audio file for one-shot prediction.")
    args = parser.parse_args()

    # Fallback to basic features model if combined model isn't present
    model_path = args.model_path
    if not model_path.exists():
        alt = Path("reports/enhanced_models/enhanced_gb_basic_features.joblib")
        if alt.exists():
            model_path = alt
        else:
            print(
                f"Model not found at {args.model_path} or {alt}. "
                "Train it first via enhanced_gb_training.py.",
                file=sys.stderr,
            )
            sys.exit(1)

    if args.file is not None:
        predict_file(
            model_path=model_path,
            audio_path=args.file,
            sample_rate=args.sr,
            threshold=args.threshold,
            labels_only=args.labels_only,
            verbose=args.verbose,
        )
    else:
        stream_microphone(
            model_path=model_path,
            device=args.device,
            sample_rate=args.sr,
            window_seconds=args.window_seconds,
            step_seconds=args.step_seconds,
            gain=args.gain,
            verbose=args.verbose,
            threshold=args.threshold,
            labels_only=args.labels_only,
            max_seconds=args.max_seconds,
        )


if __name__ == "__main__":
    main()


