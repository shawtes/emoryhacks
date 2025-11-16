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
from typing import Optional

import numpy as np

try:
    import sounddevice as sd  # type: ignore
except Exception:
    sd = None

from emoryhacks.services.realtime_classifier import (
    DEFAULT_REALTIME_MODEL_PATH,
    Prediction,
    RealtimeClassifier,
    RealtimeClassifierError,
)


def stream_microphone(
    classifier: RealtimeClassifier,
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

    channels = 1
    blocksize = int(sample_rate * step_seconds)
    window_size = int(sample_rate * window_seconds)
    ring = np.zeros(window_size, dtype=np.float32)
    write_idx = 0
    started = False

    if verbose:
        print("Realtime classifier ready")
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
                # Predict
                try:
                    result: Prediction = classifier.predict_from_array(y64, sample_rate)
                    p_dementia = result.probability
                except RealtimeClassifierError as exc:
                    if verbose:
                        print(f"[predict] {exc}", file=sys.stderr)
                    continue
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
    classifier: RealtimeClassifier,
    audio_path: Path,
    sample_rate: int,
    threshold: float,
    labels_only: bool,
    verbose: bool,
) -> None:
    """
    One-shot prediction on an audio file (useful when mic is unavailable).
    """
    with open(audio_path, "rb") as fh:
        audio_bytes = fh.read()
    result = classifier.predict_from_bytes(audio_bytes, sample_rate)
    p_dementia = result.probability
    label = "Dementia" if p_dementia >= threshold else "All clear"
    if verbose:
        print(f"Loaded audio: {audio_path} (sr={sample_rate})")
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

    model_path = args.model_path if args.model_path.exists() else DEFAULT_REALTIME_MODEL_PATH
    try:
        classifier = RealtimeClassifier(model_path=model_path)
    except RealtimeClassifierError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)

    if args.file is not None:
        predict_file(
            classifier=classifier,
            audio_path=args.file,
            sample_rate=args.sr,
            threshold=args.threshold,
            labels_only=args.labels_only,
            verbose=args.verbose,
        )
    else:
        stream_microphone(
            classifier=classifier,
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


