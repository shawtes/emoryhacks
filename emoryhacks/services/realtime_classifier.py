from __future__ import annotations

import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np

try:  # pragma: no cover - compatibility shim for NumPy 2 pickles
    import numpy._core as _np_core  # type: ignore
except Exception:  # pragma: no cover
    try:
        import numpy.core as _np_core  # type: ignore

        import sys as _sys

        _sys.modules["numpy._core"] = _np_core
    except Exception:
        pass

import librosa
import tempfile

from advanced_features_extractor import (
    extract_advanced_spectral_features,
    extract_formant_features,
    extract_prosodic_features,
    extract_sound_object_features,
    extract_voice_quality_features,
)
from src.features import extract_high_level_features


DEFAULT_REALTIME_MODEL_PATH = Path(
    os.environ.get(
        "REALTIME_MODEL_PATH",
        os.environ.get(
            "MODEL_PATH",
            "/Users/sineshawmesfintesfaye/Downloads/enhanced_gb_combined_features.joblib",
        ),
    )
)


class RealtimeClassifierError(RuntimeError):
    """Raised when realtime classification cannot proceed."""


def _safe_float(value: float) -> float:
    val = float(value)
    return val if np.isfinite(val) else 0.0


def extract_realtime_feature_dict(y: np.ndarray, sr: int) -> Dict[str, float]:
    feats: Dict[str, float] = {}
    extractors = (
        extract_sound_object_features,
        extract_voice_quality_features,
        extract_prosodic_features,
        extract_formant_features,
        extract_advanced_spectral_features,
        extract_high_level_features,
    )
    for fn in extractors:
        try:
            feats.update(fn(y, sr))
        except Exception:
            continue
    clean: Dict[str, float] = {}
    for key, value in feats.items():
        try:
            clean[key] = _safe_float(value)
        except Exception:
            clean[key] = 0.0
    return clean


def features_to_model_vector(
    feature_names: List[str],
    feats: Dict[str, float],
) -> np.ndarray:
    if feature_names:
        ordered = [feats.get(name, 0.0) for name in feature_names]
    else:
        # Fallback to deterministic ordering
        ordered = [feats[key] for key in sorted(feats.keys())]
    return np.asarray(ordered, dtype=np.float32).reshape(1, -1)


def load_realtime_model(model_path: Optional[Path]) -> object:
    path = model_path or DEFAULT_REALTIME_MODEL_PATH
    resolved = Path(path)
    if not resolved.exists():
        raise RealtimeClassifierError(
            f"Realtime model not found at {resolved}. "
            "Set REALTIME_MODEL_PATH to the trained joblib file."
        )
    try:
        return joblib.load(resolved)
    except Exception as exc:  # pragma: no cover
        raise RealtimeClassifierError(f"Failed to load realtime model: {exc}") from exc


@dataclass
class Prediction:
    label: str
    probability: float


def _load_audio_from_bytes(audio_bytes: bytes, sample_rate: int) -> Tuple[np.ndarray, int]:
    try:
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=sample_rate)
        if y.size == 0:
            raise ValueError("Audio contains no data.")
        return y, sr
    except Exception:
        tmp = tempfile.NamedTemporaryFile(suffix=".webm", delete=False)
        try:
            tmp.write(audio_bytes)
            tmp.flush()
            tmp.close()
            y, sr = librosa.load(tmp.name, sr=sample_rate)
            if y.size == 0:
                raise ValueError("Audio contains no data.")
            return y, sr
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass


class RealtimeClassifier:
    """Reusable helper to run the enhanced GB realtime pipeline."""

    def __init__(self, model_path: Optional[Path] = None, model: Optional[object] = None):
        self._model = model or load_realtime_model(model_path)
        self.feature_names: List[str] = list(getattr(self._model, "feature_names", []))

    @property
    def model(self) -> object:
        if self._model is None:
            raise RealtimeClassifierError("Realtime model not loaded.")
        return self._model

    def predict_from_array(self, y: np.ndarray, sr: int) -> Prediction:
        feats = extract_realtime_feature_dict(y.astype(np.float64, copy=False), sr)
        vector = features_to_model_vector(self.feature_names, feats)
        model = self.model
        try:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(vector)[0]
                dementia_prob = float(proba[1] if proba.shape[0] >= 2 else proba[-1])
            else:
                dementia_prob = float(model.predict(vector)[0])
        except Exception as exc:
            raise RealtimeClassifierError(f"Prediction failed: {exc}") from exc
        label = "dementia" if dementia_prob >= 0.5 else "no_dementia"
        return Prediction(label=label, probability=dementia_prob)

    def predict_from_bytes(self, audio_bytes: bytes, sample_rate: int = 22050) -> Prediction:
        try:
            y, sr = _load_audio_from_bytes(audio_bytes, sample_rate)
        except Exception as exc:
            raise RealtimeClassifierError(f"Audio decoding failed: {exc}") from exc
        return self.predict_from_array(y, sr)


__all__ = [
    "DEFAULT_REALTIME_MODEL_PATH",
    "RealtimeClassifier",
    "RealtimeClassifierError",
    "Prediction",
    "extract_realtime_feature_dict",
    "features_to_model_vector",
    "load_realtime_model",
]

