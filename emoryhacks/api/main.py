"""
FastAPI backend for dementia detection via speech analysis.
Research use only - not a medical device.
"""
import io
import logging
import os
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import soundfile as sf
import requests
import librosa
import sys as _sys
try:
    import numpy._core as _np_core  # type: ignore
except Exception:
    try:
        import numpy.core as _np_core  # type: ignore
        _sys.modules['numpy._core'] = _np_core  # alias for NumPy 2.x pickles
    except Exception:
        pass
from enhanced_gb_training import AdvancedGradientBoostingClassifier  # noqa: F401
from advanced_features_extractor import (
    extract_sound_object_features,
    extract_voice_quality_features,
    extract_prosodic_features,
    extract_formant_features,
    extract_advanced_spectral_features,
)
from src.features import extract_frame_features, extract_high_level_features
from src.features_agg import aggregate_all
from src.preprocess import load_audio, normalize_peak, spectral_denoise
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
# Ensure project root on path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Dementia Detection API",
    description="Research-only API for dementia screening via speech analysis",
    version="1.0.0"
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model cache
_model_cache = {}
DEFAULT_MODEL_PATH = Path(os.environ.get(
    "MODEL_PATH",
    "/Users/sineshawmesfintesfaye/Downloads/enhanced_gb_combined_features.joblib"
))


class PredictionResponse(BaseModel):
    prediction: str  # "dementia" or "no_dementia"
    probability: float
    confidence: str  # "low", "medium", "high"
    message: str


def load_model(model_path: Optional[Path] = None) -> Optional[object]:
    """Load the trained model. Falls back to a dummy model if not found."""
    if model_path and model_path.exists():
        try:
            return joblib.load(model_path)
        except Exception as e:
            logger.warning(f"Failed to load model from {model_path}: {e}")
    
    # Try to find any model in models directory
    project_root = Path(__file__).parent.parent
    models_dir = project_root / "models"
    
    if models_dir.exists():
        # Look for ensemble models first
        for pattern in ["**/voting_soft.joblib", "**/stacking.joblib", "**/rf_fold_*.joblib", "**/*.joblib"]:
            model_files = list(models_dir.glob(pattern))
            if model_files:
                try:
                    model = joblib.load(model_files[0])
                    logger.info(f"Loaded model from {model_files[0]}")
                    return model
                except Exception as e:
                    logger.warning(f"Failed to load {model_files[0]}: {e}")
                    continue
    
    logger.warning("No trained model found. Using dummy model for demonstration.")
    return None


def preprocess_audio(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    """Load and preprocess audio from bytes."""
    # Load audio from bytes
    audio_io = io.BytesIO(audio_bytes)
    try:
        audio, sr = sf.read(audio_io, always_2d=False, dtype="float32")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read audio file: {e}")
    
    # Convert to mono if stereo
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    
    # Basic preprocessing
    audio = spectral_denoise(audio, sr)
    audio = normalize_peak(audio, peak=0.95)
    
    return audio, sr


def extract_features_for_ml(audio: np.ndarray, sr: int) -> np.ndarray:
    """Extract aggregated features for ML models."""
    try:
        # Extract frame-level features
        frame_features = extract_frame_features(audio, sr)
        
        # Extract high-level features
        global_features = extract_high_level_features(audio, sr)
        
        # Aggregate to ML-ready format
        aggregated = aggregate_all(frame_features, global_features)
        
        # Convert to numpy array (need to maintain consistent feature order)
        # For now, use a simple approach - convert dict to sorted array
        feature_names = sorted(aggregated.keys())
        feature_vector = np.array([aggregated.get(name, 0.0) for name in feature_names])
        
        # Handle NaN values
        feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
        
        return feature_vector.reshape(1, -1)  # Return as 2D array for sklearn
    except Exception as e:
        logger.error(f"Feature extraction error: {e}")
        # Return a dummy feature vector if extraction fails
        # This should match the expected feature dimension from training
        return np.zeros((1, 100))  # Placeholder - adjust based on your model


def predict_with_model(model: object, features: np.ndarray) -> tuple[str, float]:
    """Make prediction using the model."""
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features)[0]
            # Assuming binary classification: [no_dementia_prob, dementia_prob]
            if proba.ndim > 0 and len(proba) >= 2:
                dementia_prob = float(proba[1])
            else:
                dementia_prob = float(proba[0]) if proba.ndim > 0 else 0.5
        elif hasattr(model, "predict"):
            pred = model.predict(features)[0]
            dementia_prob = float(pred) if isinstance(pred, (int, float)) else 0.5
        else:
            # Dummy prediction
            dementia_prob = 0.3
        
        prediction = "dementia" if dementia_prob >= 0.5 else "no_dementia"
        return prediction, dementia_prob
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        # Fallback to dummy prediction
        return "no_dementia", 0.3


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    project_root = Path(__file__).parent.parent
    models_dir = project_root / "models"
    
    # Try to load best model
    model = load_model(DEFAULT_MODEL_PATH) or load_model()
    if model is not None:
        _model_cache["model"] = model
        logger.info("Model loaded successfully")
    else:
        logger.warning("Running without trained model - predictions will be dummy")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "message": "Dementia Detection API - Research Use Only",
        "model_loaded": "model" in _model_cache
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
async def predict_audio(file: UploadFile = File(...)):
    """
    Predict dementia status from audio file.
    
    Accepts WAV, MP3, or other audio formats supported by soundfile.
    Returns prediction with probability and confidence level.
    """
    # Validate file type
    if not file.content_type or not any(
        t in file.content_type for t in ["audio", "video", "octet-stream"]
    ):
        # Allow anyway - content-type might not be set correctly
        logger.warning(f"Unexpected content type: {file.content_type}")
    
    # Read audio file
    try:
        audio_bytes = await file.read()
        if len(audio_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {e}")
    
    # Preprocess audio
    try:
        audio, sr = preprocess_audio(audio_bytes)
        if len(audio) == 0:
            raise HTTPException(status_code=400, detail="Audio file contains no data")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Audio preprocessing failed: {e}")
    
    # Extract features
    try:
        features = extract_features_for_ml(audio, sr)
    except Exception as e:
        logger.error(f"Feature extraction error: {e}")
        raise HTTPException(status_code=500, detail=f"Feature extraction failed: {e}")
    
    # Make prediction
    model = _model_cache.get("model")
    if model is None:
        # Dummy prediction for demo
        prediction = "no_dementia"
        probability = 0.3
        message = "Model not loaded - this is a dummy prediction. Please train a model first."
    else:
        try:
            prediction, prob = predict_with_model(model, features)
            probability = prob if prediction == "dementia" else (1.0 - prob)
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
    
    # Determine confidence level
    if probability < 0.4 or probability > 0.6:
        confidence = "low"
    elif probability < 0.3 or probability > 0.7:
        confidence = "medium"
    else:
        confidence = "high"
    
    # Adjust confidence based on probability
    prob_diff = abs(probability - 0.5)
    if prob_diff > 0.3:
        confidence = "high"
    elif prob_diff > 0.2:
        confidence = "medium"
    else:
        confidence = "low"
    
    message = (
        f"Prediction: {prediction.replace('_', ' ').title()}. "
        f"Probability: {probability:.1%}. "
        f"This is a research tool only and not a medical diagnosis."
    )
    
    return PredictionResponse(
        prediction=prediction,
        probability=probability,
        confidence=confidence,
        message=message
    )


# ===== Advanced (Realtime) feature path aligned to Enhanced GB model =====
def extract_realtime_feature_dict(y: np.ndarray, sr: int) -> dict[str, float]:
    feats: dict[str, float] = {}
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
    try:
        feats.update(extract_high_level_features(y, sr))
    except Exception:
        pass
    clean: dict[str, float] = {}
    for k, v in feats.items():
        try:
            val = float(v)
        except Exception:
            val = 0.0
        if not np.isfinite(val):
            val = 0.0
        clean[k] = val
    return clean


def vector_from_features(feature_names: list[str], feats: dict[str, float]) -> np.ndarray:
    return np.asarray([feats.get(n, 0.0) for n in feature_names], dtype=np.float32).reshape(1, -1)


class PredictUrlRequest(BaseModel):
    url: str


@app.post("/predict-url", response_model=PredictionResponse)
async def predict_audio_from_url(payload: PredictUrlRequest):
    """
    Predict dementia status from a downloadable URL (e.g., Firebase Storage download URL).
    Uses the advanced/realtime feature path compatible with Enhanced GB model.
    """
    url = payload.url
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.content
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download audio: {e}")
    try:
        y, sr = librosa.load(io.BytesIO(data), sr=22050)
        if y.size == 0:
            raise HTTPException(status_code=400, detail="Downloaded audio contains no data")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Audio decoding failed: {e}")
    # Build features aligned to model
    feats = extract_realtime_feature_dict(y.astype(np.float64, copy=False), sr)
    model = _model_cache.get("model")
    if model is None:
        prediction = "no_dementia"
        probability = 0.3
        message = "Model not loaded - this is a dummy prediction. Please train/load a model."
    else:
        try:
            if hasattr(model, "feature_names"):
                names = list(getattr(model, "feature_names"))
                X = vector_from_features(names, feats)
            else:
                # Fallback: best-effort vectorization
                names = sorted(feats.keys())
                X = np.asarray([feats[k] for k in names], dtype=np.float32).reshape(1, -1)
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)[0]
                dementia_prob = float(proba[1] if proba.shape[0] >= 2 else proba[-1])
            else:
                dementia_prob = float(model.predict(X)[0])
            prediction = "dementia" if dementia_prob >= 0.5 else "no_dementia"
            probability = dementia_prob if prediction == "dementia" else (1.0 - dementia_prob)
            message = f"Prediction: {prediction.replace('_',' ').title()}. Probability: {probability:.1%}."
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
    # Confidence
    prob_diff = abs(probability - 0.5)
    if prob_diff > 0.3:
        confidence = "high"
    elif prob_diff > 0.2:
        confidence = "medium"
    else:
        confidence = "low"
    return PredictionResponse(
        prediction=prediction,
        probability=probability,
        confidence=confidence,
        message=message,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

