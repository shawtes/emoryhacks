## Project plan: Speech-based dementia/Alzheimer’s detection in Python

### Goal
- Build a research-grade pipeline that ingests voice recordings and predicts dementia status (screening, not diagnosis). Emphasize early screening, subject-wise evaluation, and reproducibility. Align with findings from “Dementia Detection from Speech Using ML and DL Architectures” (Sensors, 2022).

### Data and ethics
- Dataset: DementiaBank Pitt Corpus (Cookie Theft) [request access via](https://dementia.talkbank.org/).
- Labels: dementia vs healthy control (HC). If available, capture MCI/AD subtypes for future work.
- Ethics: research-only; not a medical device. Obtain IRB/consent if adding new data. Remove PII from filenames/metadata.
- Splits: strictly subject-wise to prevent leakage (no segments from the same subject across train/val/test).

### Repository structure
```
demntia mlin/
  data/
    raw/                 # original audio (.wav/.mp3) (read-only)
    interim/             # cleaned/denoised, diarized
    processed/           # feature arrays, frame-level tensors, labels
    splits/              # json/csv of subject-wise folds
  models/
    ml/                  # RF/SVM artifacts (sklearn joblib)
    dl/                  # CNN/RNN/PRCNN checkpoints (tf/torch)
  reports/
    figures/             # confusion matrices, ROC, PR curves
    metrics/             # per-fold results, bootstrapped CIs
  src/
    config/              # YAML/OMEGACONF params
    data_ingest.py
    preprocess.py        # VAD, diarization, denoise, normalize, resample
    segment.py           # silence truncation, fixed-length chunks
    features.py          # MFCC/GTCC/delta, F0, formants, jitter, shimmer
    features_agg.py      # aggregates for ML track
    ml_train.py          # RF, SVM, trees (+search)
    dl_models.py         # CNN, (Bi-)GRU/LSTM, PRCNN
    dl_train.py
    evaluate.py          # metrics, plots, calibration, SHAP
    api.py               # FastAPI inference service
  notebooks/
    EDA_*.ipynb
    ErrorAnalysis_*.ipynb
  README.md
  requirements.txt or environment.yml
```

### Environment
- Python 3.10+
- Core libs: numpy, scipy, pandas, scikit-learn, joblib, matplotlib, seaborn
- Audio: librosa, soundfile, pydub, webrtcvad (or pyannote.audio for diarization), noisereduce
- Voice features: parselmouth (Praat) for jitter/shimmer/formants/F0
- DL: tensorflow or pytorch (choose one; paper used TF-Keras with Nadam)
- Config & utils: pyyaml/omegaconf, hydra-core, shap (interpretability)

### End-to-end pipeline
1) Ingestion
   - Validate formats; convert to 16-bit mono PCM WAV.
   - Keep original sample rate. Paper used 44.1 kHz; alternatively standardize to 16 kHz (pick one and stay consistent in all stages; document choice).
2) Preprocessing
   - Remove interviewer voice: preferred diarization (pyannote.audio); fallback: manual regions/transcript alignment or energy-based heuristics.
   - Noise reduction: spectral gating (noisereduce) targeted to background noise.
   - Amplitude normalization to a consistent loudness target.
3) Silence handling and segmentation (DL track)
   - Compress long silences >0.75 s down to 0.75 s.
   - Segment into fixed 15 s windows (with small overlap optional), save segment metadata.
4) Feature extraction
   - Frame window/hop: 25 ms / 10–12 ms.
   - Frame-level features (62 dims per paper):
     - MFCC (14), ΔMFCC (14)
     - GTCC (14), ΔGTCC (14)
     - Log-energy (1)
     - Formants F1–F4 (4)
     - Fundamental frequency F0 (1)
   - Time-(in)dependent features (for ML track; ~44 dims):
     - Jitter (5 variants), Shimmer (5 variants), F0, Formants (4), MFCC/GTCC aggregates.
   - Aggregations for ML: mean, std, median, IQR, percentiles as needed → compact 44-D set (match paper).
5) Splitting
   - Five-fold stratified cross-validation, subject-wise. Ensure segments from one subject stay in a single fold.

### Modeling
ML track (compact, strong baseline; paper best ≈87.6% accuracy with RF)
- Features: 44-D aggregated set (prosodic/voice quality + cepstral).
- Models: RandomForest (primary), SVM (RBF), REP tree, RandomTree, Logistic baseline.
- Search: cross-validated grid/random search for RF (n_estimators, max_depth, min_samples_*), SVM (C, gamma).
- Calibration: Platt/Isotonic if probabilities used.

DL track (target ≈85% with PRCNN per paper)
- Inputs: T×62 (frame-level).
- CNN-1D: 3–4 conv blocks (filters: 32→64→128; kernels: 32/18/12), max-pool, dropout 0.3; dense head; Sigmoid output.
- RNN: (Bi-)GRU/LSTM (2×128 units), time-distributed dense (64→32→16→8), flatten, dense head.
- PRCNN: parallel CNN branch + RNN branch, concatenate, dropout 0.5, dense head.
- Training: Nadam, lr = 1e-4, batch size 16–64, early stopping on val F1 (patience 10), class weighting if needed.

### Evaluation
- Metrics: accuracy, precision, recall (sensitivity), F1; emphasize recall on dementia class. Report confusion matrices, ROC-AUC/PR-AUC.
- Cross-validation: report per-fold and mean±std; bootstrap CIs over subjects.
- Leakage guard: verify no subject/segment leakage across folds.
- Model selection: prefer higher dementia recall with strong F1, then operational considerations (size/speed).

### Interpretability and analysis
- ML: permutation importance, SHAP summary and dependence plots (feature-level insights).
- DL: Grad-CAM-like saliency on feature maps (optional), ablations per feature group (prosodic vs cepstral).
- Error analysis: stratify by recording length, SNR, speaker age/sex (if available), and silence proportion.

### Inference and deployment (research)
- FastAPI microservice:
  - POST /predict accepts .wav or base64; runs preprocess → features → model.
  - Returns probability, class, and confidence; include research-only disclaimer.
- Batch CLI for offline scoring and auditing.
- On-device feasibility: export light RF model or quantized DL (future work).
- Privacy: process locally; avoid cloud storage; purge intermediates; log only non-PII aggregates.

### Reproducibility
- Deterministic seeds; save versions of data splits and configs.
- Track artifacts: models, metrics, and code hash. Consider MLflow or Weights & Biases.

### Risks and mitigations
- Small dataset: use subject-wise CV, confidence intervals, simpler ML models; avoid overfitting.
- Speaker diarization quality: verify with spot checks; maintain fallback energy-based removal.
- Domain shift (mic, room noise): augmentations (noise, reverb), robust normalization.
- Medical use risk: clearly label as research tool; not for clinical diagnosis.

### Milestones
- Week 1: Environment, repo scaffold, data access, subject-wise splits.
- Week 2: Preprocessing (VAD/diarization, denoise, normalization), feature extractors validated.
- Week 3: ML baselines (RF/SVM), CV results + report, interpretability.
- Week 4: DL models (CNN/RNN/PRCNN), CV results, compare to ML.
- Week 5: API/CLI, documentation, packaging; final report with metrics and limitations.

### Immediate next steps
1) Create environment and install dependencies.
2) Implement `preprocess.py` with VAD/denoise/normalization and tests on 5–10 files.
3) Implement `features.py` + `features_agg.py`; validate shapes and expected ranges.
4) Generate subject-wise 5-fold splits; persist in `data/splits/`.
5) Train RF baseline; target mean accuracy≈85–88% (recall prioritized).

### References (selected)
- DementiaBank Pitt Corpus: https://dementia.talkbank.org/
- Sensors (2022): Dementia Detection from Speech Using ML and DL Architectures (reported RF≈87.6%, PRCNN≈85% on compact features).


