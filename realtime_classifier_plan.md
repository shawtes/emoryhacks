## Plan: Wire `realtime_classify.py` into the VoiceVital assessment flow

### 1. Current state
- **Frontend flow**: `/patientassessment` records audio, uploads to Firebase Storage, then calls `predictByUrl` → FastAPI `/predict-url`, which downloads the file and runs the aggregated-feature model. Results render via `ResultsDisplay`.
- **Realtime script**: `realtime_classify.py` is a CLI utility that reads from the microphone, runs advanced biomarkers + the gradient-boosting model, and prints labels to stdout. It is not exposed as a module or API and cannot be invoked by the browser.

### 2. Goal
When a patient submits audio, the same gradient-boosting “realtime” pipeline should analyze the uploaded file, and the frontend should show those results alongside the existing Firebase/Firestore updates—without requiring manual CLI steps.

### 3. High-level architecture
1. **Refactor the realtime classifier into a reusable module**  
   - Move reusable logic from `realtime_classify.py` into `emoryhacks/services/realtime_classifier.py`.  
   - Expose functions such as `load_realtime_model()`, `predict_from_bytes(audio_bytes)`, and `predict_from_url(download_url)`.  
   - Keep the CLI wrapper (for demos) but make it import from this module.

2. **Extend the FastAPI backend**  
   - Add a new dependency-injected singleton for the realtime model so we do not reload joblib per request.  
   - Create an internal helper `analyze_firebase_audio(url: str)` that downloads the file, calls `predict_from_bytes`, and returns the structured response.  
   - Decide whether to replace `/predict-url`’s implementation with the new helper or add a parallel endpoint (e.g., `/predict-realtime`). Prefer replacing it so the frontend API remains unchanged.

3. **Queue vs. inline analysis**  
   - **Inline (simpler)**: keep the current synchronous call—upload → backend downloads → run model → respond to frontend.  
   - **Queued (scales better)**: publish a Firestore/Functions task when Storage uploads complete, run the realtime model in Cloud Functions or Cloud Run, then update Firestore with the prediction. For hackathon timelines, start with inline calls and document the queue upgrade as future work.

4. **Frontend adjustments**  
   - Reuse the existing `predictByUrl` call; once the backend serves realtime results, `ResultsDisplay` shows them automatically.  
   - Optionally add UI copy (“Realtime biomarker model”) and show streaming states if you later move to queued processing.  
   - Ensure error handling distinguishes between upload failures and analysis failures so judges see clear messaging.

5. **Firebase & permissions**  
   - The backend already downloads using Storage public URLs; if you lock buckets, move analysis into a Firebase Function triggered by `finalize` events to avoid public reads.  
   - Store classifier outputs (probability, timestamp, metadata) in Firestore under the patient recording document so the dashboard shows a historical log.

6. **Deployment checklist**  
   - Update `requirements.txt` / `emoryhacks/requirements.txt` so production environments have `soundfile`, `librosa`, and other deps required by the realtime classifier.  
   - Regenerate the Firebase Hosting build and redeploy the backend (FastAPI on ECS/Fargate or Firebase Functions, depending on your infra).  
   - Add smoke tests: a CLI test invoking the module with a known WAV and an integration test hitting `/predict-url` with a fixture URL.

### 4. Implementation steps & ownership
| Step | Description | Owner | Notes |
| --- | --- | --- | --- |
| 1 | Extract `extract_realtime_feature_dict`, `features_to_model_vector`, and model-loading logic into `services/realtime_classifier.py` | Backend | Keep CLI compatibility by importing the new module. |
| 2 | Update FastAPI startup to load the realtime model (single joblib load) | Backend | Use env var `REALTIME_MODEL_PATH`. |
| 3 | Replace `/predict-url` handler to call the new realtime helper | Backend | Return the same `PredictionResponse` shape; include confidence + message strings. |
| 4 | After upload, write prediction metadata back to Firestore (optional but recommended) | Backend | Enables dashboards to show AI verdicts without re-querying backend. |
| 5 | Frontend copy tweaks: clarify that analysis uses the realtime biomarker model; handle “analysis pending/failed” pills | Frontend | No API contract change if `/predict-url` stays the same. |
| 6 | Deploy & validate: run `npm run build`, `firebase deploy --only hosting`, and redeploy the FastAPI service | DevOps | Confirm env vars and model files exist on the server. |

### 5. Future enhancements
- **Streaming playback**: extend the API to support chunked uploads so long recordings are analyzed while still recording.  
- **Edge inference**: package the realtime classifier as a WebAssembly module (or TFLite) to run partially in-browser for instant feedback.  
- **Explainability UI**: surface the “feature importance” metrics from the classifier so doctors can see which biomarkers triggered the alert.

With this plan, every patient submission automatically runs through the realtime classifier, and the frontend keeps its existing API contract while gaining richer results.

