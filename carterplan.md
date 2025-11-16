# Carter Plan: Firebase Patient, Doctor & Clinic System

## Objectives
- Model three Firestore user databases: `clinics`, `doctors`, and `patients`, plus an audio repository patients create and doctors consume.
- Restrict login/signup to patient and doctor personas; clinics only seed clinic IDs used during doctor onboarding.
- Guarantee immutable clinic IDs, mutable patient↔doctor relationships, and secure delivery of patient-generated audio to the assigned doctor.

## Current Status (Nov 16, 2025)
- ✅ Firebase Auth Email/Password provider enabled (anonymous disabled).
- ✅ Frontend `.env` configured with doctor + patient Firebase web app keys.
- ✅ Cloud Functions implemented (`registerDoctor`, `createPatientStub`, `activatePatient`, `issueUploadToken`, `validateAudioMetadata`).
- ✅ Firestore + Storage security rules written; Firebase Emulator Suite configured for Auth, Firestore, Functions, Storage.
- ✅ Auth/login flows, Firestore-backed dashboards, audio capture/upload UI, doctor playback, and Hosting deployment implemented with persona-aware Firebase clients.
- ✅ `npm run seed:demo` provisions `CLINIC_DEMO` with linked doctor/patient accounts plus sample audio.
- ⚠️ Pending: machine learning backend to analyze audio. Until ML is live, uploads are marked “Analyzed – Good” for demo purposes.

## Remaining Work (Post-Firebase)
- **ML audio analysis backend:** Build and deploy the inference service (FastAPI/Cloud Run/etc.) that consumes uploaded audio, runs analysis, and writes results back into each recording’s metadata.
- **Placeholder analysis status:** Until the ML backend ships, keep marking `analysisStatus` as “analyzed-good” after uploads so the UI shows a completed review loop.
- **Future enhancements:** doctor reassignment history (`patientDoctorHistory`), notifications (FCM/email), transcription pipeline, and richer reporting driven by ML outputs.

