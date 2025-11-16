# Schema & Seeding Plan

## Goal
Provision realistic Firestore + Storage data so we can demo the doctor↔patient flows:
1. Seed `clinics` to generate immutable IDs.
2. Register at least one doctor linked to that clinic.
3. Create/activate a patient tied to the same clinic/doctor.
4. Ensure security rules reflect the relationship.
5. Upload patient audio metadata/storage files.
6. Confirm the doctor can query the audio for playback.

## Prerequisites
- Firebase CLI logged in (`firebase login`) and pointing to `voicevitals-7167d`.
- Emulator Suite running or deploy-ready environment.
- Frontend `.env` configured (already done).

## Step-by-Step

### 1. Seed `clinics`
Option A (Console):
1. Go to Firestore console → Start collection `clinics`.
2. Doc ID: `CLINIC_DEMO`.
3. Fields:
   - `name`: "VoiceVital Demo Clinic"
   - `contactEmail`: "ops@voicevital.health"
   - `phone`: "+1-555-0100"
   - `isActive`: true
   - `createdAt`: server timestamp.

Option B (script):
```ts
// scripts/seedClinic.ts
import { initializeApp, cert } from 'firebase-admin/app';
import { getFirestore } from 'firebase-admin/firestore';

initializeApp({ credential: cert(require('./serviceAccount.json')) });
await getFirestore().doc('clinics/CLINIC_DEMO').set({
  name: 'VoiceVital Demo Clinic',
  contactEmail: 'ops@voicevital.health',
  phone: '+1-555-0100',
  isActive: true,
  createdAt: Date.now(),
});
```

Option C (automated script):
- Set credentials: `export GOOGLE_APPLICATION_CREDENTIALS=/path/to/serviceAccount.json`.
- Run `node scripts/seedDemoData.js`.
- This script will create the clinic, doctor, patient, and demo audio in one shot (see console output for generated IDs/passwords).

### 2. Register a doctor
- Use `/doctor/signup` in the frontend with:
  - Clinic ID: `CLINIC_DEMO`
  - Email/password you control (e.g., `doctor.demo@voicevital.health`)
  - Name: “Dr. Jamie Rivera”
  - Specialties: “Neurology, Geriatrics”
- The callable `registerDoctor` will:
  - Create a Firebase Auth user (role `doctor`, `clinicId` claim).
  - Create `doctors/{uid}` doc.

### 3. Create a patient stub
- Sign in as the doctor (or rely on the current placeholder login).
- Doctor dashboard → “Generate patient ID” button.
- Save the returned `patientId` (e.g., `PT_DEMO_001`).

### 4. Activate patient
- Go to `/patient/signup`.
- Enter the generated `patientId`, email (e.g., `patient.demo@voicevital.health`), and password.
- The callable `activatePatient` creates an Auth user (role `patient`, claims `clinicId`, `patientId`), updates `patients/{patientId}`.

### 5. Upload audio test data
- In the patient dashboard, request an upload token (optionally specify the doctor ID).
- Use the returned signed URL to PUT a small `.webm` file (can be dummy audio recorded in-browser).
- Call “Validate metadata” to persist `patientAudio/{patientId}/recordings/{recordingId}` with notes.
- Check Firebase Storage for `audio/patients/{patientId}/{recordingId}.webm`.

### 6. Verify doctor access
- Sign in as the doctor.
- Query Firestore (or add a UI list) for `patientAudio/{patientId}/recordings`.
- Ensure security rules permit the doctor (same clinic & doctorId).
- Document the `recordingId` for backend playback integration later.

## Follow-up Tasks
- Build admin tooling/seed script to automate clinic creation.
- Expand dashboards to pull real Firestore data instead of static mocks.
- Add Storage download functionality for doctors (signed URLs or `getDownloadURL` with security rules).
- Connect backend (FastAPI/ML) to fetch audio metadata/output once ready.

