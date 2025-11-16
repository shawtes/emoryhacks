# Automated Seeding Plan (Option A)

Use a Node script with the Firebase Admin SDK to seed the demo clinic, doctor, patient, and audio metadata.

## 1. Setup
- Install dependencies at repo root (already done): `npm install firebase-admin`.
- Generate a Firebase service account key (Console → Project Settings → Service accounts → Generate new private key).
- Save the JSON (e.g., `C:/keys/voicevitals-admin.json`).

## 2. Minimal Admin SDK Bootstrapping
Example snippet (`scripts/adminSetup.js`):
```js
const admin = require('firebase-admin');
const serviceAccount = require('path/to/serviceAccountKey.json');

admin.initializeApp({
  credential: admin.credential.cert(serviceAccount),
  storageBucket: 'voicevitals-7167d.firebasestorage.app',
});

module.exports = {
  db: admin.firestore(),
  auth: admin.auth(),
  bucket: admin.storage().bucket(),
};
```

## 3. Run the bundled seeder
- Set credentials (PowerShell):
  ```powershell
  $env:GOOGLE_APPLICATION_CREDENTIALS="C:\keys\voicevitals-admin.json"
  ```
- Execute the script:
  ```powershell
  npm run seed:demo
  ```
- Script actions:
  1. Upsert `clinics/CLINIC_DEMO` (active clinic).
  2. Create doctor Auth user (`doctor.demo@voicevital.health` / `DoctorDemo123!`), set custom claims, and write `doctors/{uid}`.
  3. Create/activate patient (`patient.demo@voicevital.health` / `PatientDemo123!`, ID `PT_DEMO_001`) tied to the clinic/doctor.
  4. Upload a sample `.webm` file to `audio/patients/PT_DEMO_001/demoRecording01.webm` and store Firestore metadata.

## 4. Verify data
- Firestore console:
  - `clinics/CLINIC_DEMO`
  - `doctors/{uid}`
  - `patients/PT_DEMO_001`
  - `patientAudio/PT_DEMO_001/recordings/demoRecording01`
- Storage console:
  - `audio/patients/PT_DEMO_001/demoRecording01.webm`

## 5. Next steps
- Use the demo credentials in the React app to walk through signup/login flows.
- Extend the script if you need multiple clinics/doctors/patients or reset functionality.

