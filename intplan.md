# Integration Plan: Frontend ↔ Firebase Cloud Functions

## Goals
- Connect the existing React app (`webapp/`) to callable Cloud Functions with persona-aware Firebase instances.
- Support doctor and patient flows without blocking access to the public home/login/signup pages.
- Handle error scenarios gracefully (invalid clinic IDs, double activation, upload token timeouts).

## Persona-aware Firebase Clients
1. Reuse `getFirebaseClients('doctor' | 'patient')` from `src/lib/firebase.ts`.
2. Create a helper (`src/services/firebaseFunctions.ts`) that exports typed `httpsCallable` wrappers:
   - `callRegisterDoctor(data)` ― uses `doctor` app instance.
   - `callCreatePatientStub(data)` ― doctor persona.
   - `callActivatePatient(data)` ― patient persona.
   - `callIssueUploadToken(data)` and `callValidateAudioMetadata(data)` ― patient persona.
3. Ensure the helper only initializes Auth/Functions once per persona and reuses them across calls.

## Routing & Public Access
- Keep `Home`, `DoctorLogin`, `DoctorSignup`, `PatientLogin`, `PatientSignup` routes public (no auth guard).
- Persona selection (buttons/links) should just set context for which Firebase config to use; they must not require login.
- Protect only the dashboards/routes that need authenticated access with an `AuthContext` + `ProtectedRoute`.

## UI Changes by Flow
### Doctor Signup (`/doctor/signup`)
- Form fields: `clinicId`, `fullName`, `email`, `password`, optional specialties/phone.
- On submit:
  1. Call `callRegisterDoctor`.
  2. If success, display success toast + redirect to login; if error, show specific message (“Clinic inactive”, “Email already in use”).

### Doctor Dashboard Actions
- “Generate Patient ID” button triggers `callCreatePatientStub`.
- Show newly created `patientId` with copy-to-clipboard.
- Error handling: differentiate `permission-denied` (token expired) vs others.

### Patient Signup (`/patient/signup`)
- Form fields: `patientId`, `email`, `password`.
- Submit calls `callActivatePatient`.
- Handle errors: `not-found` (invalid ID), `failed-precondition` (already active), general failure.

### Patient Audio Upload
- Before uploading, call `callIssueUploadToken` to get `uploadUrl` + enforced metadata.
- Use returned URL for PUT upload; handle `expiresInSeconds` by refreshing token if needed.
- After upload, call `callValidateAudioMetadata` with `recordingId`, notes, etc.
- Surface errors when token expired or metadata mismatch.

## Error Surfacing
- Build a utility to map Firebase errors to user-friendly strings (e.g., `permission-denied` → “Please sign in again”).
- Show inline validation for required fields before hitting Cloud Functions.
- Log unexpected errors to console + optional monitoring (e.g., Sentry) later.

## Testing Plan
1. Run `firebase emulators:start --only functions,firestore,auth,storage`.
2. Configure frontend to use emulator hosts via env flag.
3. Exercise doctor signup → patient stub creation → patient activation → upload token workflow end-to-end.
4. Validate that unauthenticated visitors can still load `/`, `/doctor/login`, `/patient/signup`, etc.

## Deployment Considerations
- Once integrated, deploy functions (`firebase deploy --only functions`) and Hosting after verifying emulator tests.
- Keep `.env` synced with production Firebase web app configs; CI/CD must supply the same `VITE_FIREBASE_*` vars.

Following this plan keeps the public marketing pages reachable while adding full doctor/patient flows powered by the new Cloud Functions.


