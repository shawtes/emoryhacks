## Data Integration Plan

### Overview
The demo now needs real-time data flowing between the Firebase backend and the React dashboards. This plan summarizes the data sources, client responsibilities, and UI changes required so both doctor and patient views reflect Firestore + Storage content (no static mocks).

### Data Sources
1. **`clinics/{clinicId}`** — clinic metadata (name, contact info).
2. **`doctors/{doctorUid}`** — doctor profile, linked `clinicId`.
3. **`patients/{patientId}`** — patient demographics, `doctorId`, `status`.
4. **`patientAudio/{patientId}/recordings/{recordingId}`** — audio metadata referencing Storage objects.
5. **Firebase Storage** — actual audio blobs at `audio/patients/{patientId}/{recordingId}.webm`.

### Doctor Dashboard
1. **Persona enforcement** — ensure Firebase app initialized with doctor config before subscribing.
2. **Profile + clinic snapshot** — `onSnapshot(doctors/{uid})`, `onSnapshot(clinics/{clinicId})`.
3. **Patient list** — query `patients` filtered by `clinicId` + `doctorId`, ordered by `updatedAt`.
4. **Recording feed** — `collectionGroup('recordings')` query filtered by clinic + doctor, ordered by `recordedAt` (limit 10).
5. **Patient ID generator** — continues to call `createPatientStub`, updating UI with returned ID/errors.
6. **UI cleanup** — remove mock cards (tasks/alerts) once Firestore data drives the page.

### Patient Dashboard
1. **Persona enforcement** — initialize patient Firebase app via `selectPersona('patient')`.
2. **Patient snapshot** — `onSnapshot(patients/{patientId})` to derive `doctorId`, `clinicId`.
3. **Doctor/clinic snapshots** — subscribe to linked docs for display.
4. **Recording history** — query `patientAudio/{patientId}/recordings` ordered by `recordedAt` (limit 10) to populate history panel + metrics.
5. **Audio capture flow**
   - Use `AudioRecorder` to capture WebAudio.
   - Call `issueUploadToken` (doctorId + mimeType) to get signed URL + recordingId.
   - Upload via signed URL with progress meter; allow doctor ID override + notes before upload.
   - Call `validateAudioMetadata` with notes, duration, `analysisStatus: 'analyzed-good'` (placeholder until ML lands).
6. **UI cleanup** — remove manual token/metadata forms; show real stats (total recordings, assigned doctor, clinic contact).

### Shared Considerations
- **Auth context** — guard routes via `ProtectedRoute` using Firebase Auth state + custom claims.
- **Local caching** — store persona selection in `localStorage` so refreshes keep the correct Firebase app.
- **Error handling** — surface Firestore/Functions errors in dashboard panels (`status-pill danger`) and reset after dismiss.
- **Seeding** — run `npm run seed:demo` to provision `CLINIC_DEMO`, doctor/patient accounts, and starter audio for testing.

