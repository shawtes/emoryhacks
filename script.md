## VoiceVital Webapp — 2‑Minute Demo Script

> Goal: record a tight walkthrough that proves end-to-end readiness—patient and doctor portals, AI status, and backend services. Aim for ~2 minutes total; keep narration energetic and concise.

### 0:00 – 0:15 · Cold open & context
1. Start on the VoiceVital landing page (Home).  
   - Line: “This is VoiceVital—our speech biomarkers platform for dementia screening.”  
   - Hover hero CTAs to show micro-interactions and prefetch behavior.

### 0:15 – 0:45 · Patient assessment flow
2. Click **Patient Assessment** CTA (or `/patientassessment`).  
   - Highlight guided script card (“Suggested script”) and LiveStatusBadge.  
   - Line: “Patients get clinic-linked prompts plus live status, so they know their doctor sees every upload.”
3. Hit **Record** in the AudioRecorder, capture a 3–4 second clip, stop.  
   - Mention preview waveform, notes field, and optional caregiver message.
4. Click **Submit to clinic**.  
   - Narrate: “Uploads stream to Firebase + Cloud Functions; progress bar shows our upload stage.”  
   - Wait for success pill, point at Recording ID + recent uploads list.

### 0:45 – 1:15 · Patient dashboard
5. Navigate to `/patient/dashboard`.  
   - Highlight metrics (total recordings, assigned doctor, clinic contact).  
   - Scroll through “Your recent recordings” to show the new submission instantly available.
   - Line: “Streaming listeners pull real-time Firestore updates—no refresh needed.”

### 1:15 – 1:40 · Doctor experience
6. Switch to `/doctor/dashboard`.  
   - Call out patient cards, AI risk pills, and task list.  
   - Mention “explainable AI snippets and triage-ready actions populate from the same backend.”

### 1:40 – 2:00 · API + wrap-up
7. Open terminal pane (or VS Code integrated terminal) and run `./start_api.sh` or show `realtime_classify.py`.  
   - Line: “Behind the scenes, our FastAPI service and real-time classifier power the same datasets—ready for AWS deployment.”
8. Return to Home, end with CTA.  
   - Closing line: “VoiceVital marries patient-friendly UX with clinic-grade tooling—ready for pilots after this hackathon.”

### Recording tips
- Keep browser zoom at 90–100 % for full-width visuals.
- Turn on system audio if you want the recorder chime; otherwise mute for clarity.
- Use cursor highlights when clicking CTAs so judges can follow quickly.

