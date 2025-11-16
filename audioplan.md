# Audio Assessment Page Plan

## Purpose & Relationship to `PatientAssessment`
- Extend the existing `PatientAssessment` page by layering an audio-specific guidance mode that reuses the current layout, patient context, and recorder component (`AudioRecorder.tsx`).
- Preserve patient demographic/context panels while swapping the central content area with the scripted read-aloud experience.
- Ensure the plan keeps parity with current design tokens (colors, typography) while introducing premium animation that feels bespoke to the audio workflow.

## Experience Flow (Happy Path)
1. **Entry Overlay (0–3s)**  
   - Dim the existing page content and raise a glassmorphism overlay card.  
   - Animated headline slides up (`Framer Motion` or CSS keyframes) to brief the user: “We’re going to record a short reading.”  
   - CTA button “Begin Guided Script” pulses subtly to invite action.
2. **Pop-up Script Carousel (~45s total)**  
   - On start, the overlay transitions into a floating text card positioned above the waveform visual.  
   - Script is chunked into five segments (≈9s each). Each card fades/zooms in, stays visible for its segment duration, then drifts upward & fades out as the next card arrives from below.  
   - Timing tokens (delay, duration) defined in a `motion.ts` file to guarantee precise sequencing.  
   - Progress indicator dots (or a slim bar) at the bottom show where the user is in the script.
3. **Recording Controls**  
   - Record button mirrors `PatientAssessment` styling but adds a glowing ring while live.  
   - Users can restart mid-script; restarting rewinds the script animation with a quick rewind indicator.
4. **Post-Record Analysis State (~2s)**  
   - Upon stopping, freeze the waveform and reveal a top-aligned loading bar labeled `Analyzing audii…`.  
   - Bar animates left→right in 2 seconds, then morphs into a pill-shaped green card with a checkmark icon.  
   - Green card rests for 1s (“Audio captured”) before smoothly shrinking and fading to reveal the next section.
5. **Completion & Actions**  
   - Main panel thanks the user: “Thanks for recording your audio!”  
   - Two prominent buttons: `Redo recording` (ghost style) and `Submit to clinic` (solid primary).  
   - Existing audio clip preview remains accessible; playback controls sit above the buttons so users can review before deciding.  
   - On submit, call a storage helper that saves the blob into a new folder `devaudio/` (frontend dev stub + backend hook placeholder).  
   - Confirmation toast briefly appears after successful save; page returns to patient dashboard CTA or stays put based on existing UX rules.

## 45-Second Guided Script (≈145 words)
Display this copy via the animated pop-up cards:

> “Today I feel focused and ready to collaborate. My morning started with a brisk walk, followed by a balanced breakfast of oatmeal, berries, and tea. As I speak, I notice how evenly I’m breathing and how each sentence flows from the last. I’m thinking about the people I appreciate and the work we can do together to stay healthy. The room around me is calm, with soft light and a gentle hum from the vents. I’m grateful for the chance to check in and describe how I’m doing. I’ll finish by saying that I’m hopeful, curious, and paying close attention.”

- Split into 5 cards (~29 words each) so the pacing remains comfortable.  
- Offer captions/subtitles toggle for accessibility; respect `prefers-reduced-motion` by providing a static card alternative.

## Script Logistics & Localization
- Store the canonical script in `webapp/src/content/audioScripts.ts` as an array of segments to match the 5-card carousel:

```ts
export const assessmentScript = {
  locale: 'en-US',
  totalDurationMs: 45000,
  segments: [
    { id: 'intro', durationMs: 9000, text: 'Today I feel focused…' },
    // …
  ],
};
```

- Structure allows `locale` expansion later (`en-GB`, `es-ES`); future translations drop into the same schema.
- Provide a helper `getAssessmentScript(locale)` that falls back to `en-US` if a translation is missing and logs the fallback for analytics.
- The UI timeline pulls from this data at runtime, guaranteeing a single source of truth for copy, timings, and accessibility text.
- Maintain a short QA checklist per segment (word count, duration) stored alongside the script file to simplify updates.

## Visual & Motion Direction
- **Pop-up styling**: rounded glass panels with subtle drop shadows, color tokens from the patient palette (#EDF4FF / #1F4B99).  
- **Animation polish**: leverage `Framer Motion` for spring-based pop-ins; use CSS `backdrop-filter: blur(16px)` for depth.  
- **Timing**: 250ms intro, 200ms exit, 9s dwell per card, 300ms gap between cards.  
- **Loading-to-check transition**: use `clip-path` or scale transform to morph the bar into the green check pill without jarring jumps.

## State Diagram & Transitions
| State | Entry Trigger | Visible Elements | Exit Trigger |
| --- | --- | --- | --- |
| `idle` | Page mount | Standard `PatientAssessment` layout, CTA to start | User clicks “Begin Guided Script” |
| `overlayIntro` | CTA press | Glass overlay, animated headline, begin button disabled after press | Intro animation completes (~3s) |
| `scriptActive` | Intro complete | Floating script card cycling segments, progress dots | User hits record (→ `recording`) or exits (→ `idle`) |
| `recording` | Microphone granted & button pressed | Glowing record ring, waveform live, script continues | User stops recording (→ `analyzing`) or error occurs (→ `error`) |
| `analyzing` | Recording stopped successfully | Frozen waveform, “Analyzing audii” loading bar | Timer completes 2s (→ `success`) or analysis fails (→ `error`) |
| `success` | Analysis timer complete | Green check pill, thank-you message, redo/submit buttons | User selects `redo` (→ `overlayIntro`) or `submit` (→ `submitted`) |
| `submitted` | Submit action resolved | Confirmation toast, optional navigation prompt | Auto-dismiss toast; state persists until nav |
| `error` | Any failure | Error toast/banner with retry instructions | User retries (state-dependent) or cancels (→ `idle`) |

Provide a lightweight state machine diagram (PlantUML or Figma) in the design handoff so engineers can map component props to states.

## Demo-Ready Error Handling & Retry UX
- **Microphone permission denied**: show inline alert above controls with OS-specific guidance, keep CTA visible to retry permission request, and pause script progression until resolved.
- **Script asset fail** (e.g., fetch issue for future localized copies): display fallback static script card, log error, allow recording to proceed so demos don’t block.
- **Recording interruption** (browser tab focus loss, device unplug): pause script, surface modal with “Resume” and “Discard” options; retain buffered audio if resume selected.
- **Analysis timeout**: if 2s fake analysis exceeds 3s, switch bar to orange, text “Taking longer than expected…”, then present retry + support link.
- **Submit failure** (client-side storage write): show persistent banner, keep buttons enabled, provide `Download audio` action as backup so demos still succeed.
- Every error state includes telemetry (`state`, `reason`, `timestamp`, `patientId`) to support demo logging.

## Edge Cases & Controls
- Handle interruptions (network loss, permissions revoked) with toast-style alerts layered above the script card.  
- If the user exits early, persist partial audio and last script segment index so resuming feels seamless.  
- Provide keyboard triggers (space = start/stop, R = redo) and screen-reader labels describing each step.  
- Logging: capture timestamps for start/stop, segment completion, redo usage, and final submission to inform QA.

## Storage & Folder Strategy
- For demo phase, persist audio blobs client-side via IndexedDB (e.g., `idb-keyval`) within a virtual `devaudio/` collection.  
- Expose `saveDevAudio(blob, metadata)` that writes `{patientId}/{timestamp}.webm` plus JSON metadata (duration, segment completion).  
- Each submission saved as `{patientId}_{timestamp}.webm` plus optional JSON metadata (duration, transcript).  
- Also mirror files to an in-memory cache so demos can fetch immediately without async delay.  
- Plan for future server validation: placeholder function `queueDevAudioSave(blob, metadata)` that currently routes to mock storage but can later call the real API.

## Testing Metrics & Instrumentation
- **Flow completion rate**: % of users reaching `success` after starting script; target ≥95% for demos.
- **Redo utilization**: monitor how often redo is tapped; flag >30% as potential UX confusion.
- **Error budget**: zero tolerance for unhandled errors; automated tests should assert each failure path surfaces the correct UI copy.
- **Timing fidelity**: write unit tests ensuring script segment durations sum to 45s ± 1s and analysis timer resolves within 2s ± 100ms.
- **Storage verification**: Cypress test confirms audio blob exists in IndexedDB `devaudio/` bucket after submit and survives page reload.
- Emit analytics events (`audio_script_started`, `audio_recording_stopped`, `audio_submit_success`, `audio_submit_error`) for dashboard tracking.

## Next Steps
- Audit `PatientAssessment.tsx` and `AudioRecorder.tsx` to confirm component boundaries for injecting the overlay + script timeline.  
- Prototype the script pop-up animation in Storybook or a dedicated sandbox route.  
- Align with design on exact visual specs before implementing to keep engineering iterations tight.  
- Write Cypress test plan covering recording flow, redo, submit, and `devaudio` save confirmation.


