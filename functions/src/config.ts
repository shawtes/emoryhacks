// functions/src/config.ts
export const COLLECTIONS = {
  clinics: "clinics",
  doctors: "doctors",
  patients: "patients",
  patientAudio: "patientAudio",
  patientDoctorHistory: "patientDoctorHistory",
} as const;

export const ROLES = {
  doctor: "doctor",
  patient: "patient",
  clinicAdmin: "clinicAdmin",
} as const;

export const AUDIO_BUCKET_PREFIX = "audio/patients";
