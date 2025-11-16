#!/usr/bin/env node
/**
 * Seed demo clinic, doctor, patient, and audio metadata in Firestore/Storage.
 *
 * Usage:
 *   export GOOGLE_APPLICATION_CREDENTIALS=/path/to/serviceAccountKey.json
 *   node scripts/seedDemoData.js
 *
 * Optional environment overrides:
 *   SEED_CLINIC_ID, SEED_DOCTOR_EMAIL, SEED_DOCTOR_PASSWORD,
 *   SEED_PATIENT_ID, SEED_PATIENT_EMAIL, SEED_PATIENT_PASSWORD,
 *   SEED_RECORDING_ID
 */

const { readFile } = require('node:fs/promises')
const { initializeApp, applicationDefault } = require('firebase-admin/app')
const { getFirestore, FieldValue } = require('firebase-admin/firestore')
const { getAuth } = require('firebase-admin/auth')
const { getStorage } = require('firebase-admin/storage')

const CLINIC_ID = process.env.SEED_CLINIC_ID ?? 'CLINIC_DEMO'
const DOCTOR_EMAIL = process.env.SEED_DOCTOR_EMAIL ?? 'doctor.demo@voicevital.health'
const DOCTOR_PASSWORD = process.env.SEED_DOCTOR_PASSWORD ?? 'DoctorDemo123!'
const DOCTOR_NAME = process.env.SEED_DOCTOR_NAME ?? 'Dr. Jamie Rivera'

const PATIENT_ID = process.env.SEED_PATIENT_ID ?? 'PT_DEMO_001'
const PATIENT_EMAIL = process.env.SEED_PATIENT_EMAIL ?? 'patient.demo@voicevital.health'
const PATIENT_PASSWORD = process.env.SEED_PATIENT_PASSWORD ?? 'PatientDemo123!'
const PATIENT_NAME = process.env.SEED_PATIENT_NAME ?? 'Maya Demo'

const RECORDING_ID = process.env.SEED_RECORDING_ID ?? 'demoRecording01'

const app = initializeApp({
  credential: applicationDefault(),
  storageBucket: process.env.FIREBASE_STORAGE_BUCKET ?? 'voicevitals-7167d.firebasestorage.app',
})

const db = getFirestore(app)
const auth = getAuth(app)
const bucket = getStorage(app).bucket()

const log = (message, extra) => {
  if (extra) {
    console.log(message, extra)
  } else {
    console.log(message)
  }
}

async function upsertClinic() {
  const clinicRef = db.doc(`clinics/${CLINIC_ID}`)
  await clinicRef.set(
    {
      name: 'VoiceVital Demo Clinic',
      contactEmail: 'ops@voicevital.health',
      phone: '+1-555-0100',
      isActive: true,
      updatedAt: FieldValue.serverTimestamp(),
      createdAt: FieldValue.serverTimestamp(),
    },
    { merge: true },
  )
  log(`✅ Clinic ${CLINIC_ID} ready`)
}

async function ensureDoctor() {
  let user
  try {
    user = await auth.getUserByEmail(DOCTOR_EMAIL)
    log(`ℹ️ Doctor user already exists: ${user.uid}`)
  } catch (error) {
    if (error.code === 'auth/user-not-found') {
      user = await auth.createUser({
        email: DOCTOR_EMAIL,
        password: DOCTOR_PASSWORD,
        displayName: DOCTOR_NAME,
      })
      log(`✅ Created doctor user: ${user.uid}`)
    } else {
      throw error
    }
  }

  await auth.setCustomUserClaims(user.uid, { role: 'doctor', clinicId: CLINIC_ID })

  await db.doc(`doctors/${user.uid}`).set(
    {
      authUid: user.uid,
      clinicId: CLINIC_ID,
      status: 'active',
      profile: {
        fullName: DOCTOR_NAME,
        specialties: ['Neurology', 'Geriatrics'],
        phone: '+1-555-0200',
      },
      updatedAt: FieldValue.serverTimestamp(),
      createdAt: FieldValue.serverTimestamp(),
    },
    { merge: true },
  )

  return user.uid
}

async function ensurePatient(doctorUid) {
  const patientRef = db.doc(`patients/${PATIENT_ID}`)

  await patientRef.set(
    {
      clinicId: CLINIC_ID,
      doctorId: doctorUid,
      status: 'active',
      displayName: PATIENT_NAME,
      createdByDoctorUid: doctorUid,
      createdAt: FieldValue.serverTimestamp(),
      updatedAt: FieldValue.serverTimestamp(),
    },
    { merge: true },
  )

  let patientUser
  try {
    patientUser = await auth.getUserByEmail(PATIENT_EMAIL)
    log(`ℹ️ Patient user already exists: ${patientUser.uid}`)
  } catch (error) {
    if (error.code === 'auth/user-not-found') {
      patientUser = await auth.createUser({
        email: PATIENT_EMAIL,
        password: PATIENT_PASSWORD,
        displayName: PATIENT_NAME,
      })
      log(`✅ Created patient user: ${patientUser.uid}`)
    } else {
      throw error
    }
  }

  await auth.setCustomUserClaims(patientUser.uid, {
    role: 'patient',
    clinicId: CLINIC_ID,
    patientId: PATIENT_ID,
    doctorId: doctorUid,
  })

  await patientRef.set(
    {
      authUid: patientUser.uid,
      email: PATIENT_EMAIL,
      activatedAt: FieldValue.serverTimestamp(),
      status: 'active',
    },
    { merge: true },
  )

  return patientUser.uid
}

async function seedPatientAudio(doctorUid) {
  const storagePath = `audio/patients/${PATIENT_ID}/${RECORDING_ID}.webm`
  const file = bucket.file(storagePath)

  const demoAudio = Buffer.from('RIFFdemoWEBMVOICEVITAL', 'utf-8')
  await file.save(demoAudio, {
    contentType: 'audio/webm',
    metadata: { clinicId: CLINIC_ID, patientId: PATIENT_ID, doctorId: doctorUid },
  })

  const recordingRef = db
    .collection('patientAudio')
    .doc(PATIENT_ID)
    .collection('recordings')
    .doc(RECORDING_ID)

  await recordingRef.set({
    clinicId: CLINIC_ID,
    doctorId: doctorUid,
    patientId: PATIENT_ID,
    storagePath,
    recordedAt: FieldValue.serverTimestamp(),
    notes: 'Seeded demo recording',
    analysisStatus: 'pendingReview',
    createdAt: FieldValue.serverTimestamp(),
  })

  log(`✅ Seeded demo audio ${recordingRef.path}`)
}

async function main() {
  if (!process.env.GOOGLE_APPLICATION_CREDENTIALS) {
    console.warn('⚠️  GOOGLE_APPLICATION_CREDENTIALS is not set. Using default application credentials.')
  } else {
    const credPath = process.env.GOOGLE_APPLICATION_CREDENTIALS
    await readFile(credPath)
    log(`🔑 Using service account credentials from ${credPath}`)
  }

  await upsertClinic()
  const doctorUid = await ensureDoctor()
  await ensurePatient(doctorUid)
  await seedPatientAudio(doctorUid)

  log('🎉 Demo data ready!')
  log(`Doctor login → ${DOCTOR_EMAIL} / ${DOCTOR_PASSWORD}`)
  log(`Patient login → ${PATIENT_EMAIL} / ${PATIENT_PASSWORD}`)
}

main().catch((error) => {
  console.error('❌ Seeding failed:', error)
  process.exit(1)
})

