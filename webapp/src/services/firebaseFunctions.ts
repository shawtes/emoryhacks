import {
  connectFunctionsEmulator,
  getFunctions,
  httpsCallable,
  type Functions,
} from 'firebase/functions'
import { getFirebaseClients, type Persona } from '../lib/firebase'

const useEmulators = import.meta.env.VITE_USE_FIREBASE_EMULATORS === 'true'
const emulatorHost = import.meta.env.VITE_FIREBASE_EMULATOR_HOST ?? '127.0.0.1'
const emulatorPort = Number(import.meta.env.VITE_FIREBASE_FUNCTIONS_EMULATOR_PORT ?? '5001')

const functionsCache: Partial<Record<Persona, Functions>> = {}

const getFunctionsClient = async (persona: Persona) => {
  if (!functionsCache[persona]) {
    const { app } = await getFirebaseClients(persona)
    const functions = getFunctions(app)
    if (useEmulators) {
      connectFunctionsEmulator(functions, emulatorHost, emulatorPort)
    }
    functionsCache[persona] = functions
  }
  return functionsCache[persona]!
}

export interface RegisterDoctorPayload {
  clinicId: string
  email: string
  password: string
  profile: {
    fullName: string
    specialties?: string[]
    phone?: string | null
  }
}

export const callRegisterDoctor = async (payload: RegisterDoctorPayload) => {
  const functions = await getFunctionsClient('doctor')
  const callable = httpsCallable<RegisterDoctorPayload, { doctorId: string }>(
    functions,
    'registerDoctor',
  )
  const response = await callable(payload)
  return response.data
}

export interface CreatePatientStubPayload {
  patientId?: string
  patient?: Record<string, unknown>
}

export const callCreatePatientStub = async (payload: CreatePatientStubPayload = {}) => {
  const functions = await getFunctionsClient('doctor')
  const callable = httpsCallable<CreatePatientStubPayload, { patientId: string }>(
    functions,
    'createPatientStub',
  )
  const response = await callable(payload)
  return response.data
}

export interface ActivatePatientPayload {
  patientId: string
  email: string
  password: string
}

export const callActivatePatient = async (payload: ActivatePatientPayload) => {
  const functions = await getFunctionsClient('patient')
  const callable = httpsCallable<ActivatePatientPayload, { patientId: string; authUid: string }>(
    functions,
    'activatePatient',
  )
  const response = await callable(payload)
  return response.data
}

export interface IssueUploadTokenPayload {
  recordingId?: string
  doctorId?: string
  mimeType?: string
  expiresInSeconds?: number
}

export interface IssueUploadTokenResponse {
  recordingId: string
  storagePath: string
  uploadUrl: string | null
  expiresInSeconds: number
  metadata: {
    clinicId: string | null
    doctorId: string | null
    patientId: string
    mimeType: string
  }
}

export const callIssueUploadToken = async (payload: IssueUploadTokenPayload = {}) => {
  const functions = await getFunctionsClient('patient')
  const callable = httpsCallable<IssueUploadTokenPayload, IssueUploadTokenResponse>(
    functions,
    'issueUploadToken',
  )
  const response = await callable(payload)
  return response.data
}

export interface ValidateAudioMetadataPayload {
  patientId?: string
  recordingId: string
  metadata?: {
    doctorId?: string | null
    clinicId?: string | null
    recordedAt?: string
    durationSeconds?: number
    notes?: string
    analysisStatus?: string
    storagePath?: string | null
  }
}

export const callValidateAudioMetadata = async (payload: ValidateAudioMetadataPayload) => {
  const functions = await getFunctionsClient('patient')
  const callable = httpsCallable<ValidateAudioMetadataPayload, { patientId: string; recordingId: string }>(
    functions,
    'validateAudioMetadata',
  )
  const response = await callable(payload)
  return response.data
}

