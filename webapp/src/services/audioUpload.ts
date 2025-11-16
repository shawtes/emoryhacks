import { getStorage, ref, uploadBytesResumable } from 'firebase/storage'
import { getFirebaseApp } from '../lib/firebase'
import type {
  IssueUploadTokenPayload,
  ValidateAudioMetadataPayload,
} from './firebaseFunctions'
import { callIssueUploadToken, callValidateAudioMetadata } from './firebaseFunctions'

interface UploadPatientRecordingOptions {
  file: File
  patientId?: string | null
  doctorId?: string | null
  notes?: string | null
  durationSeconds?: number
  onProgress?: (percentage: number) => void
}

const uploadFileToFirebaseStorage = (
  storagePath: string,
  file: File,
  onProgress?: (percentage: number) => void,
) =>
  new Promise<void>((resolve, reject) => {
    const storage = getStorage(getFirebaseApp('patient'))
    const storageRef = ref(storage, storagePath)
    const uploadTask = uploadBytesResumable(storageRef, file, {
      contentType: file.type,
    })

    uploadTask.on(
      'state_changed',
      (snapshot) => {
        if (!onProgress) {
          return
        }
        if (snapshot.totalBytes > 0) {
          const percentage = Math.round((snapshot.bytesTransferred / snapshot.totalBytes) * 100)
          onProgress(percentage)
        }
      },
      (error) => reject(error),
      () => resolve(),
    )
  })

export interface UploadPatientRecordingResult {
  recordingId: string
  storagePath: string
}

export const uploadPatientRecording = async ({
  file,
  patientId,
  doctorId,
  notes,
  durationSeconds,
  onProgress,
}: UploadPatientRecordingOptions): Promise<UploadPatientRecordingResult> => {
  const issuePayload: IssueUploadTokenPayload = {
    mimeType: file.type,
  }
  if (doctorId) {
    issuePayload.doctorId = doctorId
  }

  const tokenResponse = await callIssueUploadToken(issuePayload)
  await uploadFileToFirebaseStorage(tokenResponse.storagePath, file, onProgress)

  const metadata: NonNullable<ValidateAudioMetadataPayload['metadata']> = {
    recordedAt: new Date().toISOString(),
    durationSeconds: durationSeconds ? Math.round(durationSeconds) : undefined,
    analysisStatus: 'analyzed-good',
    storagePath: tokenResponse.storagePath,
  }
  if (doctorId) {
    metadata.doctorId = doctorId
  }
  if (notes) {
    metadata.notes = notes
  }

  await callValidateAudioMetadata({
    patientId: patientId ?? undefined,
    recordingId: tokenResponse.recordingId,
    metadata,
  })

  return {
    recordingId: tokenResponse.recordingId,
    storagePath: tokenResponse.storagePath,
  }
}

