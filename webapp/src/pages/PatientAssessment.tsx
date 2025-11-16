import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import LiveStatusBadge from '../components/LiveStatusBadge'
import useStaggeredReveal from '../hooks/useStaggeredReveal'
import { useFirebase } from '../context/FirebaseContext'
import { useAuth } from '../context/AuthContext'
import FileUploader from '../components/FileUploader'
import { uploadPatientRecording, type UploadPatientRecordingResult } from '../services/audioUpload'
import { getFirebaseErrorMessage } from '../utils/firebaseErrors'
import { getAssessmentScript } from '../content/audioScripts'
import { getFirebaseApp } from '../lib/firebase'
import { collection, doc, getFirestore, limit, onSnapshot, orderBy, query } from 'firebase/firestore'
import { getStorage, ref, getDownloadURL } from 'firebase/storage'
import ResultsDisplay from '../components/ResultsDisplay'
import type { PredictionResult } from '../types'
import { predictByUrl } from '../services/api'
import PageLoader from '../components/PageLoader'

interface RecordingHistory {
  id: string
  recordedAt?: Date | null
  notes?: string | null
  analysisStatus?: string
}

interface PatientDocData {
  displayName?: string
  doctorId?: string
  clinicId?: string
}

interface DoctorDocData {
  profile?: {
    fullName?: string
    specialties?: string[]
  }
}

interface ClinicDocData {
  name?: string
  phone?: string
  contactEmail?: string
}

type UploadStage = 'idle' | 'uploading' | 'saving' | 'success' | 'error'

export default function PatientAssessment() {
  const { selectPersona } = useFirebase()
  const { user, persona, claims } = useAuth()
  const log = (...args: unknown[]) => {
    // eslint-disable-next-line no-console -- intentional demo logging
    console.log('[PatientAssessment]', ...args)
  }
  const [patientDoc, setPatientDoc] = useState<PatientDocData | null>(null)
  const [doctorDoc, setDoctorDoc] = useState<DoctorDocData | null>(null)
  const [clinicDoc, setClinicDoc] = useState<ClinicDocData | null>(null)
  const [recordings, setRecordings] = useState<RecordingHistory[]>([])
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [selectedFileUrl, setSelectedFileUrl] = useState<string | null>(null)
  const [notesInput, setNotesInput] = useState('')
  const [uploadStage, setUploadStage] = useState<UploadStage>('idle')
  const [uploadStatusMessage, setUploadStatusMessage] = useState<string | null>(null)
  const [uploadError, setUploadError] = useState<string | null>(null)
  const [lastRecordingId, setLastRecordingId] = useState<string | null>(null)
  const [uploadProgress, setUploadProgress] = useState<number | null>(null)
  const [predictionResult, setPredictionResult] = useState<PredictionResult | null>(null)
  const [analyzing, setAnalyzing] = useState(false)
  const patientIdClaim = (claims?.patientId as string) ?? null
  const script = getAssessmentScript('en-US')

  useStaggeredReveal()

  useEffect(() => {
    if (persona !== 'patient') {
      log('Persona mismatch, switching to patient persona', persona)
      void selectPersona('patient')
    }
  }, [persona, selectPersona])

  useEffect(() => {
    if (!patientIdClaim || persona !== 'patient') {
      return
    }
    const db = getFirestore(getFirebaseApp('patient'))
    const patientRef = doc(db, 'patients', patientIdClaim)
    const unsubscribe = onSnapshot(patientRef, (snap) => {
      log('Received patient document snapshot', snap.exists())
      setPatientDoc(snap.exists() ? (snap.data() as PatientDocData) : null)
    })
    return () => unsubscribe()
  }, [patientIdClaim, persona])

  useEffect(() => {
    if (!patientDoc) {
      return
    }
    const db = getFirestore(getFirebaseApp('patient'))
    const doctorId = patientDoc.doctorId as string | undefined
    const clinicId = patientDoc.clinicId as string | undefined

    let unsubDoctor: (() => void) | undefined
    let unsubClinic: (() => void) | undefined

    if (doctorId) {
      const doctorRef = doc(db, 'doctors', doctorId)
      unsubDoctor = onSnapshot(doctorRef, (snap) => setDoctorDoc(snap.exists() ? (snap.data() as DoctorDocData) : null))
    } else {
      setDoctorDoc(null)
    }

    if (clinicId) {
      const clinicRef = doc(db, 'clinics', clinicId)
      unsubClinic = onSnapshot(
        clinicRef,
        (snap) => setClinicDoc(snap.exists() ? (snap.data() as ClinicDocData) : null),
      )
    } else {
      setClinicDoc(null)
    }

    return () => {
      unsubDoctor?.()
      unsubClinic?.()
    }
  }, [patientDoc])

  useEffect(() => {
    if (!patientIdClaim || persona !== 'patient') {
      return
    }
    const db = getFirestore(getFirebaseApp('patient'))
    const recordingsRef = collection(db, 'patientAudio', patientIdClaim, 'recordings')
    const recordingsQuery = query(recordingsRef, orderBy('recordedAt', 'desc'), limit(5))
    const unsubscribe = onSnapshot(recordingsQuery, (snapshot) => {
      log('Fetched recent recordings', snapshot.size)
      const mapped = snapshot.docs.map((docSnap) => {
        const data = docSnap.data()
        return {
          id: docSnap.id,
          recordedAt: data.recordedAt?.toDate?.() ?? null,
          notes: data.notes ?? null,
          analysisStatus: data.analysisStatus ?? 'pending',
        }
      })
      setRecordings(mapped)
    })
    return () => unsubscribe()
  }, [patientIdClaim, persona])

  const handleFileSelect = (file: File) => {
    log('File selected', { size: file.size, type: file.type, name: file.name })
    // Only allow audio, prefer mp3
    if (!file.type.startsWith('audio/') && !file.name.match(/\.mp3$/i)) {
      setUploadError('Please select an MP3 or audio file.')
      return
    }
    setUploadError(null)
    setSelectedFile(file)
    if (selectedFileUrl) {
      URL.revokeObjectURL(selectedFileUrl)
    }
    setSelectedFileUrl(URL.createObjectURL(file))
    setUploadStatusMessage(null)
  }

  const handleUpload = async () => {
    if (!selectedFile) {
      log('Upload blocked: no file selected')
      setUploadError('Please choose an MP3 file before submitting.')
      return
    }
    log('Beginning upload', { size: selectedFile.size, patientIdClaim })
    setPredictionResult(null)
    setUploadStage('uploading')
    setUploadProgress(0)
    setUploadStatusMessage('Uploading MP3 to Firebase…')
    setUploadError(null)
    try {
      const res: UploadPatientRecordingResult = await uploadPatientRecording({
        file: selectedFile,
        patientId: patientIdClaim,
        doctorId: patientDoc?.doctorId ?? null,
        notes: notesInput || null,
        durationSeconds: undefined,
        onProgress: (percentage) => {
          log('Upload progress', percentage)
          setUploadProgress(percentage)
        },
      })
      log('Upload complete', res)
      setLastRecordingId(res.recordingId)
      setUploadStage('success')
      setUploadStatusMessage('Uploaded to Firebase and shared with your doctor. Starting analysis…')
      setNotesInput('')

      // Analyze via backend using Firebase download URL
      setAnalyzing(true)
      try {
        const storage = getStorage(getFirebaseApp('patient'))
        const url = await getDownloadURL(ref(storage, res.storagePath))
        log('Fetched storage download URL', url)
        const result = await predictByUrl(url)
        log('Prediction result received', result)
        setPredictionResult(result)
        setUploadStatusMessage('Analysis complete.')
      } catch (analysisError) {
        log('Analysis error', analysisError)
        setUploadStatusMessage('Upload complete, but analysis failed.')
        const msg = analysisError instanceof Error ? analysisError.message : 'Unknown analysis error'
        setUploadError(`Analysis error: ${msg}`)
      } finally {
        setAnalyzing(false)
      }
    } catch (error) {
      log('Upload or analysis error', error)
      setUploadStage('error')
      const message = getFirebaseErrorMessage(error)
      setUploadError(`Upload failed. ${message}`)
      setUploadStatusMessage('There was a problem uploading to Firebase.')
    } finally {
      setUploadProgress(null)
      log('Upload flow finished')
      // Keep preview and messages visible so user can see results
    }
  }

  if (!user || persona !== 'patient') {
    return <PageLoader />
  }

  return (
    <div className="patient-assessment">
      <header className="patient-dashboard__hero" data-animate>
        <div className="patient-dashboard__hero-copy">
          <p className="eyebrow">
            {clinicDoc?.name ? `Connected to ${clinicDoc.name}` : 'Patient assessment'}
          </p>
          <h1>Guided voice assessment</h1>
          <p>Follow the script, record your voice, and submit the audio directly to your care team.</p>
          <p className="patient-info-note">
            Assigned doctor: {doctorDoc?.profile?.fullName ?? 'Pending assignment'}
          </p>
          <LiveStatusBadge />
        </div>
        <div className="patient-dashboard__hero-actions">
          <Link to="/patient/dashboard" className="btn-nav-secondary">
            ← Back to dashboard
          </Link>
        </div>
      </header>

      <section className="guided-script-card" data-animate>
        <div className="script-card-eyebrow">Suggested script</div>
        <ul className="script-list">
          {script.segments.slice(0, 5).map((segment) => (
            <li key={segment.id}>{segment.text}</li>
          ))}
        </ul>
        <p className="script-hint">
          Speak clearly for about a minute. You can re-record if you need to. Your doctor receives every submission.
        </p>
      </section>

      <section className="guided-controls-card" data-animate>
        <div className="script-card-eyebrow">Upload MP3</div>
        <FileUploader onFileSelect={handleFileSelect} />
        {selectedFile && (
          <div className="audio-review-card">
            <div className="audio-review-head">
              <strong>Preview</strong>
              <span>{Math.round((selectedFile.size / (16000 * 2)) || 0)}s</span>
            </div>
            {selectedFileUrl && <audio controls src={selectedFileUrl} className="audio-player" />}
            <label className="form-field">
              <span>Notes for your doctor (optional)</span>
              <textarea
                rows={3}
                value={notesInput}
                onChange={(event) => setNotesInput(event.target.value)}
                placeholder="Describe how you felt during this recording…"
              />
            </label>
            <button className="btn-primary" onClick={handleUpload}>
              {uploadStage === 'uploading' || uploadStage === 'saving' ? 'Submitting…' : 'Submit to clinic'}
            </button>
            {analyzing && <p className="status-pill info" style={{ marginTop: '0.5rem' }}>Analyzing…</p>}
            {predictionResult && (
              <div style={{ marginTop: '1rem' }}>
                <ResultsDisplay result={predictionResult} />
              </div>
            )}
          </div>
        )}
        {uploadStatusMessage && <p className="status-pill success">{uploadStatusMessage}</p>}
        {uploadStage === 'uploading' && uploadProgress !== null && (
          <progress value={uploadProgress} max={100}>
            {uploadProgress}%
          </progress>
        )}
        {uploadError && <p className="status-pill danger">{uploadError}</p>}
        {lastRecordingId && (
          <p className="upload-details">
            Recording ID: <strong>{lastRecordingId}</strong>
          </p>
        )}
      </section>

      <section className="patient-dashboard__panel" data-animate>
        <div className="panel-header">
          <div>
            <p className="eyebrow">Recent uploads</p>
            <h2>Your last recordings</h2>
          </div>
        </div>
        {recordings.length === 0 ? (
          <p>You haven’t uploaded any audio yet. Record above to send your first submission.</p>
        ) : (
          <ul className="clinic-message-list">
            {recordings.map((recording) => (
              <li key={recording.id} className="clinic-message">
                <div>
                  <p className="clinic-message-sender">
                    {recording.recordedAt ? recording.recordedAt.toLocaleString() : 'Unknown time'}
                  </p>
                  <p className="clinic-message-snippet">
                    {recording.notes ? recording.notes : 'No notes provided.'}
                  </p>
                </div>
                <span className="status-pill">{recording.analysisStatus ?? 'pending'}</span>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  )
}

