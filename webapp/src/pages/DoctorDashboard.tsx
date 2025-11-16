import { useEffect, useMemo, useState, type FormEvent } from 'react'
import { Link } from 'react-router-dom'
import { getPrefetchEvents } from '../routes/lazyPages'
import { useFirebase } from '../context/FirebaseContext'
import { useAuth } from '../context/AuthContext'
import {
  collection,
  collectionGroup,
  doc,
  getFirestore,
  limit,
  onSnapshot,
  orderBy,
  query,
  where,
} from 'firebase/firestore'
import { getStorage, ref, getDownloadURL } from 'firebase/storage'
import { getFirebaseApp } from '../lib/firebase'
import PageLoader from '../components/PageLoader'
import { callCreatePatientStub } from '../services/firebaseFunctions'
import { getFirebaseErrorMessage } from '../utils/firebaseErrors'

interface PatientSummary {
  id: string
  displayName: string
  status?: string
  updatedAt?: Date | null
}

interface RecordingSummary {
  id: string
  patientId: string
  recordedAt?: Date | null
  notes?: string | null
  analysisStatus?: string
  storagePath?: string | null
  downloadUrl?: string | null
}

interface DoctorDocData {
  profile?: {
    fullName?: string
    specialties?: string[]
    phone?: string
  }
}

interface ClinicDocData {
  name?: string
  phone?: string
  contactEmail?: string
}

export default function DoctorDashboard() {
  const { selectPersona } = useFirebase()
  const { user, claims, persona } = useAuth()
  const [doctorInfo, setDoctorInfo] = useState<DoctorDocData | null>(null)
  const [clinicInfo, setClinicInfo] = useState<ClinicDocData | null>(null)
  const [patients, setPatients] = useState<PatientSummary[]>([])
  const [recordings, setRecordings] = useState<RecordingSummary[]>([])
  const [newPatientName, setNewPatientName] = useState('')
  const [patientStubResult, setPatientStubResult] = useState<string | null>(null)
  const [stubError, setStubError] = useState<string | null>(null)
  const [isCreatingStub, setIsCreatingStub] = useState(false)
  const [dataError, setDataError] = useState<string | null>(null)

  const clinicId = (claims?.clinicId as string) ?? null

  useEffect(() => {
    if (persona !== 'doctor') {
      void selectPersona('doctor')
    }
  }, [persona, selectPersona])

  useEffect(() => {
    if (!user || persona !== 'doctor' || !clinicId) {
      return
    }
    const app = getFirebaseApp('doctor')
    const db = getFirestore(app)
    const doctorRef = doc(db, 'doctors', user.uid)
    const clinicRef = doc(db, 'clinics', clinicId)
    const patientsQuery = query(
      collection(db, 'patients'),
      where('clinicId', '==', clinicId),
      where('doctorId', '==', user.uid),
      orderBy('updatedAt', 'desc'),
    )
    const recordingsQuery = query(
      collectionGroup(db, 'recordings'),
      where('clinicId', '==', clinicId),
      where('doctorId', '==', user.uid),
      orderBy('recordedAt', 'desc'),
      limit(10),
    )

    const unsubDoctor = onSnapshot(
      doctorRef,
      (snap) => setDoctorInfo(snap.exists() ? (snap.data() as DoctorDocData) : null),
      (error) => setDataError(error.message),
    )
    const unsubClinic = onSnapshot(
      clinicRef,
      (snap) => setClinicInfo(snap.exists() ? (snap.data() as ClinicDocData) : null),
      (error) => setDataError(error.message),
    )
    const unsubPatients = onSnapshot(
      patientsQuery,
      (snapshot) => {
        const mapped = snapshot.docs.map((docSnap) => {
          const data = docSnap.data()
          return {
            id: docSnap.id,
            displayName: data.displayName ?? data.fullName ?? docSnap.id,
            status: data.status,
            updatedAt: data.updatedAt?.toDate?.() ?? data.activatedAt?.toDate?.() ?? null,
          }
        })
        setPatients(mapped)
      },
      (error) => setDataError(error.message),
    )
    const unsubRecordings = onSnapshot(
      recordingsQuery,
      async (snapshot) => {
        const mapped = await Promise.all(
          snapshot.docs.map(async (docSnap) => {
            const data = docSnap.data()
            const storagePath = (data.storagePath ?? data.path ?? null) as string | null
            let downloadUrl: string | null = null
            if (storagePath) {
              try {
                const storage = getStorage(app)
                const fileRef = ref(storage, storagePath)
                downloadUrl = await getDownloadURL(fileRef)
              } catch (error) {
                console.error('Failed to fetch audio URL', error)
              }
            }
            return {
              id: docSnap.id,
              patientId: data.patientId ?? 'Unknown',
              recordedAt: data.recordedAt?.toDate?.() ?? null,
              notes: data.notes ?? null,
              analysisStatus: data.analysisStatus ?? 'pending',
              storagePath,
              downloadUrl,
            }
          }),
        )
        setRecordings(mapped)
      },
      (error) => setDataError(error.message),
    )

    return () => {
      unsubDoctor()
      unsubClinic()
      unsubPatients()
      unsubRecordings()
    }
  }, [user, persona, clinicId])

  const handleCreatePatientStub = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    setIsCreatingStub(true)
    setPatientStubResult(null)
    setStubError(null)
    try {
      const response = await callCreatePatientStub(
        newPatientName.trim()
          ? {
              patient: {
                displayName: newPatientName.trim(),
              },
            }
          : undefined,
      )
      setPatientStubResult(response.patientId)
      setNewPatientName('')
    } catch (error) {
      setStubError(getFirebaseErrorMessage(error))
    } finally {
      setIsCreatingStub(false)
    }
  }

  const headerSubtitle = useMemo(() => {
    if (!clinicInfo) {
      if (!clinicId) {
        return 'Missing clinic assignment. Contact an admin to be linked to a clinic.'
      }
      return `Clinic ID: ${clinicId}`
    }
    return `${clinicInfo.name ?? 'Clinic'} • ${clinicInfo.phone ?? clinicInfo.contactEmail ?? ''}`
  }, [clinicInfo, clinicId])

  if (!user || persona !== 'doctor') {
    return <PageLoader />
  }

  return (
    <div className="dashboard-page">
      <div className="dashboard-container">
        <header className="dashboard-header">
          <div>
            <p className="dashboard-eyebrow">Doctor workspace</p>
            <h1>{doctorInfo?.profile?.fullName ?? 'Doctor dashboard'}</h1>
            <p className="dashboard-subtitle">{headerSubtitle}</p>
          </div>
          <div className="dashboard-header-actions">
            <Link to="/patient/signup" className="btn-primary" {...getPrefetchEvents('PatientSignup')}>
              Invite patient
            </Link>
            <Link to="/" className="btn-nav-secondary">
              Return home
            </Link>
          </div>
        </header>

        {dataError && <p className="status-pill danger">{dataError}</p>}

        <section className="dashboard-panel onboarding-panel">
          <div className="dashboard-panel-header">
            <h2>Generate patient ID</h2>
            <p className="dashboard-panel-subtitle">
              Create a patient record and share the ID so they can activate their portal.
            </p>
          </div>
          <form className="patient-stub-form" onSubmit={handleCreatePatientStub}>
            <label className="form-field">
              <span>Patient display name (optional)</span>
              <input
                type="text"
                value={newPatientName}
                onChange={(event) => setNewPatientName(event.target.value)}
                placeholder="e.g., Sam G."
              />
            </label>
            <button type="submit" className="btn-primary" disabled={isCreatingStub}>
              {isCreatingStub ? 'Generating…' : 'Generate patient ID'}
            </button>
            {patientStubResult && (
              <p className="status-pill success">
                Share this ID with your patient: <strong>{patientStubResult}</strong>
              </p>
            )}
            {stubError && <p className="status-pill danger">{stubError}</p>}
          </form>
        </section>

        <section className="dashboard-panel">
          <div className="dashboard-panel-header">
            <h2>Assigned patients ({patients.length})</h2>
          </div>
          {patients.length === 0 ? (
            <p>No patients assigned yet. Generate an ID and share it with your clinic.</p>
          ) : (
            <div className="dashboard-table">
              <div className="dashboard-table-row dashboard-table-head">
                <span>Patient</span>
                <span>Status</span>
                <span>Last update</span>
              </div>
              {patients.map((patient) => (
                <div key={patient.id} className="dashboard-table-row">
                  <div>
                    <p className="dashboard-table-primary">{patient.displayName}</p>
                    <p className="dashboard-table-secondary">{patient.id}</p>
                  </div>
                  <span>
                    <span className="status-pill">{patient.status ?? 'unknown'}</span>
                  </span>
                  <span>{patient.updatedAt ? patient.updatedAt.toLocaleString() : '—'}</span>
                </div>
              ))}
            </div>
          )}
        </section>

        <section className="dashboard-panel">
          <div className="dashboard-panel-header">
            <h2>Latest recordings</h2>
            <p className="dashboard-panel-subtitle">
              Most recent uploads across your clinic. Audio analysis will update once ML is connected.
            </p>
          </div>
          {recordings.length === 0 ? (
            <p>No recordings yet. Ask patients to upload from their dashboard.</p>
          ) : (
            <div className="dashboard-table">
              <div className="dashboard-table-row dashboard-table-head">
                <span>Patient ID</span>
                <span>Recorded</span>
                <span>Status</span>
                <span>Notes</span>
                <span>Listen</span>
              </div>
              {recordings.map((recording) => (
                <div key={recording.id} className="dashboard-table-row">
                  <span>{recording.patientId}</span>
                  <span>{recording.recordedAt ? recording.recordedAt.toLocaleString() : '—'}</span>
                  <span>
                    <span className="status-pill">{recording.analysisStatus ?? 'pending'}</span>
                  </span>
                  <span className="dashboard-table-secondary">
                    {recording.notes ? recording.notes : 'No notes yet'}
                  </span>
                  <span>
                    {recording.downloadUrl ? (
                      <audio controls src={recording.downloadUrl} />
                    ) : (
                      '—'
                    )}
                  </span>
                </div>
              ))}
            </div>
          )}
        </section>
      </div>
    </div>
  )
}



