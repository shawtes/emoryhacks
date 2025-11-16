import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import LiveStatusBadge from '../components/LiveStatusBadge'
import useStaggeredReveal from '../hooks/useStaggeredReveal'
import { useFirebase } from '../context/FirebaseContext'
import { useAuth } from '../context/AuthContext'
import { getFirebaseApp } from '../lib/firebase'
import { collection, doc, getFirestore, limit, onSnapshot, orderBy, query } from 'firebase/firestore'
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

export default function PatientDashboard() {
  const { selectPersona } = useFirebase()
  const { user, persona, claims } = useAuth()
  const [patientDoc, setPatientDoc] = useState<PatientDocData | null>(null)
  const [doctorDoc, setDoctorDoc] = useState<DoctorDocData | null>(null)
  const [clinicDoc, setClinicDoc] = useState<ClinicDocData | null>(null)
  const [recordings, setRecordings] = useState<RecordingHistory[]>([])
  const patientIdClaim = (claims?.patientId as string) ?? null

  useStaggeredReveal()

  useEffect(() => {
    if (persona !== 'patient') {
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
    const recordingsQuery = query(recordingsRef, orderBy('recordedAt', 'desc'), limit(10))
    const unsubscribe = onSnapshot(recordingsQuery, (snapshot) => {
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

  const totalRecordings = recordings.length
  const lastRecorded = recordings[0]?.recordedAt

  if (!user || persona !== 'patient') {
    return <PageLoader />
  }

  return (
    <div className="patient-dashboard">
      <div className="patient-dashboard__shell">
        <header className="patient-dashboard__hero" data-animate>
          <div className="patient-dashboard__hero-copy">
            <p className="eyebrow">
              {clinicDoc?.name ? `Connected to ${clinicDoc.name}` : 'Connected clinic'}
            </p>
            <h1>Welcome back{patientDoc?.displayName ? `, ${patientDoc.displayName}` : ''}</h1>
            <p>Your care team can review every recording you send from here.</p>
            <LiveStatusBadge />
          </div>
          <div className="patient-dashboard__hero-actions">
            <Link to="/patientassessment" className="btn-primary">
              Resume audio check-in
            </Link>
            <Link to="/patient/signup" className="btn-nav-secondary">
              Invite a caregiver
            </Link>
            <Link to="/" className="btn-nav-secondary">
              Return home
            </Link>
          </div>
        </header>

        <section className="patient-dashboard__panel cta-panel" data-animate>
          <div className="panel-header">
            <div>
              <p className="eyebrow">Guided recording</p>
              <h2>Ready for your next voice check-in?</h2>
            </div>
          </div>
          <p>
            Launch the patient assessment to walk through the scripted recording experience and upload
            a fresh audio sample directly to your doctor.
          </p>
          <Link to="/patientassessment" className="btn-primary">
            Start patient assessment
          </Link>
        </section>

        <section className="patient-dashboard__metrics" data-animate>
          <article className="patient-dashboard__metric-card">
            <p className="metric-label">Total recordings</p>
            <p className="metric-value">{totalRecordings}</p>
            <p className="metric-helper">
              {lastRecorded ? `Last upload ${lastRecorded.toLocaleDateString()}` : 'No uploads yet'}
            </p>
          </article>
          <article className="patient-dashboard__metric-card">
            <p className="metric-label">Assigned doctor</p>
            <p className="metric-value">{doctorDoc?.profile?.fullName ?? 'Pending'}</p>
            <p className="metric-helper">
              {doctorDoc?.profile?.specialties?.join(', ') ?? 'Clinic will assign you shortly'}
            </p>
          </article>
          <article className="patient-dashboard__metric-card">
            <p className="metric-label">Clinic contact</p>
            <p className="metric-value">{clinicDoc?.phone ?? clinicDoc?.contactEmail ?? 'TBD'}</p>
            <p className="metric-helper">Reach out if you need help or a reminder</p>
          </article>
        </section>

        <section className="patient-dashboard__panel" data-animate>
          <div className="panel-header">
            <div>
              <p className="eyebrow">Upload history</p>
              <h2>Your recent recordings</h2>
            </div>
            <Link to="/patientassessment" className="btn-nav-secondary small">
              Record new audio
            </Link>
          </div>
          {recordings.length === 0 ? (
            <p>You haven’t uploaded any audio yet. Use the recorder above to send your first update.</p>
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
    </div>
  )
}
