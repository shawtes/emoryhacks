import { useEffect, useState, type ChangeEvent, type FormEvent } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useFirebase } from '../context/FirebaseContext'
import {
  callActivatePatient,
  callRegisterDoctor,
  type RegisterDoctorPayload,
} from '../services/firebaseFunctions'
import { getFirebaseErrorMessage } from '../utils/firebaseErrors'
import { useAuth } from '../context/AuthContext'

type AuthRole = 'patient' | 'doctor'
type AuthMode = 'login' | 'signup'

interface RoleHighlight {
  title: string
  description: string
}

interface RoleContent {
  icon: string
  label: string
  loginTitle: string
  loginSubtitle: string
  signupTitle: string
  signupSubtitle: string
  loginAction: string
  signupAction: string
  loginSecondary: {
    text: string
    linkLabel: string
    link: string
  }
  signupSecondary: {
    text: string
    linkLabel: string
    link: string
  }
  highlights: RoleHighlight[]
  partnerPrompt: string
  partnerLabel: string
  partnerLink: string
}

const ROLE_COPY: Record<AuthRole, RoleContent> = {
  doctor: {
    icon: '🏥',
    label: 'Clinics & Providers',
    loginTitle: 'Doctor Login',
    loginSubtitle: 'Sign in to manage dementia assessments, notes, and reports.',
    signupTitle: 'Create Doctor Account',
    signupSubtitle: 'Offer AI-powered speech screening across your service lines.',
    loginAction: 'Enter Doctor Portal',
    signupAction: 'Create Doctor Account',
    loginSecondary: {
      text: 'New to the platform?',
      linkLabel: 'Register your practice',
      link: '/doctor/signup',
    },
    signupSecondary: {
      text: 'Already working with us?',
      linkLabel: 'Go to doctor login',
      link: '/doctor/login',
    },
    highlights: [
      {
        title: 'Clinic-ready workflows',
        description: 'Enroll patients, capture notes, and keep caregivers updated from one dashboard.',
      },
      {
        title: 'Audio-first assessments',
        description: 'Record live or upload speech samples to generate dementia risk indicators.',
      },
      {
        title: 'White-labeled reports',
        description: 'Deliver physician-friendly PDFs and share biomarkers directly with your EHR.',
      },
    ],
    partnerPrompt: 'Supporting a patient or caregiver?',
    partnerLabel: 'Switch to patient portal',
    partnerLink: '/patient/login',
  },
  patient: {
    icon: '👤',
    label: 'Patients & Care Partners',
    loginTitle: 'Patient Login',
    loginSubtitle: 'Check in on voice assessments and share updates with your care team.',
    signupTitle: 'Create Patient Account',
    signupSubtitle: 'Follow guided speech prompts and send results securely to your doctor.',
    loginAction: 'Enter Patient Portal',
    signupAction: 'Create Patient Account',
    loginSecondary: {
      text: 'Need an account?',
      linkLabel: 'Register as a patient',
      link: '/patient/signup',
    },
    signupSecondary: {
      text: 'Already registered?',
      linkLabel: 'Go to patient login',
      link: '/patient/login',
    },
    highlights: [
      {
        title: 'Guided speech tasks',
        description: 'Complete storytelling exercises from any phone, tablet, or laptop.',
      },
      {
        title: 'Secure sharing',
        description: 'Let loved ones and clinicians view your latest results instantly.',
      },
      {
        title: 'Progress tracking',
        description: 'Monitor how voice biomarkers trend over time before each visit.',
      },
    ],
    partnerPrompt: 'Managing a clinic or research program?',
    partnerLabel: 'Switch to doctor portal',
    partnerLink: '/doctor/login',
  },
}

interface RoleAuthPageProps {
  role: AuthRole
  mode: AuthMode
}

export default function RoleAuthPage({ role, mode }: RoleAuthPageProps) {
  const navigate = useNavigate()
  const { selectPersona, isLoading } = useFirebase()
  const { signIn, isAuthLoading, error: authError } = useAuth()
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [statusMessage, setStatusMessage] = useState<string | null>(null)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [doctorForm, setDoctorForm] = useState({
    clinicId: '',
    fullName: '',
    email: '',
    password: '',
    specialties: '',
    phone: '',
  })
  const [patientForm, setPatientForm] = useState({
    patientId: '',
    email: '',
    password: '',
  })
  const [loginForm, setLoginForm] = useState({
    email: '',
    password: '',
  })
  const content = ROLE_COPY[role]
  const isLogin = mode === 'login'
  const isSignup = !isLogin

  useEffect(() => {
    void selectPersona(role)
  }, [role, selectPersona])

  const heading = isLogin ? content.loginTitle : content.signupTitle
  const subtitle = isLogin ? content.loginSubtitle : content.signupSubtitle
  const primaryLabel = isLogin ? content.loginAction : content.signupAction
  const secondary = isLogin ? content.loginSecondary : content.signupSecondary

  const handleSecondaryClick = () => {
    void selectPersona(role)
  }

  const handlePartnerClick = () => {
    const partnerRole = role === 'doctor' ? 'patient' : 'doctor'
    void selectPersona(partnerRole)
  }

  const handleDoctorInput =
    (field: keyof typeof doctorForm) =>
    (event: ChangeEvent<HTMLInputElement>) => {
      setDoctorForm((prev) => ({ ...prev, [field]: event.target.value }))
    }

  const handlePatientInput =
    (field: keyof typeof patientForm) =>
    (event: ChangeEvent<HTMLInputElement>) => {
      setPatientForm((prev) => ({ ...prev, [field]: event.target.value }))
    }

  const handleLoginInput =
    (field: keyof typeof loginForm) =>
    (event: ChangeEvent<HTMLInputElement>) => {
      setLoginForm((prev) => ({ ...prev, [field]: event.target.value }))
    }

  const handleLoginSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    setIsSubmitting(true)
    setErrorMessage(null)
    setStatusMessage(null)
    try {
      await signIn(role, loginForm.email.trim(), loginForm.password)
      const destination = role === 'patient' ? '/patient/dashboard' : '/doctor/dashboard'
      navigate(destination, { replace: true })
    } catch (error) {
      const friendlyMessage = getFirebaseErrorMessage(error)
      setErrorMessage(friendlyMessage)
      if (friendlyMessage?.toLowerCase().includes('use the patient portal') || friendlyMessage?.toLowerCase().includes('use the doctor portal')) {
        window.alert(friendlyMessage)
      }
    } finally {
      setIsSubmitting(false)
    }
  }

  const handleSignupSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    setIsSubmitting(true)
    setErrorMessage(null)
    setStatusMessage(null)
    try {
      if (role === 'doctor') {
        const payload: RegisterDoctorPayload = {
          clinicId: doctorForm.clinicId.trim(),
          email: doctorForm.email.trim(),
          password: doctorForm.password,
          profile: {
            fullName: doctorForm.fullName.trim(),
            specialties: doctorForm.specialties
              .split(',')
              .map((spec) => spec.trim())
              .filter(Boolean),
            phone: doctorForm.phone.trim() || null,
          },
        }
        await callRegisterDoctor(payload)
        setStatusMessage('Doctor account created. You can sign in once your clinic approves the invite.')
        setDoctorForm({
          clinicId: '',
          fullName: '',
          email: '',
          password: '',
          specialties: '',
          phone: '',
        })
      } else {
        await callActivatePatient({
          patientId: patientForm.patientId.trim(),
          email: patientForm.email.trim(),
          password: patientForm.password,
        })
        setStatusMessage('Patient account activated. You can now log in with your email.')
        setPatientForm({
          patientId: '',
          email: '',
          password: '',
        })
      }
    } catch (error) {
      setErrorMessage(getFirebaseErrorMessage(error))
    } finally {
      setIsSubmitting(false)
    }
  }

  const doctorFields = (
    <>
      <label className="form-field">
        <span>Clinic ID</span>
        <input
          type="text"
          value={doctorForm.clinicId}
          onChange={handleDoctorInput('clinicId')}
          placeholder="Provided by your clinic admin"
          required
        />
      </label>
      <label className="form-field">
        <span>Full name</span>
        <input
          type="text"
          value={doctorForm.fullName}
          onChange={handleDoctorInput('fullName')}
          placeholder="Dr. Jamie Rivera"
          required
        />
      </label>
      <label className="form-field">
        <span>Email</span>
        <input
          type="email"
          value={doctorForm.email}
          onChange={handleDoctorInput('email')}
          placeholder="you@clinic.com"
          required
        />
      </label>
      <label className="form-field">
        <span>Password</span>
        <input
          type="password"
          value={doctorForm.password}
          onChange={handleDoctorInput('password')}
          placeholder="Create a secure password"
          required
        />
      </label>
      <label className="form-field">
        <span>Specialties (comma separated)</span>
        <input
          type="text"
          value={doctorForm.specialties}
          onChange={handleDoctorInput('specialties')}
          placeholder="Neurology, Geriatrics"
        />
      </label>
      <label className="form-field">
        <span>Clinic phone (optional)</span>
        <input
          type="tel"
          value={doctorForm.phone}
          onChange={handleDoctorInput('phone')}
          placeholder="(555) 123-4567"
        />
      </label>
    </>
  )

  const patientFields = (
    <>
      <label className="form-field">
        <span>Patient ID</span>
        <input
          type="text"
          value={patientForm.patientId}
          onChange={handlePatientInput('patientId')}
          placeholder="Provided by your doctor"
          required
        />
      </label>
      <label className="form-field">
        <span>Email</span>
        <input
          type="email"
          value={patientForm.email}
          onChange={handlePatientInput('email')}
          placeholder="you@email.com"
          required
        />
      </label>
      <label className="form-field">
        <span>Password</span>
        <input
          type="password"
          value={patientForm.password}
          onChange={handlePatientInput('password')}
          placeholder="Create a secure password"
          required
        />
      </label>
    </>
  )

  const signupForm = isSignup ? (
    <form className="auth-form auth-form-stack" onSubmit={handleSignupSubmit}>
      {role === 'doctor' ? doctorFields : patientFields}
      {errorMessage && <p className="auth-error">{errorMessage}</p>}
      {statusMessage && <p className="auth-success">{statusMessage}</p>}
      <button type="submit" className="btn-auth-primary" disabled={isSubmitting || isLoading}>
        {isSubmitting ? 'Submitting…' : primaryLabel}
      </button>
      <div className="auth-divider">
        <span>{secondary.text}</span>
      </div>
      <Link
        to={secondary.link}
        className="btn-auth-secondary"
        onClick={handleSecondaryClick}
        style={{ width: '100%', textAlign: 'center' }}
      >
        {secondary.linkLabel}
      </Link>
      <Link to="/" className="btn-return-home">
        ← Return home
      </Link>
    </form>
  ) : null

  const loginFormNode = (
    <form className="auth-form auth-form-stack" onSubmit={handleLoginSubmit}>
      <label className="form-field">
        <span>Email</span>
        <input
          type="email"
          value={loginForm.email}
          onChange={handleLoginInput('email')}
          placeholder="you@email.com"
          required
        />
      </label>
      <label className="form-field">
        <span>Password</span>
        <input
          type="password"
          value={loginForm.password}
          onChange={handleLoginInput('password')}
          placeholder="Your password"
          required
        />
      </label>
      {(errorMessage || authError) && (
        <p className="auth-error">{errorMessage ?? authError}</p>
      )}
      <button
        type="submit"
        className="btn-auth-primary"
        disabled={isSubmitting || isLoading || isAuthLoading}
      >
        {isSubmitting || isAuthLoading ? 'Signing in…' : primaryLabel}
      </button>
      <div className="auth-divider">
        <span>{secondary.text}</span>
      </div>
      <Link
        to={secondary.link}
        className="btn-auth-secondary"
        onClick={handleSecondaryClick}
        style={{ width: '100%', textAlign: 'center' }}
      >
        {secondary.linkLabel}
      </Link>
      <Link to="/" className="btn-return-home">
        ← Return home
      </Link>
    </form>
  )

  return (
    <div className="auth-page">
      <div className="auth-container auth-container-wide">
        <div className="auth-header">
          <Link to="/" className="auth-logo">
            🧠 Dementia Detection
          </Link>
          <div className={`auth-role-pill ${role}`}>
            <span className="auth-role-icon">{content.icon}</span>
            <span>{content.label}</span>
          </div>
          <h1>{heading}</h1>
          <p className="auth-subtitle">{subtitle}</p>
        </div>

        {signupForm ?? loginFormNode}

        <ul className="auth-highlights">
          {content.highlights.map((highlight) => (
            <li key={highlight.title} className="auth-highlight">
              <div className="auth-highlight-icon">✓</div>
              <div className="auth-highlight-text">
                <h4>{highlight.title}</h4>
                <p>{highlight.description}</p>
              </div>
            </li>
          ))}
        </ul>

        <div className="auth-switch-role">
          <span>{content.partnerPrompt} </span>
          <Link to={content.partnerLink} onClick={handlePartnerClick}>
            {content.partnerLabel}
          </Link>
        </div>
      </div>
    </div>
  )
}


