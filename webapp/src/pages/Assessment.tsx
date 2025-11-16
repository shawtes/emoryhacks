import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import AudioRecorder from '../components/AudioRecorder'
import FileUploader from '../components/FileUploader'
import ResultsDisplay from '../components/ResultsDisplay'
import PatientForm, { PatientInfo } from '../components/PatientForm'
import { PredictionResult } from '../types'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

interface AssessmentSession {
  id: string
  patientInfo: PatientInfo
  result: PredictionResult
  timestamp: Date
  audioFileName?: string
}

export default function Assessment() {
  const navigate = useNavigate()
  // Start directly at audio step - no patient form required
  const [currentStep, setCurrentStep] = useState<'patient' | 'audio' | 'results'>('audio')
  // Use default patient info - can be filled later if needed
  const [patientInfo, setPatientInfo] = useState<PatientInfo>({
    patientId: 'TEMP-' + Date.now(),
    name: 'Temporary Patient',
    age: '',
    gender: '',
    dateOfBirth: '',
    notes: '',
  })
  const [result, setResult] = useState<PredictionResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [sessions, setSessions] = useState<AssessmentSession[]>([])
  const [showHistory, setShowHistory] = useState(false)

  const handleLogout = () => {
    // TODO: Add actual logout logic (clear tokens, etc.)
    navigate('/login')
  }

  const handlePatientSubmit = (info: PatientInfo) => {
    setPatientInfo(info)
    setCurrentStep('audio')
  }

  const handlePrediction = async (audioFile: File) => {
    setLoading(true)
    setError(null)
    setResult(null)

    const formData = new FormData()
    formData.append('file', audioFile)

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }))
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      setResult(data)
      setCurrentStep('results')

      // Save session
      const session: AssessmentSession = {
        id: `session-${Date.now()}`,
        patientInfo,
        result: data,
        timestamp: new Date(),
        audioFileName: audioFile.name,
      }
      setSessions(prev => [session, ...prev])
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to process audio')
      console.error('Prediction error:', err)
    } finally {
      setLoading(false)
    }
  }

  const handleNewAssessment = () => {
    setCurrentStep('audio')
    // Reset to default patient info
    setPatientInfo({
      patientId: 'TEMP-' + Date.now(),
      name: 'Temporary Patient',
      age: '',
      gender: '',
      dateOfBirth: '',
      notes: '',
    })
    setResult(null)
    setError(null)
  }

  const handleExportPDF = () => {
    if (!result) return

    const content = `
DEMENTIA DETECTION ASSESSMENT REPORT
=====================================

Patient Information:
- Patient ID: ${patientInfo.patientId}
- Name: ${patientInfo.name}
- Date of Birth: ${patientInfo.dateOfBirth}
- Gender: ${patientInfo.gender || 'Not specified'}

Assessment Date: ${new Date().toLocaleString()}

Results:
- Prediction: ${result.prediction.replace('_', ' ').toUpperCase()}
- Probability: ${(result.probability * 100).toFixed(1)}%
- Confidence Level: ${result.confidence.toUpperCase()}

${patientInfo.notes ? `Clinical Notes:\n${patientInfo.notes}\n` : ''}

IMPORTANT DISCLAIMER:
This tool is for research and screening purposes only.
It is not a medical device and should not be used for clinical diagnosis.
Always consult with qualified healthcare professionals for medical advice.
    `.trim()

    const blob = new Blob([content], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `assessment-${patientInfo.patientId}-${Date.now()}.txt`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  return (
    <div className="app">
      <div className="container">
        <header className="header">
          <div className="header-content">
            <div className="header-main">
              <h1>Dementia Detection Assessment</h1>
              <p className="subtitle">Clinical Speech Analysis System</p>
            </div>
            <div className="header-actions">
              {sessions.length > 0 && (
                <button
                  className="btn-history"
                  onClick={() => setShowHistory(!showHistory)}
                >
                  {showHistory ? 'Hide' : 'View'} History ({sessions.length})
                </button>
              )}
              <button className="btn-logout" onClick={handleLogout}>
                Logout
              </button>
            </div>
          </div>
          <div className="disclaimer-banner">
            <span className="disclaimer-icon">⚠️</span>
            <span>Research use only - Not a medical device</span>
          </div>
        </header>

        <main className="main-content">
          {/* Step Indicator */}
          <div className="step-indicator">
            <div className={`step ${currentStep === 'patient' ? 'active' : currentStep === 'audio' || currentStep === 'results' ? 'completed' : ''}`}>
              <span className="step-number">1</span>
              <span className="step-label">Patient Info</span>
            </div>
            <div className={`step-line ${currentStep === 'audio' || currentStep === 'results' ? 'completed' : ''}`}></div>
            <div className={`step ${currentStep === 'audio' ? 'active' : currentStep === 'results' ? 'completed' : ''}`}>
              <span className="step-number">2</span>
              <span className="step-label">Audio Assessment</span>
            </div>
            <div className={`step-line ${currentStep === 'results' ? 'completed' : ''}`}></div>
            <div className={`step ${currentStep === 'results' ? 'active' : ''}`}>
              <span className="step-number">3</span>
              <span className="step-label">Results</span>
            </div>
          </div>

          {/* Patient Form Step */}
          {currentStep === 'patient' && (
            <div className="step-content">
              <PatientForm onSubmit={handlePatientSubmit} />
            </div>
          )}

          {/* Audio Upload Step */}
          {currentStep === 'audio' && (
            <div className="step-content">
              <div className="patient-info-bar">
                <div className="patient-info-item">
                  <span className="label">Patient:</span>
                  <span className="value">{patientInfo.name} (ID: {patientInfo.patientId})</span>
                </div>
                <button className="btn-link" onClick={() => setCurrentStep('patient')}>
                  Edit Patient Info
                </button>
              </div>

              <div className="upload-section">
                <div className="upload-options">
                  <div className="upload-card">
                    <h2>Record Audio</h2>
                    <AudioRecorder onRecordingComplete={handlePrediction} disabled={loading} />
                  </div>
                  <div className="divider">OR</div>
                  <div className="upload-card">
                    <h2>Upload Audio File</h2>
                    <FileUploader onFileSelect={handlePrediction} disabled={loading} />
                  </div>
                </div>
              </div>

              {loading && (
                <div className="loading">
                  <div className="spinner"></div>
                  <p>Analyzing audio sample...</p>
                  <p className="loading-subtext">This may take a few moments</p>
                </div>
              )}

              {error && (
                <div className="error">
                  <p>❌ Error: {error}</p>
                </div>
              )}
            </div>
          )}

          {/* Results Step */}
          {currentStep === 'results' && result && (
            <div className="step-content">
              <div className="results-header">
                <div className="patient-info-bar">
                  <div className="patient-info-item">
                    <span className="label">Patient:</span>
                    <span className="value">{patientInfo.name} (ID: {patientInfo.patientId})</span>
                  </div>
                  <div className="patient-info-item">
                    <span className="label">Date:</span>
                    <span className="value">{new Date().toLocaleDateString()}</span>
                  </div>
                </div>
                <div className="results-actions">
                  <button className="btn-secondary" onClick={handleExportPDF}>
                    Export Report
                  </button>
                  <button className="btn-primary" onClick={handleNewAssessment}>
                    New Assessment
                  </button>
                </div>
              </div>
              <ResultsDisplay result={result} patientInfo={patientInfo} />
            </div>
          )}

          {/* History Panel */}
          {showHistory && sessions.length > 0 && (
            <div className="history-panel">
              <h3>Assessment History</h3>
              <div className="history-list">
                {sessions.map((session) => (
                  <div key={session.id} className="history-item">
                    <div className="history-item-header">
                      <span className="history-patient">{session.patientInfo.name}</span>
                      <span className="history-date">{session.timestamp.toLocaleString()}</span>
                    </div>
                    <div className="history-item-details">
                      <span className={`history-result ${session.result.prediction}`}>
                        {session.result.prediction.replace('_', ' ').toUpperCase()}
                      </span>
                      <span className="history-probability">
                        {(session.result.probability * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </main>

        <footer className="footer">
          <p>
            This tool is for research purposes only and should not be used for clinical diagnosis.
            Please consult with qualified healthcare professionals for medical advice.
          </p>
        </footer>
      </div>
    </div>
  )
}

