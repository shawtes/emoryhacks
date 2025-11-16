import { PredictionResult } from '../types'
import { PatientInfo } from './PatientForm'

interface ResultsDisplayProps {
  result: PredictionResult
  patientInfo?: PatientInfo
}

export default function ResultsDisplay({ result, patientInfo }: ResultsDisplayProps) {
  const { prediction, probability, confidence, message } = result

  const isDementia = prediction === 'dementia'
  const confidenceLevel = confidence.toLowerCase()

  // Color coding based on prediction and confidence
  const getConfidenceColor = () => {
    if (confidenceLevel === 'high') return '#28a745'
    if (confidenceLevel === 'medium') return '#ffc107'
    return '#ffc107'
  }

  const getPredictionColor = () => {
    return isDementia ? '#dc3545' : '#28a745'
  }

  return (
    <div className="results-display">
      {patientInfo && (
        <div className="patient-info-section">
          <h3>Patient Information</h3>
          <div className="patient-info-grid">
            <div className="patient-info-row">
              <span className="label">Patient ID:</span>
              <span className="value">{patientInfo.patientId}</span>
            </div>
            <div className="patient-info-row">
              <span className="label">Name:</span>
              <span className="value">{patientInfo.name}</span>
            </div>
            {patientInfo.dateOfBirth && (
              <div className="patient-info-row">
                <span className="label">Date of Birth:</span>
                <span className="value">{new Date(patientInfo.dateOfBirth).toLocaleDateString()}</span>
              </div>
            )}
            {patientInfo.gender && (
              <div className="patient-info-row">
                <span className="label">Gender:</span>
                <span className="value">{patientInfo.gender.charAt(0).toUpperCase() + patientInfo.gender.slice(1)}</span>
              </div>
            )}
          </div>
          {patientInfo.notes && (
            <div style={{ marginTop: '0.75rem', paddingTop: '0.75rem', borderTop: '1px solid #e9ecef' }}>
              <span className="label">Clinical Notes: </span>
              <span className="value" style={{ fontWeight: 'normal' }}>{patientInfo.notes}</span>
            </div>
          )}
        </div>
      )}

      <h2>Assessment Results</h2>
      
      <div className={`prediction-card ${isDementia ? 'dementia' : 'no-dementia'}`}>
        <div className="prediction-header">
          <span className="prediction-icon">
            {isDementia ? '⚠️' : '✅'}
          </span>
          <h3 className="prediction-title">
            {isDementia ? 'Dementia Indicators Detected' : 'No Significant Indicators'}
          </h3>
        </div>

        <div className="probability-section">
          <div className="probability-label">Prediction Probability</div>
          <div className="probability-value" style={{ color: getPredictionColor() }}>
            {(probability * 100).toFixed(1)}%
          </div>
          <div className="progress-bar-container">
            <div
              className="progress-bar"
              style={{
                width: `${probability * 100}%`,
                backgroundColor: getPredictionColor()
              }}
            />
          </div>
        </div>

        <div className="confidence-section">
          <span className="confidence-label">Confidence Level:</span>
          <span
            className="confidence-badge"
            style={{ backgroundColor: getConfidenceColor() }}
          >
            {confidence.toUpperCase()}
          </span>
        </div>

        <div className="message-section">
          <p>{message}</p>
        </div>
      </div>

      <div className="disclaimer-box">
        <strong>Important:</strong> This tool is for research and screening purposes only.
        It is not a medical device and should not be used for clinical diagnosis.
        Always consult with qualified healthcare professionals for medical advice.
      </div>
    </div>
  )
}

