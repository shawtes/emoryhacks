import { useState } from 'react'

export interface PatientInfo {
  patientId: string
  name: string
  age: number | ''
  gender: 'male' | 'female' | 'other' | ''
  dateOfBirth: string
  notes: string
}

interface PatientFormProps {
  onSubmit: (info: PatientInfo) => void
  onCancel?: () => void
  initialData?: Partial<PatientInfo>
}

export default function PatientForm({ onSubmit, onCancel, initialData }: PatientFormProps) {
  const [formData, setFormData] = useState<PatientInfo>({
    patientId: initialData?.patientId || '',
    name: initialData?.name || '',
    age: initialData?.age || '',
    gender: initialData?.gender || '',
    dateOfBirth: initialData?.dateOfBirth || '',
    notes: initialData?.notes || '',
  })

  const handleChange = (field: keyof PatientInfo, value: string | number) => {
    setFormData(prev => ({ ...prev, [field]: value }))
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    // Skip validation - just submit and go to audio assessment
    onSubmit(formData)
  }

  return (
    <form className="patient-form card" onSubmit={handleSubmit}>
      <div className="form-header">
        <h3>Patient Information</h3>
        <p className="form-subtitle">Enter patient details for this assessment</p>
      </div>

      <div className="form-grid">
        <div className="form-group">
          <label htmlFor="patientId">
            Patient ID <span className="required">*</span>
          </label>
          <input
            id="patientId"
            type="text"
            value={formData.patientId}
            onChange={(e) => handleChange('patientId', e.target.value)}
            placeholder="Enter patient ID"
          />
        </div>

        <div className="form-group">
          <label htmlFor="name">
            Patient Name <span className="required">*</span>
          </label>
          <input
            id="name"
            type="text"
            value={formData.name}
            onChange={(e) => handleChange('name', e.target.value)}
            placeholder="Enter patient name"
          />
        </div>

        <div className="form-group">
          <label htmlFor="dateOfBirth">
            Date of Birth <span className="required">*</span>
          </label>
          <input
            id="dateOfBirth"
            type="date"
            value={formData.dateOfBirth}
            onChange={(e) => handleChange('dateOfBirth', e.target.value)}
            max={new Date().toISOString().split('T')[0]}
          />
        </div>

        <div className="form-group">
          <label htmlFor="gender">Gender</label>
          <select
            id="gender"
            value={formData.gender}
            onChange={(e) => handleChange('gender', e.target.value as PatientInfo['gender'])}
          >
            <option value="">Select gender</option>
            <option value="male">Male</option>
            <option value="female">Female</option>
            <option value="other">Other</option>
          </select>
        </div>

        <div className="form-group full-width">
          <label htmlFor="notes">Clinical Notes (Optional)</label>
          <textarea
            id="notes"
            value={formData.notes}
            onChange={(e) => handleChange('notes', e.target.value)}
            placeholder="Enter any relevant clinical notes or observations"
            rows={3}
          />
        </div>
      </div>

      <div className="form-actions">
        {onCancel && (
          <button type="button" className="btn-secondary" onClick={onCancel}>
            Cancel
          </button>
        )}
        <button type="submit" className="btn-primary">
          Continue to Assessment
        </button>
      </div>
    </form>
  )
}

