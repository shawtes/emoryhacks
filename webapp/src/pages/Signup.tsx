import { Link, useNavigate } from 'react-router-dom'

export default function Signup() {
  const navigate = useNavigate()

  const handleSignUp = () => {
    // No authentication system yet - just navigate to assessment
    navigate('/assessment')
  }

  return (
    <div className="auth-page">
      <div className="auth-container auth-container-wide">
        <div className="auth-header">
          <Link to="/" className="auth-logo">
            🧠 Dementia Detection
          </Link>
          <h1>Create Doctor Account</h1>
          <p className="auth-subtitle">Register to access the clinical assessment platform</p>
        </div>

        <div className="auth-form">
          <button 
            type="button" 
            className="btn-auth-primary" 
            onClick={handleSignUp}
            style={{ width: '100%', marginBottom: '1rem' }}
          >
            Create Account
          </button>

          <div className="auth-divider">
            <span>Already have an account?</span>
          </div>

          <Link to="/login" className="btn-auth-secondary" style={{ width: '100%', textAlign: 'center' }}>
            Sign In
          </Link>
        </div>

        <div className="auth-footer">
          <p>
            <strong>Research Use Only</strong> - This platform is for research and screening purposes only.
          </p>
        </div>
      </div>
    </div>
  )
}

