import { Link, useNavigate } from 'react-router-dom'

export default function Login() {
  const navigate = useNavigate()

  const handleSignIn = () => {
    // No authentication system yet - just navigate to assessment
    navigate('/assessment')
  }

  return (
    <div className="auth-page">
      <div className="auth-container">
        <div className="auth-header">
          <Link to="/" className="auth-logo">
            🧠 Dementia Detection
          </Link>
          <h1>Doctor Login</h1>
          <p className="auth-subtitle">Sign in to access the clinical assessment platform</p>
        </div>

        <div className="auth-form">
          <button 
            type="button" 
            className="btn-auth-primary" 
            onClick={handleSignIn}
            style={{ width: '100%', marginBottom: '1rem' }}
          >
            Sign In
          </button>

          <div className="auth-divider">
            <span>Don't have an account?</span>
          </div>

          <Link to="/signup" className="btn-auth-secondary" style={{ width: '100%', textAlign: 'center' }}>
            Create Account
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

