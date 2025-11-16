import { Link } from 'react-router-dom'

export default function Home() {
  return (
    <div className="home-page">
      <header className="home-header">
        <div className="home-header-content">
          <div className="logo-section">
            <h1 className="logo">🧠 Dementia Detection</h1>
            <p className="tagline">Clinical Speech Analysis System</p>
          </div>
          <nav className="home-nav">
            <Link to="/login" className="btn-nav-secondary">Login</Link>
            <Link to="/signup" className="btn-nav-primary btn-primary">Sign Up</Link>
          </nav>
        </div>
      </header>

      <main className="home-main">
        <section className="hero-section">
          <div className="hero-content">
            <h2 className="hero-title">
              Advanced Dementia Detection Through Speech Analysis
            </h2>
            <p className="hero-subtitle">
              A scalable, research-grade platform for healthcare professionals to screen 
              for dementia indicators using advanced machine learning and audio analysis.
            </p>
            <div className="hero-actions">
              <Link to="/signup" className="btn-hero-primary">
                Get Started
              </Link>
              <Link to="/login" className="btn-hero-secondary">
                Sign In
              </Link>
            </div>
          </div>
        </section>

        <section className="features-section">
          <div className="features-container">
            <h3 className="section-title">Key Features</h3>
            <div className="features-grid">
              <div className="feature-card">
                <div className="feature-icon">🎯</div>
                <h4>Accurate Screening</h4>
                <p>
                  Advanced ML models analyze speech patterns to identify potential 
                  dementia indicators with high accuracy.
                </p>
              </div>
              <div className="feature-card">
                <div className="feature-icon">⚡</div>
                <h4>Fast Results</h4>
                <p>
                  Get assessment results in minutes, enabling quick clinical decision-making 
                  and patient care planning.
                </p>
              </div>
              <div className="feature-card">
                <div className="feature-icon">🔒</div>
                <h4>Secure & HIPAA Compliant</h4>
                <p>
                  Built with healthcare security standards in mind, ensuring patient data 
                  privacy and compliance.
                </p>
              </div>
              <div className="feature-card">
                <div className="feature-icon">📊</div>
                <h4>Clinical Reports</h4>
                <p>
                  Generate detailed assessment reports with patient information, results, 
                  and clinical notes for medical records.
                </p>
              </div>
              <div className="feature-card">
                <div className="feature-icon">🏥</div>
                <h4>Hospital Ready</h4>
                <p>
                  Designed for seamless integration into hospital workflows and 
                  electronic medical record systems.
                </p>
              </div>
              <div className="feature-card">
                <div className="feature-icon">📈</div>
                <h4>Scalable Platform</h4>
                <p>
                  Built to scale across multiple hospitals and clinics nationwide, 
                  supporting high-volume assessments.
                </p>
              </div>
            </div>
          </div>
        </section>

        <section className="workflow-section">
          <div className="workflow-container">
            <h3 className="section-title">How It Works</h3>
            <div className="workflow-steps">
              <div className="workflow-step">
                <div className="step-number">1</div>
                <h4>Patient Information</h4>
                <p>Enter patient details and clinical notes</p>
              </div>
              <div className="workflow-arrow">→</div>
              <div className="workflow-step">
                <div className="step-number">2</div>
                <h4>Audio Assessment</h4>
                <p>Record or upload patient speech sample</p>
              </div>
              <div className="workflow-arrow">→</div>
              <div className="workflow-step">
                <div className="step-number">3</div>
                <h4>Analysis & Results</h4>
                <p>AI analyzes speech patterns and generates results</p>
              </div>
              <div className="workflow-arrow">→</div>
              <div className="workflow-step">
                <div className="step-number">4</div>
                <h4>Clinical Report</h4>
                <p>Export detailed report for medical records</p>
              </div>
            </div>
          </div>
        </section>

        <section className="cta-section">
          <div className="cta-content">
            <h3>Ready to Get Started?</h3>
            <p>Join healthcare professionals across the country using our platform</p>
            <Link to="/signup" className="btn-cta">
              Create Account
            </Link>
          </div>
        </section>
      </main>

      <footer className="home-footer">
        <div className="footer-content">
          <div className="footer-disclaimer">
            <p>
              <strong>Research Use Only</strong> - This tool is for research and screening purposes only. 
              It is not a medical device and should not be used for clinical diagnosis.
            </p>
          </div>
          <div className="footer-links">
            <a href="#privacy">Privacy Policy</a>
            <a href="#terms">Terms of Service</a>
            <a href="#contact">Contact</a>
          </div>
          <p className="footer-copyright">
            © 2024 Dementia Detection Platform. All rights reserved.
          </p>
        </div>
      </footer>
    </div>
  )
}

