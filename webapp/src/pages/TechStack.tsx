import { Link } from 'react-router-dom'
import { useEffect, useState } from 'react'
import { getPrefetchEvents } from '../routes/lazyPages'

const stackSections = [
  {
    title: 'Frontend',
    status: 'In use',
    description:
      'React + TypeScript app built with Vite, React Router, and a single stylesheet.',
    bullets: [
      'React 18, TypeScript 5, Vite 5',
      'Client API in src/services/api.ts (fetch-based)',
      'ESLint 9 + TypeScript ESLint 8',
    ],
  },
  {
    title: 'Backend / API',
    status: 'In use',
    description:
      'FastAPI service exposes /predict and /predict-url endpoints for audio analysis.',
    bullets: [
      'FastAPI + Uvicorn',
      'CORS middleware enabled',
      'Procfile/start scripts for local/dev',
    ],
  },
  {
    title: 'ML & Audio',
    status: 'In use',
    description:
      'Classical ML with engineered features; robust audio IO/decoding for WAV/MP3/WebM.',
    bullets: [
      'scikit-learn (Gradient Boosting, RF, ensembles), joblib artifacts',
      'librosa (features, loading) + soundfile',
      'PyAV + FFmpeg (WebM/Opus decode fallback)',
      'SHAP (explainability)',
    ],
  },
  {
    title: 'Data & Ops',
    status: 'In use',
    description:
      'Firebase for patient/doctor flows and audio storage; reports versioned in-repo.',
    bullets: [
      'Firebase Auth, Functions, Storage',
      'Artifacts and reports committed under reports/',
      'Emulator-ready functions in functions/',
    ],
  },
]

// (Compliance checklist section removed to streamline the page)

export default function TechStack() {
  const [reportSummary, setReportSummary] = useState<string>('Loading technical report summary…')

  useEffect(() => {
    let isMounted = true
    const summarize = (md: string) => {
      // Basic summarization: prefer first heading + following 8-10 lines, otherwise first ~1200 chars
      const lines = md.split(/\r?\n/)
      const firstHeaderIdx = lines.findIndex((l) => /^#|^==|^##/.test(l.trim()))
      let snippet = ''
      if (firstHeaderIdx !== -1) {
        const slice = lines.slice(firstHeaderIdx, firstHeaderIdx + 14)
        snippet = slice.join('\n').trim()
      } else {
        snippet = md.slice(0, 1200).trim()
      }
      // Remove excessive markdown symbols for inline viewing
      snippet = snippet.replace(/^#{1,6}\s+/gm, '')
      return snippet
    }
    fetch('/reports/technical_report.md')
      .then((r) => (r.ok ? r.text() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then((text) => {
        if (!isMounted) return
        setReportSummary(summarize(text))
      })
      .catch(() => {
        if (!isMounted) return
        setReportSummary('Unable to load technical report. Please open /reports/technical_report.md directly.')
      })
    return () => {
      isMounted = false
    }
  }, [])

  return (
    <div className="techstack-page">
      <header className="techstack-hero">
        <div>
          <p className="techstack-kicker">VoiceVital – System overview</p>
          <h1>Tech stack and model reports</h1>
          <p>Overview of the running stack and links to the latest model metrics/visuals in this repo.</p>
        </div>
        <div className="techstack-hero-actions">
          <Link to="/patient/signup" className="btn-primary" {...getPrefetchEvents('PatientSignup')}>
            Patient flow
          </Link>
          <Link to="/doctor/dashboard" className="btn-secondary" {...getPrefetchEvents('DoctorDashboard')}>
            Doctor dashboard
          </Link>
        </div>
      </header>

      <section className="techstack-grid">
        {stackSections.map((section) => (
          <article key={section.title} className="techstack-card">
            <div className="techstack-card-header">
              <h2>{section.title}</h2>
              <span className="techstack-status">{section.status}</span>
            </div>
            <p>{section.description}</p>
            <ul>
              {section.bullets.map((bullet) => (
                <li key={bullet}>{bullet}</li>
              ))}
            </ul>
          </article>
        ))}
      </section>

      <section className="compliance-section">
        <div className="compliance-header">
          <h2>Reports & analytics</h2>
          <p>Key visualizations and a brief summary from the technical report.</p>
        </div>
        <div className="compliance-grid">
          <article className="compliance-card">
            <h3>Visualizations</h3>
            <div style={{ display: 'grid', gap: '1rem' }}>
              <div>
                <p><strong>Enhanced GB analysis</strong></p>
                <a href="/reports/visualizations/enhanced_gb_analysis.png" target="_blank" rel="noreferrer">
                  <img
                    src="/reports/visualizations/enhanced_gb_analysis.png"
                    alt="Enhanced Gradient Boosting analysis"
                    style={{ maxWidth: '100%', border: '1px solid #ddd', borderRadius: 6 }}
                  />
                </a>
              </div>
              <div>
                <p><strong>Feature category analysis</strong></p>
                <a href="/reports/visualizations/feature_category_analysis.png" target="_blank" rel="noreferrer">
                  <img
                    src="/reports/visualizations/feature_category_analysis.png"
                    alt="Feature category analysis"
                    style={{ maxWidth: '100%', border: '1px solid #ddd', borderRadius: 6 }}
                  />
                </a>
              </div>
            </div>
          </article>
          <article className="compliance-card">
            <h3>Technical report (summary)</h3>
            <div style={{ whiteSpace: 'pre-wrap' }}>
              {reportSummary}
            </div>
            <p style={{ marginTop: '0.75rem' }}>
              <a href="/reports/technical_report.md" target="_blank" rel="noreferrer" className="btn-secondary">
                View full technical report
              </a>
            </p>
          </article>
          <article className="compliance-card">
            <h3>Metrics JSON</h3>
            <ul>
              <li><a href="/reports/metrics/ensemble/ensemble_cv_metrics.json" target="_blank" rel="noreferrer">Ensemble CV metrics</a></li>
              <li><a href="/reports/metrics/rf/rf_cv_metrics.json" target="_blank" rel="noreferrer">Random Forest CV metrics</a></li>
            </ul>
          </article>
        </div>
      </section>

      <footer className="techstack-footer">
        <p>Need deeper detail? Ask for the deployment diagram or security memo during Q&A.</p>
        <Link to="/" className="btn-secondary" {...getPrefetchEvents('Home')}>
          Return to VoiceVital home
        </Link>
      </footer>
    </div>
  )
}



