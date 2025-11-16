import { Link } from 'react-router-dom'
import { getPrefetchEvents } from '../routes/lazyPages'

const stackSections = [
  {
    title: 'Intro to VoiceVital',
    status: 'Pitch-ready',
    description:
      'VoiceVital turns natural speech into objective biomarkers for dementia screening, bridging patients, caregivers, and clinicians.',
    bullets: [
      'Voice-first UX keeps visits human.',
      'Dual portals for patients and clinics share the same AI signal.',
      'Research-grade reports make reimbursement and approvals simple.',
    ],
  },
  {
    title: 'Frontend stack',
    status: 'Complete',
    description:
      'React + Vite + custom design system, with Framer Motion-ready components, lazy-loaded hero media, and accessibility baked in.',
    bullets: [
      'Reusable hero, insights ribbon, and carousel components.',
      'Custom hooks for reduced motion + scroll-triggered reveals.',
      'Glassmorphism tokens, sticky judge-mode shortcuts, and CTA micro-interactions.',
    ],
  },
  {
    title: 'Backend & models',
    status: 'Training in progress',
    description:
      'FastAPI services orchestrate data ingestion, with DL acoustic + transformer language models training on de-identified corpora.',
    bullets: [
      'Mock endpoints keep the demo responsive while models train.',
      'Pipeline supports continual fine-tuning + A/B of inference endpoints.',
      'Audit logging + PHI redaction modules queued for next sprint.',
    ],
  },
  {
    title: 'Deployment pipeline',
    status: 'AWS ready',
    description:
      'CI/CD pushes the frontend to S3 + CloudFront and the backend/models to ECS Fargate with automated health checks.',
    bullets: [
      'GitHub Actions handle lint → test → build → deploy per branch.',
      'Parameter Store + Secrets Manager control runtime credentials.',
      'CloudWatch dashboards monitor latency, queue depth, and GPU utilization.',
    ],
  },
]

const complianceStatus = [
  {
    name: 'HIPAA-ready',
    complete: ['TLS everywhere', 'Role-based access', 'Audit logging scaffold'],
    todo: ['Execute BAA', 'Independent HIPAA assessment'],
  },
  {
    name: 'WCAG 2.1 AA',
    complete: ['High-contrast palette', 'Keyboard navigation', 'Reduced-motion support'],
    todo: ['Manual screen reader audit'],
  },
  {
    name: 'SOC 2 roadmap',
    complete: ['Infrastructure as code', 'Secrets rotation plan'],
    todo: ['Policy documentation', '3rd-party audit scheduling'],
  },
]

export default function TechStack() {
  return (
    <div className="techstack-page">
      <header className="techstack-hero">
        <div>
          <p className="techstack-kicker">VoiceVital architecture poster</p>
          <h1>Everything judges need before the live demo</h1>
          <p>
            This page is the launchpad for the hackathon walkthrough. Review the stack, jump into the patient or doctor
            flows, or explore compliance progress in one click.
          </p>
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
          <h2>Compliance & readiness badges</h2>
          <p>Transparent checklist of what’s done and what remains before full production launch.</p>
        </div>
        <div className="compliance-grid">
          {complianceStatus.map((item) => (
            <article key={item.name} className="compliance-card">
              <h3>{item.name}</h3>
              <div className="compliance-lists">
                <div>
                  <p className="compliance-label">✅ Completed</p>
                  <ul>
                    {item.complete.map((task) => (
                      <li key={task}>{task}</li>
                    ))}
                  </ul>
                </div>
                <div>
                  <p className="compliance-label">🔶 Next steps</p>
                  <ul>
                    {item.todo.map((task) => (
                      <li key={task}>{task}</li>
                    ))}
                  </ul>
                </div>
              </div>
            </article>
          ))}
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



