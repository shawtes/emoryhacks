import { Link } from 'react-router-dom'
import { memo } from 'react'
import { getPrefetchEvents } from '../../routes/lazyPages'

const heroCards = [
  {
    title: 'Patients & caregivers',
    subtitle: 'Turn bedtime stories into longitudinal biomarkers.',
    points: [
      'Guided prompts and reminders between visits.',
      'Invite a caregiver to co-manage uploads securely.',
      'See AI-backed trendlines before the next appointment.',
    ],
    primary: { label: 'Join as patient', to: '/patient/signup', route: 'PatientSignup' as const },
    secondary: { label: 'Patient login', to: '/patient/login', route: 'PatientLogin' as const },
    pill: 'Home-ready',
  },
  {
    title: 'Clinics & research teams',
    subtitle: 'Stand up reimbursable voice services in days, not months.',
    points: [
      'Record or ingest audio from any device.',
      'View risk triage + explainable tags in one dashboard.',
      'Export white-labeled PDF + push data to the EHR.',
    ],
    primary: { label: 'Partner with VoiceVital', to: '/doctor/signup', route: 'DoctorSignup' as const },
    secondary: { label: 'Doctor portal', to: '/doctor/login', route: 'DoctorLogin' as const },
    pill: 'Clinic-ready',
  },
]

const journeySteps = [
  {
    title: 'Capture the story',
    detail: 'Patients read or tell a custom story; VoiceVital normalizes audio for model input.',
  },
  {
    title: 'Explainable AI scoring',
    detail: 'DL + linguistic biomarkers detect hesitation, vocabulary shifts, and memory gaps.',
  },
  {
    title: 'Clinician workflows',
    detail: 'Neurologists triage risk, add notes, and trigger caregiver updates or new prompts.',
  },
]

const showcaseCards = [
  {
    title: 'Guided prompt experience',
    description: 'Real-time waveform + progress cues keep patients on script while capturing high-fidelity samples.',
    cta: { label: 'Preview patient flow', to: '/patient/signup', route: 'PatientSignup' as const },
  },
  {
    title: 'Doctor dashboard',
    description: 'See AI tags, history, and action cards across your entire panel—ready for billing out of the box.',
    cta: { label: 'Open doctor demo', to: '/doctor/dashboard', route: 'DoctorDashboard' as const },
  },
  {
    title: 'Explainable AI snippet',
    description: 'Shareable “reason codes” show which phrases, pauses, and tones influenced the model output.',
    cta: { label: 'Read tech stack', to: '/tech-stack', route: 'TechStack' as const },
  },
]

const trustSignals = [
  'HIPAA-ready architecture',
  'WCAG 2.1 AA-first design',
  'AWS-native deployment',
  'Clinician-in-the-loop reviews',
]

const ShowcaseCarousel = memo(function ShowcaseCarousel() {
  return (
    <div className="showcase-carousel" data-animate>
      {showcaseCards.map((card) => (
        <article className="showcase-card" key={card.title}>
          <h4>{card.title}</h4>
          <p>{card.description}</p>
          <Link to={card.cta.to} className="btn-secondary" {...getPrefetchEvents(card.cta.route)}>
            {card.cta.label}
          </Link>
        </article>
      ))}
    </div>
  )
})

export default function DeferredHomeSections() {
  return (
    <>
      <section className="hero-card-grid">
        {heroCards.map((card) => (
          <div className="hero-card" key={card.title} data-animate>
            <div className="hero-card-pill">{card.pill}</div>
            <h3>{card.title}</h3>
            <p className="hero-card-subtitle">{card.subtitle}</p>
            <ul>
              {card.points.map((point) => (
                <li key={point}>{point}</li>
              ))}
            </ul>
            <div className="hero-card-actions">
              <Link to={card.primary.to} className="btn-primary" {...getPrefetchEvents(card.primary.route)}>
                {card.primary.label}
              </Link>
              <Link to={card.secondary.to} className="btn-secondary" {...getPrefetchEvents(card.secondary.route)}>
                {card.secondary.label}
              </Link>
            </div>
          </div>
        ))}
      </section>

      <section className="storyline-section" data-animate>
        <div className="storyline-header">
          <h3>From bedtime story to clinical insight in 3 steps</h3>
          <p>VoiceVital keeps patients talking, AI analyzing, and clinicians deciding.</p>
        </div>
        <div className="storyline-steps">
          {journeySteps.map((step, idx) => (
            <div className="storyline-card" key={step.title} data-animate>
              <div className="storyline-index">{idx + 1}</div>
              <h4>{step.title}</h4>
              <p>{step.detail}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="showcase-section" data-animate>
        <div className="showcase-header">
          <h3>See VoiceVital in action</h3>
          <p>Short cards, big signals—built for patients at home and clinics on site.</p>
        </div>
        <ShowcaseCarousel />
      </section>

      <section className="trust-strip" data-animate>
        <div className="trust-marquee" aria-label="VoiceVital trust signals">
          {[...trustSignals, ...trustSignals].map((signal, index) => (
            <div key={`${signal}-${index}`} className="trust-pill">
              {signal}
            </div>
          ))}
        </div>
      </section>

      <section className="workflow-section" data-animate>
        <div className="workflow-container">
          <h3 className="section-title">How VoiceVital flows through the visit</h3>
          <div className="workflow-steps">
            <div className="workflow-step" data-animate>
              <div className="step-number">1</div>
              <h4>Patient snapshot</h4>
              <p>VoiceVital syncs demographics + caregiver data straight from the patient dashboard.</p>
            </div>
            <div className="workflow-arrow" aria-hidden="true">
              →
            </div>
            <div className="workflow-step" data-animate>
              <div className="step-number">2</div>
              <h4>Audio assessment</h4>
              <p>Guided mic tips focus solely on capture—no duplicate patient questions mid-story.</p>
            </div>
            <div className="workflow-arrow" aria-hidden="true">
              →
            </div>
            <div className="workflow-step" data-animate>
              <div className="step-number">3</div>
              <h4>Analysis & results</h4>
              <p>AI flags risk, highlights quotes, and explains every decision before handoff.</p>
            </div>
            <div className="workflow-arrow" aria-hidden="true">
              →
            </div>
            <div className="workflow-step" data-animate>
              <div className="step-number">4</div>
              <h4>Clinical report</h4>
              <p>Share VoiceVital summaries with the care team instantly.</p>
            </div>
          </div>
        </div>
      </section>

      <section className="cta-section" data-animate>
        <div className="cta-content">
          <h3>VoiceVital is ready for your care network.</h3>
          <p>Spin up the patient flow, open the doctor dashboard, or inspect the tech stack in minutes.</p>
          <div className="cta-buttons">
            <Link to="/patient/signup" className="btn-cta" {...getPrefetchEvents('PatientSignup')}>
              Launch patient demo
            </Link>
            <Link to="/tech-stack" className="btn-cta-secondary" {...getPrefetchEvents('TechStack')}>
              View tech stack
            </Link>
          </div>
        </div>
      </section>
    </>
  )
}


