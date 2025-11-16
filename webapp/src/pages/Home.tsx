import type { CSSProperties, MouseEvent } from 'react'
import { memo, useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import LiveStatusBadge from '../components/LiveStatusBadge'
import usePrefersReducedMotion from '../hooks/usePrefersReducedMotion'
import useStaggeredReveal from '../hooks/useStaggeredReveal'
import { useInViewCountUp } from '../hooks/useInViewCountUp'
import { getPrefetchEvents } from '../routes/lazyPages'
import DeferredHomeSections from './home/DeferredSections'
import { useFirebase } from '../context/FirebaseContext'
import type { Persona } from '../lib/firebase'

const heroWaveBars = [
  60, 90, 120, 150, 180, 210, 180, 150, 120, 90, 60, 80, 140, 200, 220, 200, 140, 80, 60, 90, 120, 150, 180, 210, 180,
  150, 120, 90, 60, 100, 140, 180, 230, 260, 230, 180, 140, 100, 60, 90, 130, 170, 205, 230, 205, 170, 130, 90, 60, 80,
  120, 160, 190, 210, 190, 160, 120, 80, 60, 75, 110, 150, 190, 220, 190, 150, 110, 75, 60, 95, 135, 175, 215, 245,
  215, 175, 135, 95,
]

type HeroTiltStyle = CSSProperties & {
  '--hero-tilt-x'?: string
  '--hero-tilt-y'?: string
}

interface HeroStat {
  label: string
  value: number
  suffix?: string
  helper: string
  decimals?: number
}

const heroStats: HeroStat[] = [
  { label: 'Words analyzed', value: 1.2, suffix: 'M', helper: 'Rolling 30 days', decimals: 1 },
  { label: 'Care networks onboarded', value: 82, suffix: '+', helper: 'Across 9 states' },
  { label: 'Weekly assessments', value: 340, helper: 'Avg. per network' },
]

const HeroStatCard = memo(function HeroStatCard({ label, value, suffix = '', helper, decimals = 0 }: HeroStat) {
  const { ref, value: animatedValue } = useInViewCountUp(value, { duration: 2200 })
  const formattedValue =
    decimals > 0
      ? animatedValue.toLocaleString(undefined, {
          minimumFractionDigits: decimals,
          maximumFractionDigits: decimals,
        })
      : Math.round(animatedValue).toLocaleString()

  return (
    <article ref={ref} className="hero-stat-card" data-animate>
      <p className="hero-stat-label">{label}</p>
      <p className="hero-stat-value">
        {formattedValue}
        {suffix && <span className="hero-stat-suffix">{suffix}</span>}
      </p>
      <span className="hero-stat-helper">{helper}</span>
    </article>
  )
})

export default function Home() {
  const [isHeaderPinned, setIsHeaderPinned] = useState(false)
  const [heroTilt, setHeroTilt] = useState({ x: 0, y: 0 })
  const heroRef = useRef<HTMLDivElement | null>(null)
  const prefersReducedMotion = usePrefersReducedMotion()
  const { selectPersona } = useFirebase()

  useStaggeredReveal()

  useEffect(() => {
    const handleScroll = () => {
      setIsHeaderPinned(window.scrollY > 24)
    }

    handleScroll()
    window.addEventListener('scroll', handleScroll, { passive: true })

    return () => window.removeEventListener('scroll', handleScroll)
  }, [])

  const handleHeroPointer = (event: MouseEvent<HTMLDivElement>) => {
    if (prefersReducedMotion || !heroRef.current) {
      return
    }

    const rect = heroRef.current.getBoundingClientRect()
    const relativeX = ((event.clientX - rect.left) / rect.width - 0.5) * 18
    const relativeY = ((event.clientY - rect.top) / rect.height - 0.5) * 18
    setHeroTilt({ x: relativeX, y: relativeY })
  }

  const resetHeroPointer = () => {
    setHeroTilt({ x: 0, y: 0 })
  }

  const heroStyle: HeroTiltStyle | undefined = prefersReducedMotion
    ? undefined
    : {
        '--hero-tilt-x': `${heroTilt.y}deg`,
        '--hero-tilt-y': `${heroTilt.x}deg`,
      }

  const handlePersonaLink = (persona: Persona) => () => {
    void selectPersona(persona)
  }

  return (
    <div className="home-page voicevital">
      <header className={`home-header ${isHeaderPinned ? 'home-header--pinned' : ''}`}>
        <div className="home-header-content" data-animate>
          <div className="logo-section">
            <h1 className="logo">🔊 VoiceVital</h1>
            <p className="tagline">AI speech intelligence for dementia screening</p>
          </div>
          <nav className="home-nav">
            <div className="home-nav-group">
              <span className="home-nav-label">Patients</span>
              <div className="home-nav-actions">
                <Link
                  to="/patient/login"
                  className="btn-nav-secondary"
                  {...getPrefetchEvents('PatientLogin')}
                  onClick={handlePersonaLink('patient')}
                >
                  Patient Login
                </Link>
                <Link
                  to="/patient/signup"
                  className="btn-nav-primary btn-primary"
                  {...getPrefetchEvents('PatientSignup')}
                  onClick={handlePersonaLink('patient')}
                >
                  Patient Register
                </Link>
              </div>
            </div>
            <div className="home-nav-group">
              <span className="home-nav-label">Clinics</span>
              <div className="home-nav-actions">
                <Link
                  to="/doctor/login"
                  className="btn-nav-secondary"
                  {...getPrefetchEvents('DoctorLogin')}
                  onClick={handlePersonaLink('doctor')}
                >
                  Doctor Login
                </Link>
                <Link
                  to="/doctor/signup"
                  className="btn-nav-primary btn-primary"
                  {...getPrefetchEvents('DoctorSignup')}
                  onClick={handlePersonaLink('doctor')}
                >
                  Partner With Us
                </Link>
              </div>
            </div>
            <Link to="/tech-stack" className="btn-nav-secondary tech-stack-link" {...getPrefetchEvents('TechStack')}>
              Tech Stack
            </Link>
          </nav>
        </div>
      </header>

      <main className="home-main">
        <section
          className="hero-section hero-bg"
          ref={heroRef}
          style={heroStyle}
          onMouseMove={handleHeroPointer}
          onMouseLeave={resetHeroPointer}
        >
          <div className="hero-content" data-animate>
            <p className="hero-kicker">VoiceVital™</p>
            <h2 className="hero-title">The voice intelligence layer for dementia care</h2>
            <p className="hero-subtitle">
              VoiceVital transforms everyday speech into measurable biomarkers so patients, caregivers, and doctors
              share the same signal—no new hardware, no extra paperwork.
            </p>
            <div className="hero-meta">
              <LiveStatusBadge />
              <span className="hero-mission">Built with neurologists, caregivers, and ML researchers.</span>
            </div>
            <div className="hero-actions">
              <Link
                to="/patient/signup"
                className="btn-hero-primary"
                {...getPrefetchEvents('PatientSignup')}
                onClick={handlePersonaLink('patient')}
              >
                Start patient journey
              </Link>
              <Link
                to="/doctor/signup"
                className="btn-hero-secondary"
                {...getPrefetchEvents('DoctorSignup')}
                onClick={handlePersonaLink('doctor')}
              >
                Launch clinic demo
              </Link>
              <Link to="/tech-stack" className="btn-hero-secondary ghost" {...getPrefetchEvents('TechStack')}>
                View tech stack
              </Link>
            </div>
            <div className="hero-stat-grid">
              {heroStats.map((stat) => (
                <HeroStatCard key={stat.label} {...stat} />
              ))}
            </div>
          </div>
          <div className="hero-wavefield" aria-hidden="true">
            {heroWaveBars.map((height, idx) => (
              <span
                key={idx}
                className="hero-wave-bar"
                style={
                  {
                    '--bar-delay': `${idx * 70}ms`,
                    '--bar-height': `${height}px`,
                  } as CSSProperties
                }
              />
            ))}
          </div>
        </section>

        <DeferredHomeSections />
      </main>

      <footer className="home-footer">
        <div className="footer-content">
          <div className="footer-disclaimer">
            <p>
              <strong>Research Use Only</strong> - VoiceVital is a research and screening tool. It is not a diagnostic
              device.
            </p>
          </div>
          <div className="footer-links">
            <a href="#privacy">Privacy Policy</a>
            <a href="#terms">Terms of Service</a>
            <a href="#contact">Contact</a>
          </div>
          <p className="footer-copyright">© {new Date().getFullYear()} VoiceVital. All rights reserved.</p>
        </div>
      </footer>
    </div>
  )
}

