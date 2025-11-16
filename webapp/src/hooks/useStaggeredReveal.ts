import { useEffect } from 'react'
import usePrefersReducedMotion from './usePrefersReducedMotion'

interface UseStaggeredRevealOptions {
  selector?: string
  threshold?: number
}

export default function useStaggeredReveal(options: UseStaggeredRevealOptions = {}) {
  const { selector = '[data-animate]', threshold = 0.25 } = options
  const prefersReducedMotion = usePrefersReducedMotion()

  useEffect(() => {
    const elements = Array.from(document.querySelectorAll<HTMLElement>(selector))

    if (!elements.length) {
      return
    }

    if (prefersReducedMotion) {
      elements.forEach((el) => el.classList.add('in-view'))
      return
    }

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add('in-view')
            observer.unobserve(entry.target)
          }
        })
      },
      { threshold }
    )

    elements.forEach((el, index) => {
      el.style.setProperty('--reveal-delay', `${index * 70}ms`)
      observer.observe(el)
    })

    return () => observer.disconnect()
  }, [prefersReducedMotion, selector, threshold])
}



