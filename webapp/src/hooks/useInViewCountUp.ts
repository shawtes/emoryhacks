import { useEffect, useRef, useState } from 'react'
import usePrefersReducedMotion from './usePrefersReducedMotion'

interface CountUpOptions {
  duration?: number
}

export function useInViewCountUp(targetValue: number, options: CountUpOptions = {}) {
  const { duration = 1800 } = options
  const prefersReducedMotion = usePrefersReducedMotion()
  const ref = useRef<HTMLElement | null>(null)
  const [value, setValue] = useState(prefersReducedMotion ? targetValue : 0)

  useEffect(() => {
    const node = ref.current

    if (!node) {
      return
    }

    if (prefersReducedMotion) {
      setValue(targetValue)
      return
    }

    let frameId: number | null = null
    let startTime: number | null = null

    const animate = (timestamp: number) => {
      if (startTime === null) {
        startTime = timestamp
      }

      const progress = Math.min((timestamp - startTime) / duration, 1)
      setValue(targetValue * progress)

      if (progress < 1) {
        frameId = window.requestAnimationFrame(animate)
      }
    }

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            observer.unobserve(entry.target)
            frameId = window.requestAnimationFrame(animate)
          }
        })
      },
      { threshold: 0.5 }
    )

    observer.observe(node)

    return () => {
      observer.disconnect()
      if (frameId) {
        window.cancelAnimationFrame(frameId)
      }
    }
  }, [duration, prefersReducedMotion, targetValue])

  return { ref, value }
}



