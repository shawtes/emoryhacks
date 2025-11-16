import type { Metric } from 'web-vitals'
import { onCLS, onINP, onLCP, onTTFB } from 'web-vitals'

const WEB_VITALS_ENDPOINT = import.meta.env.VITE_WEB_VITALS_ENDPOINT || '/api/metrics/web-vitals'

function sendMetric(metric: Metric) {
  const payload = JSON.stringify({
    ...metric,
    href: window.location.href,
    pathname: window.location.pathname,
    timestamp: Date.now(),
  })

  if ('sendBeacon' in navigator) {
    navigator.sendBeacon(WEB_VITALS_ENDPOINT, payload)
    return
  }

  fetch(WEB_VITALS_ENDPOINT, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: payload,
    keepalive: true,
  }).catch(() => {
    if (import.meta.env.DEV) {
      console.debug('[web-vitals]', metric)
    }
  })
}

export function reportWebVitals() {
  if (import.meta.env.DEV) {
    onCLS((metric) => console.debug('[CLS]', metric))
    onINP((metric) => console.debug('[INP]', metric))
    onLCP((metric) => console.debug('[LCP]', metric))
    onTTFB((metric) => console.debug('[TTFB]', metric))
    return
  }

  onCLS(sendMetric)
  onINP(sendMetric)
  onLCP(sendMetric)
  onTTFB(sendMetric)
}


