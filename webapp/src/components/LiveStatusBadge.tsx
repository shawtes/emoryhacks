import { useEffect, useState } from 'react'

const STATUS_OPTIONS = [
  { label: 'API Latency', value: '142 ms', color: 'green' },
  { label: 'Uptime', value: '99.98%', color: 'green' },
  { label: 'Model refresh', value: '3 hrs ago', color: 'yellow' },
]

export default function LiveStatusBadge() {
  const [index, setIndex] = useState(0)

  useEffect(() => {
    const id = window.setInterval(() => {
      setIndex((prev) => (prev + 1) % STATUS_OPTIONS.length)
    }, 4000)
    return () => window.clearInterval(id)
  }, [])

  const status = STATUS_OPTIONS[index]

  return (
    <span className="live-status-badge" aria-live="polite">
      <span className={`live-status-indicator ${status.color}`} />
      <span className="live-status-label">{status.label}</span>
      <span className="live-status-value">{status.value}</span>
    </span>
  )
}



