import type { PredictionResult } from '../types'

const API_BASE = import.meta.env.VITE_API_URL || '/api'

export async function predictByUrl(url: string): Promise<PredictionResult> {
  const resp = await fetch(`${API_BASE}/predict-url`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ url }),
  })
  if (!resp.ok) {
    const text = await resp.text()
    throw new Error(text || `API error ${resp.status}`)
  }
  return (await resp.json()) as PredictionResult
}


