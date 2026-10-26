import type {
  HealthResponse,
  MetricsResponse,
  ModelInfoResponse,
  PredictResponse,
} from '../types/api'

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'

async function parseError(res: Response): Promise<string> {
  try {
    const body = (await res.json()) as { detail?: string }
    if (body.detail) return body.detail
  } catch {
    /* ignore */
  }
  return `Request failed (${res.status})`
}

export async function getHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE}/api/health`)
  if (!res.ok) throw new Error(await parseError(res))
  return res.json() as Promise<HealthResponse>
}

export async function getModelInfo(): Promise<ModelInfoResponse> {
  const res = await fetch(`${API_BASE}/api/model-info`)
  if (!res.ok) throw new Error(await parseError(res))
  return res.json() as Promise<ModelInfoResponse>
}

export async function getMetrics(): Promise<MetricsResponse> {
  const res = await fetch(`${API_BASE}/api/metrics`)
  if (!res.ok) throw new Error(await parseError(res))
  return res.json() as Promise<MetricsResponse>
}

export async function predictFromBlob(blob: Blob, filename = 'digit.png'): Promise<PredictResponse> {
  const form = new FormData()
  form.append('file', blob, filename)
  const res = await fetch(`${API_BASE}/api/predict`, {
    method: 'POST',
    body: form,
  })
  if (!res.ok) throw new Error(await parseError(res))
  return res.json() as Promise<PredictResponse>
}

export async function predictFromBase64(imageBase64: string): Promise<PredictResponse> {
  const form = new FormData()
  form.append('image_base64', imageBase64)
  const res = await fetch(`${API_BASE}/api/predict`, {
    method: 'POST',
    body: form,
  })
  if (!res.ok) throw new Error(await parseError(res))
  return res.json() as Promise<PredictResponse>
}

export { API_BASE }
