import type {
  CompareResponse,
  DatasetInfo,
  PredictResponse,
  TrainedModel,
} from '../types/api'

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8001'

async function parseError(res: Response): Promise<string> {
  try {
    const body = (await res.json()) as { detail?: string }
    if (body.detail) return body.detail
  } catch {
    /* ignore */
  }
  return `Request failed (${res.status})`
}

export async function getHealth() {
  const res = await fetch(`${API_BASE}/api/health`)
  if (!res.ok) throw new Error(await parseError(res))
  return res.json() as Promise<{ status: string }>
}

export async function getDatasets(): Promise<DatasetInfo[]> {
  const res = await fetch(`${API_BASE}/api/datasets`)
  if (!res.ok) throw new Error(await parseError(res))
  const body = (await res.json()) as { datasets: DatasetInfo[] }
  return body.datasets
}

export async function getDatasetSample(name: string, limit = 8) {
  const res = await fetch(`${API_BASE}/api/datasets/${name}/sample?limit=${limit}`)
  if (!res.ok) throw new Error(await parseError(res))
  return res.json() as Promise<{ columns: string[]; rows: Record<string, unknown>[] }>
}

export async function trainModel(payload: {
  dataset: string
  algorithm: 'random_forest' | 'adaboost'
  hyperparameters?: Record<string, unknown>
  test_size?: number
  random_seed?: number
}): Promise<TrainedModel> {
  const res = await fetch(`${API_BASE}/api/train`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!res.ok) throw new Error(await parseError(res))
  return res.json() as Promise<TrainedModel>
}

export async function compareModels(payload: {
  dataset: string
  random_forest?: Record<string, unknown>
  adaboost?: Record<string, unknown>
  test_size?: number
  random_seed?: number
}): Promise<CompareResponse> {
  const res = await fetch(`${API_BASE}/api/compare`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!res.ok) throw new Error(await parseError(res))
  return res.json() as Promise<CompareResponse>
}

export async function predict(modelId: string, features: Record<string, number>): Promise<PredictResponse> {
  const res = await fetch(`${API_BASE}/api/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model_id: modelId, features }),
  })
  if (!res.ok) throw new Error(await parseError(res))
  return res.json() as Promise<PredictResponse>
}

export async function listModels(): Promise<TrainedModel[]> {
  const res = await fetch(`${API_BASE}/api/models`)
  if (!res.ok) throw new Error(await parseError(res))
  const body = (await res.json()) as { models: TrainedModel[] }
  return body.models
}

export { API_BASE }
