import { useEffect, useState } from 'react'
import { Loader2 } from 'lucide-react'
import { Bar, BarChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { getDatasets, listModels, predict } from '../api/client'
import { useExperiment } from '../context/ExperimentContext'
import type { PredictResponse, TrainedModel } from '../types/api'

export function PredictPage() {
  const { datasets, setDatasets, lastModel, datasetName } = useExperiment()
  const [models, setModels] = useState<TrainedModel[]>([])
  const [modelId, setModelId] = useState('')
  const [features, setFeatures] = useState<Record<string, string>>({})
  const [result, setResult] = useState<PredictResponse | null>(null)
  const [pending, setPending] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    void getDatasets().then(setDatasets)
    void listModels().then((m) => {
      setModels(m)
      const preferred = lastModel?.model_id ?? m[0]?.model_id ?? ''
      setModelId(preferred)
    })
  }, [setDatasets, lastModel])

  const selected = models.find((m) => m.model_id === modelId) ?? lastModel
  const dataset = datasets.find((d) => d.name === (selected?.dataset ?? datasetName))

  useEffect(() => {
    if (!selected) return
    const init: Record<string, string> = {}
    for (const name of selected.feature_names) {
      const range = dataset?.feature_ranges?.[name]
      init[name] = range ? String(Number(range.mean.toFixed(4))) : '0'
    }
    setFeatures(init)
  }, [selected?.model_id, dataset])

  const onPredict = async () => {
    if (!modelId) {
      setError('Train a model first.')
      return
    }
    setPending(true)
    setError(null)
    try {
      const numeric: Record<string, number> = {}
      for (const [k, v] of Object.entries(features)) {
        const n = Number(v)
        if (Number.isNaN(n)) throw new Error(`Invalid value for ${k}`)
        numeric[k] = n
      }
      setResult(await predict(modelId, numeric))
    } catch (e) {
      setResult(null)
      setError(e instanceof Error ? e.message : 'Prediction failed')
    } finally {
      setPending(false)
    }
  }

  const loadExample = () => {
    if (!dataset || !selected) return
    const init: Record<string, string> = {}
    selected.feature_names.forEach((name, i) => {
      // Use first sample mean-ish: feature range midpoint as demo
      const r = dataset.feature_ranges[name]
      init[name] = r ? String(Number(((r.min + r.max) / 2).toFixed(4))) : '0'
      void i
    })
    setFeatures(init)
  }

  const probData = result?.probabilities
    ? Object.entries(result.probabilities).map(([name, p]) => ({ name, probability: p * 100 }))
    : []

  return (
    <div className="space-y-6">
      <div>
        <h1 className="font-display text-3xl tracking-tight">Prediction Playground</h1>
        <p className="mt-2 text-[var(--muted)]">Enter feature values and classify with a trained model.</p>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <section className="card space-y-3">
          <label className="block text-sm">
            Trained model
            <select className="input mt-1" value={modelId} onChange={(e) => setModelId(e.target.value)}>
              {models.length === 0 && <option value="">No models yet</option>}
              {models.map((m) => (
                <option key={m.model_id} value={m.model_id}>
                  {m.algorithm} · {m.dataset} · {(m.metrics?.accuracy * 100).toFixed(1)}%
                </option>
              ))}
            </select>
          </label>

          <div className="flex gap-2">
            <button type="button" className="btn-secondary" onClick={loadExample}>
              Fill example values
            </button>
            <button type="button" className="btn-primary" disabled={pending} onClick={() => void onPredict()}>
              {pending ? <Loader2 className="animate-spin" size={16} /> : null}
              Predict
            </button>
          </div>

          <div className="max-h-[420px] space-y-2 overflow-y-auto pr-1">
            {selected?.feature_names.map((name) => (
              <label key={name} className="block text-xs">
                {name}
                <input
                  className="input mt-1"
                  value={features[name] ?? ''}
                  onChange={(e) => setFeatures((prev) => ({ ...prev, [name]: e.target.value }))}
                />
              </label>
            ))}
          </div>
          {error && <div className="alert-error">{error}</div>}
        </section>

        <section className="card space-y-4">
          <h2 className="font-medium">Prediction</h2>
          {!result && <p className="text-sm text-[var(--muted)]">Results appear after you predict.</p>}
          {result && (
            <>
              <p className="font-display text-4xl text-[var(--accent)]">{result.predicted_class}</p>
              <p className="text-sm text-[var(--muted)]">
                Confidence:{' '}
                {result.confidence != null ? `${(result.confidence * 100).toFixed(2)}%` : 'n/a'}
              </p>
              {probData.length > 0 && (
                <div className="h-56">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={probData}>
                      <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                      <YAxis tickFormatter={(v) => `${v}%`} />
                      <Tooltip formatter={(v) => `${Number(v).toFixed(2)}%`} />
                      <Bar dataKey="probability" fill="#0f766e" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              )}
            </>
          )}
        </section>
      </div>
    </div>
  )
}
