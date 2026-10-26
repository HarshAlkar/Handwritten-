import { useEffect, useState } from 'react'
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { getDatasetSample, getDatasets } from '../api/client'
import { useExperiment } from '../context/ExperimentContext'

export function DatasetPage() {
  const { datasetName, setDatasetName, datasets, setDatasets } = useExperiment()
  const [rows, setRows] = useState<Record<string, unknown>[]>([])
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    getDatasets()
      .then(setDatasets)
      .catch((e: Error) => setError(e.message))
  }, [setDatasets])

  useEffect(() => {
    if (!datasetName) return
    getDatasetSample(datasetName, 8)
      .then((s) => setRows(s.rows))
      .catch((e: Error) => setError(e.message))
  }, [datasetName])

  const current = datasets.find((d) => d.name === datasetName)
  const dist = current
    ? Object.entries(current.class_distribution).map(([name, count]) => ({ name, count }))
    : []

  return (
    <div className="space-y-6">
      <div>
        <h1 className="font-display text-3xl tracking-tight">Dataset Explorer</h1>
        <p className="mt-2 text-[var(--muted)]">Inspect samples, features, and class balance before training.</p>
      </div>

      {error && <div className="alert-error">{error}</div>}

      <div className="flex flex-wrap gap-2">
        {datasets.map((d) => (
          <button
            key={d.name}
            type="button"
            className={d.name === datasetName ? 'btn-primary' : 'btn-secondary'}
            onClick={() => setDatasetName(d.name)}
          >
            {d.display_name}
          </button>
        ))}
      </div>

      {current && (
        <>
          <section className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            {[
              ['Samples', current.n_samples],
              ['Features', current.n_features],
              ['Classes', current.n_classes],
              ['Task', current.task_type],
            ].map(([k, v]) => (
              <div key={String(k)} className="card">
                <p className="text-sm text-[var(--muted)]">{k}</p>
                <p className="mt-1 text-xl font-semibold">{v}</p>
              </div>
            ))}
          </section>

          <p className="text-sm text-[var(--muted)]">{current.description}</p>
          {current.positive_class && (
            <p className="text-sm">
              Binary positive class for metrics: <strong>{current.positive_class}</strong>
            </p>
          )}

          <section className="card">
            <h2 className="mb-3 font-medium">Class distribution</h2>
            <div className="h-56">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={dist}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis dataKey="name" tick={{ fontSize: 12 }} />
                  <YAxis tick={{ fontSize: 12 }} />
                  <Tooltip />
                  <Bar dataKey="count" fill="#0f766e" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </section>

          <section className="card overflow-x-auto">
            <h2 className="mb-3 font-medium">Feature names ({current.feature_names.length})</h2>
            <p className="mb-4 text-xs text-[var(--muted)]">{current.feature_names.join(' · ')}</p>
            <h2 className="mb-2 font-medium">Sample rows</h2>
            <table className="w-full min-w-[640px] text-left text-xs">
              <thead>
                <tr className="border-b border-[var(--border)] text-[var(--muted)]">
                  {rows[0] &&
                    Object.keys(rows[0]).map((c) => (
                      <th key={c} className="p-2 font-medium">
                        {c}
                      </th>
                    ))}
                </tr>
              </thead>
              <tbody>
                {rows.map((row, i) => (
                  <tr key={i} className="border-b border-[var(--border)]">
                    {Object.values(row).map((v, j) => (
                      <td key={j} className="p-2 font-mono">
                        {typeof v === 'number' ? Number(v).toFixed(3) : String(v)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </section>
        </>
      )}
    </div>
  )
}
