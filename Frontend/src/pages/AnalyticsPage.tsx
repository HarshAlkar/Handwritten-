import { useEffect, useState } from 'react'
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { Loader2 } from 'lucide-react'
import { getMetrics } from '../api/client'
import type { MetricsResponse } from '../types/api'

function MetricCard({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="rounded-xl border border-[var(--border)] bg-white p-4 shadow-sm">
      <p className="text-sm text-[var(--muted)]">{label}</p>
      <p className="mt-1 text-2xl font-semibold tracking-tight">{value}</p>
      {hint && <p className="mt-1 text-xs text-[var(--muted)]">{hint}</p>}
    </div>
  )
}

export function AnalyticsPage() {
  const [data, setData] = useState<MetricsResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let cancelled = false
    ;(async () => {
      try {
        const res = await getMetrics()
        if (!cancelled) setData(res)
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : 'Failed to load metrics')
      } finally {
        if (!cancelled) setLoading(false)
      }
    })()
    return () => {
      cancelled = true
    }
  }, [])

  const history = data?.history?.history
  const chartData =
    history?.loss?.map((_, i) => ({
      epoch: i + 1,
      accuracy: (history.accuracy?.[i] ?? 0) * 100,
      val_accuracy: (history.val_accuracy?.[i] ?? 0) * 100,
      loss: history.loss?.[i] ?? 0,
      val_loss: history.val_loss?.[i] ?? 0,
    })) ?? []

  const evaluation = data?.evaluation
  const cm = evaluation?.confusion_matrix

  return (
    <div className="space-y-8">
      <div>
        <h1 className="font-display text-3xl tracking-tight">Model Analytics</h1>
        <p className="mt-2 text-[var(--muted)]">
          Real training curves and held-out test metrics from the Hugging Face MNIST evaluation.
        </p>
      </div>

      {loading && (
        <div className="flex items-center gap-2 text-[var(--muted)]">
          <Loader2 className="animate-spin" size={18} />
          Loading metrics…
        </div>
      )}
      {error && (
        <div role="alert" className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-800">
          {error}
        </div>
      )}
      {data && !data.available && (
        <p className="text-[var(--muted)]">{data.message}</p>
      )}

      {evaluation && (
        <section className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <MetricCard
            label="Test accuracy"
            value={`${((evaluation.test_accuracy ?? 0) * 100).toFixed(2)}%`}
            hint="Held-out test set"
          />
          <MetricCard
            label="Precision (macro)"
            value={`${((evaluation.precision_macro ?? 0) * 100).toFixed(2)}%`}
          />
          <MetricCard
            label="Recall (macro)"
            value={`${((evaluation.recall_macro ?? 0) * 100).toFixed(2)}%`}
          />
          <MetricCard
            label="F1 (macro)"
            value={`${((evaluation.f1_macro ?? 0) * 100).toFixed(2)}%`}
          />
        </section>
      )}

      {chartData.length > 0 && (
        <section className="grid gap-6 lg:grid-cols-2">
          <div className="rounded-2xl border border-[var(--border)] bg-white p-4 shadow-sm">
            <h2 className="mb-3 font-medium">Accuracy (train vs validation)</h2>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis dataKey="epoch" tick={{ fontSize: 12 }} />
                  <YAxis domain={[80, 100]} tick={{ fontSize: 12 }} tickFormatter={(v) => `${v}%`} />
                  <Tooltip formatter={(v) => `${Number(v).toFixed(2)}%`} />
                  <Legend />
                  <Line type="monotone" dataKey="accuracy" name="Train" stroke="#0f766e" strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="val_accuracy" name="Validation" stroke="#b45309" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
          <div className="rounded-2xl border border-[var(--border)] bg-white p-4 shadow-sm">
            <h2 className="mb-3 font-medium">Loss (train vs validation)</h2>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis dataKey="epoch" tick={{ fontSize: 12 }} />
                  <YAxis tick={{ fontSize: 12 }} />
                  <Tooltip formatter={(v) => Number(v).toFixed(4)} />
                  <Legend />
                  <Line type="monotone" dataKey="loss" name="Train" stroke="#0f766e" strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="val_loss" name="Validation" stroke="#b45309" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </section>
      )}

      {cm && (
        <section className="rounded-2xl border border-[var(--border)] bg-white p-4 shadow-sm overflow-x-auto">
          <h2 className="mb-2 font-medium">Confusion matrix (test set)</h2>
          <p className="mb-4 text-sm text-[var(--muted)]">
            Rows = true label, columns = predicted label. Darker cells indicate more samples.
          </p>
          <table className="mx-auto border-collapse text-center text-xs sm:text-sm">
            <thead>
              <tr>
                <th className="p-1 text-[var(--muted)]" />
                {cm[0]?.map((_, j) => (
                  <th key={j} className="p-1 font-medium text-[var(--muted)]">
                    {j}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {cm.map((row, i) => {
                const max = Math.max(...row)
                return (
                  <tr key={i}>
                    <th className="p-1 font-medium text-[var(--muted)]">{i}</th>
                    {row.map((val, j) => {
                      const intensity = max > 0 ? val / max : 0
                      const bg = `rgba(15, 118, 110, ${0.08 + intensity * 0.75})`
                      const color = intensity > 0.55 ? '#fff' : '#111827'
                      return (
                        <td
                          key={j}
                          className="min-w-8 p-1.5 font-mono"
                          style={{ backgroundColor: bg, color }}
                          title={`True ${i} → Pred ${j}: ${val}`}
                        >
                          {val}
                        </td>
                      )
                    })}
                  </tr>
                )
              })}
            </tbody>
          </table>
        </section>
      )}

      {data?.history?.split_sizes && (
        <p className="text-sm text-[var(--muted)]">
          Dataset: {data.history.dataset ?? 'ylecun/mnist'} · Train{' '}
          {data.history.split_sizes.train?.toLocaleString()} · Validation{' '}
          {data.history.split_sizes.validation?.toLocaleString()} · Test{' '}
          {data.history.split_sizes.test?.toLocaleString()} · Epochs trained{' '}
          {data.history.epochs_trained ?? '—'}
        </p>
      )}
    </div>
  )
}
