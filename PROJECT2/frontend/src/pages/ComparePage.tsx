import { useState } from 'react'
import { Loader2 } from 'lucide-react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { compareModels } from '../api/client'
import { useExperiment } from '../context/ExperimentContext'

function Confusion({ matrix, labels }: { matrix: number[][]; labels: string[] }) {
  const max = Math.max(...matrix.flat(), 1)
  return (
    <table className="text-center text-xs">
      <thead>
        <tr>
          <th />
          {labels.map((l) => (
            <th key={l} className="p-1 text-[var(--muted)]">
              {l}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {matrix.map((row, i) => (
          <tr key={i}>
            <th className="p-1 text-[var(--muted)]">{labels[i]}</th>
            {row.map((v, j) => (
              <td
                key={j}
                className="min-w-8 p-1.5 font-mono"
                style={{
                  background: `rgba(15,118,110,${0.1 + (v / max) * 0.7})`,
                  color: v / max > 0.55 ? '#fff' : '#111',
                }}
              >
                {v}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  )
}

export function ComparePage() {
  const { datasetName, setDatasetName, comparison, setComparison, setLastModel } = useExperiment()
  const [pending, setPending] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const run = async () => {
    setPending(true)
    setError(null)
    try {
      const res = await compareModels({
        dataset: datasetName,
        random_forest: { n_estimators: 100, max_depth: 10, random_state: 42 },
        adaboost: { n_estimators: 50, learning_rate: 1.0, base_max_depth: 1, random_state: 42 },
        random_seed: 42,
        test_size: 0.2,
      })
      setComparison(res)
      setLastModel(res.models.random_forest)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Compare failed')
    } finally {
      setPending(false)
    }
  }

  const rf = comparison?.models.random_forest
  const ada = comparison?.models.adaboost
  const metricChart =
    rf && ada
      ? [
          { metric: 'Accuracy', RandomForest: rf.metrics.accuracy * 100, AdaBoost: ada.metrics.accuracy * 100 },
          { metric: 'Precision', RandomForest: rf.metrics.precision * 100, AdaBoost: ada.metrics.precision * 100 },
          { metric: 'Recall', RandomForest: rf.metrics.recall * 100, AdaBoost: ada.metrics.recall * 100 },
          { metric: 'F1', RandomForest: rf.metrics.f1 * 100, AdaBoost: ada.metrics.f1 * 100 },
        ]
      : []

  const rocData = () => {
    if (!rf?.metrics.roc_curve || !ada?.metrics.roc_curve) return []
    const n = Math.min(rf.metrics.roc_curve.fpr.length, ada.metrics.roc_curve.fpr.length)
    return Array.from({ length: n }, (_, i) => ({
      fpr: rf.metrics.roc_curve!.fpr[i],
      rf: rf.metrics.roc_curve!.tpr[i],
      ada: ada.metrics.roc_curve!.tpr[i],
    }))
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="font-display text-3xl tracking-tight">Model Comparison</h1>
        <p className="mt-2 text-[var(--muted)]">
          Both algorithms train on the same stratified split. Results depend on data and hyperparameters —
          neither is universally better.
        </p>
      </div>

      <div className="flex flex-wrap items-end gap-3">
        <label className="text-sm">
          Dataset
          <select className="input mt-1" value={datasetName} onChange={(e) => setDatasetName(e.target.value)}>
            <option value="breast_cancer">Breast Cancer</option>
            <option value="iris">Iris</option>
          </select>
        </label>
        <button type="button" className="btn-primary" disabled={pending} onClick={() => void run()}>
          {pending ? <Loader2 className="animate-spin" size={16} /> : null}
          Compare on same split
        </button>
      </div>
      {error && <div className="alert-error">{error}</div>}

      {comparison && rf && ada && (
        <>
          <section className="card">
            <h2 className="mb-3 font-medium">Metrics side-by-side</h2>
            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={metricChart}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="metric" />
                  <YAxis domain={[0, 100]} tickFormatter={(v) => `${v}%`} />
                  <Tooltip formatter={(v) => `${Number(v).toFixed(2)}%`} />
                  <Legend />
                  <Bar dataKey="RandomForest" fill="#0f766e" />
                  <Bar dataKey="AdaBoost" fill="#b45309" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </section>

          <section className="grid gap-4 sm:grid-cols-2">
            {[
              ['Random Forest train (s)', rf.train_seconds],
              ['AdaBoost train (s)', ada.train_seconds],
              ['RF predict (s)', rf.metrics.predict_seconds],
              ['AdaBoost predict (s)', ada.metrics.predict_seconds],
              ['RF ROC-AUC', rf.metrics.roc_auc],
              ['AdaBoost ROC-AUC', ada.metrics.roc_auc],
            ].map(([label, val]) => (
              <div key={String(label)} className="card">
                <p className="text-sm text-[var(--muted)]">{label}</p>
                <p className="text-xl font-semibold">{val == null ? '—' : Number(val).toFixed(4)}</p>
              </div>
            ))}
          </section>

          {rocData().length > 0 && (
            <section className="card">
              <h2 className="mb-3 font-medium">ROC curves (binary)</h2>
              <div className="h-72">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={rocData()}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="fpr" type="number" domain={[0, 1]} />
                    <YAxis domain={[0, 1]} />
                    <Tooltip />
                    <Legend />
                    <Line dataKey="rf" name="Random Forest" stroke="#0f766e" dot={false} />
                    <Line dataKey="ada" name="AdaBoost" stroke="#b45309" dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </section>
          )}

          <section className="grid gap-4 lg:grid-cols-2">
            <div className="card overflow-x-auto">
              <h2 className="mb-3 font-medium">Random Forest confusion</h2>
              <Confusion matrix={rf.metrics.confusion_matrix} labels={comparison.target_names} />
            </div>
            <div className="card overflow-x-auto">
              <h2 className="mb-3 font-medium">AdaBoost confusion</h2>
              <Confusion matrix={ada.metrics.confusion_matrix} labels={comparison.target_names} />
            </div>
          </section>
        </>
      )}
    </div>
  )
}
