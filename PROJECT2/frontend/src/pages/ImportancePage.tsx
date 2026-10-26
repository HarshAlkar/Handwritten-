import { useMemo, useState } from 'react'
import { Bar, BarChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { useExperiment } from '../context/ExperimentContext'

export function ImportancePage() {
  const { lastModel, comparison } = useExperiment()
  const [topK, setTopK] = useState(12)
  const [source, setSource] = useState<'last' | 'rf' | 'ada'>('last')

  const model =
    source === 'rf'
      ? comparison?.models.random_forest
      : source === 'ada'
        ? comparison?.models.adaboost
        : lastModel ?? comparison?.models.random_forest

  const data = useMemo(() => {
    const rows = model?.feature_importance ?? []
    return rows.slice(0, topK).map((r) => ({
      feature: r.feature.length > 22 ? `${r.feature.slice(0, 20)}…` : r.feature,
      importance: r.importance,
    }))
  }, [model, topK])

  return (
    <div className="space-y-6">
      <div>
        <h1 className="font-display text-3xl tracking-tight">Feature Importance</h1>
        <p className="mt-2 max-w-2xl text-[var(--muted)]">
          Tree-based importance ranks features by impurity reduction during training. It does not prove
          causation and can be biased toward high-cardinality features.
        </p>
      </div>

      <div className="flex flex-wrap gap-3">
        <select className="input" value={source} onChange={(e) => setSource(e.target.value as 'last' | 'rf' | 'ada')}>
          <option value="last">Last trained model</option>
          <option value="rf">Comparison · Random Forest</option>
          <option value="ada">Comparison · AdaBoost</option>
        </select>
        <label className="text-sm">
          Top features: {topK}
          <input
            type="range"
            min={5}
            max={30}
            value={topK}
            onChange={(e) => setTopK(Number(e.target.value))}
            className="ml-2 accent-[var(--accent)]"
          />
        </label>
      </div>

      {!data.length && (
        <p className="text-[var(--muted)]">Train a model or run Compare to view feature importance.</p>
      )}

      {data.length > 0 && (
        <section className="card">
          <div className="h-[420px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={data} layout="vertical" margin={{ left: 24, right: 16 }}>
                <XAxis type="number" />
                <YAxis type="category" dataKey="feature" width={140} tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="importance" fill="#0f766e" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </section>
      )}
    </div>
  )
}
