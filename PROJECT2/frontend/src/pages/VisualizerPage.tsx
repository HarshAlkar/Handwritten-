import { useExperiment } from '../context/ExperimentContext'

export function VisualizerPage() {
  const { lastModel, comparison } = useExperiment()
  const ada = lastModel?.algorithm === 'adaboost' ? lastModel : comparison?.models.adaboost
  const weights = ada?.adaboost_diagnostics?.estimator_weights?.slice(0, 20) ?? []
  const errors = ada?.adaboost_diagnostics?.estimator_errors?.slice(0, 20) ?? []

  return (
    <div className="space-y-8">
      <div>
        <h1 className="font-display text-3xl tracking-tight">Algorithm Visualizer</h1>
        <p className="mt-2 text-[var(--muted)]">
          Explanatory diagrams — not literal internal tree drawings. Real AdaBoost diagnostics appear when available.
        </p>
      </div>

      <section className="card space-y-3">
        <h2 className="font-medium text-[var(--accent)]">Random Forest (bagging)</h2>
        <div className="grid gap-2 sm:grid-cols-3">
          {['Bootstrap sample A → Tree 1', 'Bootstrap sample B → Tree 2', 'Bootstrap sample C → Tree 3'].map(
            (t) => (
              <div key={t} className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-4 text-sm">
                {t}
              </div>
            ),
          )}
        </div>
        <p className="text-center text-sm text-[var(--muted)]">↓ majority vote / probability average ↓</p>
        <div className="rounded-lg border border-[var(--accent)] bg-[var(--accent-soft)] p-4 text-center text-sm font-medium">
          Final ensemble prediction
        </div>
        <p className="text-sm text-[var(--muted)]">
          Each tree sees a random subset of features at splits, reducing correlation between trees.
        </p>
      </section>

      <section className="card space-y-3">
        <h2 className="font-medium text-[var(--accent)]">AdaBoost (boosting)</h2>
        <ol className="space-y-2 text-sm text-[var(--muted)]">
          <li>1. Train a weak learner (shallow decision tree).</li>
          <li>2. Increase weights of misclassified training samples.</li>
          <li>3. Train the next weak learner on the reweighted distribution.</li>
          <li>4. Combine learners with estimator weights into the final classifier.</li>
        </ol>
        {(weights.length > 0 || errors.length > 0) && (
          <div className="grid gap-4 sm:grid-cols-2">
            <div>
              <h3 className="mb-2 text-sm font-medium">Estimator weights (first {weights.length})</h3>
              <ul className="max-h-40 overflow-y-auto font-mono text-xs">
                {weights.map((w, i) => (
                  <li key={i}>
                    #{i + 1}: {w.toFixed(4)}
                  </li>
                ))}
              </ul>
            </div>
            <div>
              <h3 className="mb-2 text-sm font-medium">Estimator errors (first {errors.length})</h3>
              <ul className="max-h-40 overflow-y-auto font-mono text-xs">
                {errors.map((e, i) => (
                  <li key={i}>
                    #{i + 1}: {e.toFixed(4)}
                  </li>
                ))}
              </ul>
            </div>
          </div>
        )}
        {!weights.length && (
          <p className="text-sm text-[var(--muted)]">Train AdaBoost to load real estimator weights/errors.</p>
        )}
      </section>
    </div>
  )
}
