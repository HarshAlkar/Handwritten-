import { useEffect, useState } from 'react'
import { Loader2 } from 'lucide-react'
import { getDatasets, trainModel } from '../api/client'
import { useExperiment } from '../context/ExperimentContext'

export function TrainPage() {
  const { datasetName, setDatasetName, setDatasets, setLastModel, lastModel } = useExperiment()
  const [algorithm, setAlgorithm] = useState<'random_forest' | 'adaboost'>('random_forest')
  const [nEstimators, setNEstimators] = useState(100)
  const [maxDepth, setMaxDepth] = useState<number | ''>(10)
  const [minSplit, setMinSplit] = useState(2)
  const [minLeaf, setMinLeaf] = useState(1)
  const [learningRate, setLearningRate] = useState(1)
  const [baseDepth, setBaseDepth] = useState(1)
  const [testSize, setTestSize] = useState(0.2)
  const [seed, setSeed] = useState(42)
  const [pending, setPending] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    getDatasets().then(setDatasets).catch(() => undefined)
  }, [setDatasets])

  const onTrain = async () => {
    setPending(true)
    setError(null)
    try {
      const hyperparameters =
        algorithm === 'random_forest'
          ? {
              n_estimators: nEstimators,
              max_depth: maxDepth === '' ? null : maxDepth,
              min_samples_split: minSplit,
              min_samples_leaf: minLeaf,
              max_features: 'sqrt',
              random_state: seed,
            }
          : {
              n_estimators: nEstimators,
              learning_rate: learningRate,
              base_max_depth: baseDepth,
              random_state: seed,
            }
      const result = await trainModel({
        dataset: datasetName,
        algorithm,
        hyperparameters,
        test_size: testSize,
        random_seed: seed,
      })
      setLastModel(result)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Training failed')
    } finally {
      setPending(false)
    }
  }

  const m = lastModel?.metrics

  return (
    <div className="space-y-6">
      <div>
        <h1 className="font-display text-3xl tracking-tight">Model Training</h1>
        <p className="mt-2 text-[var(--muted)]">Configure hyperparameters and train AdaBoost or Random Forest.</p>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <section className="card space-y-4">
          <label className="block text-sm">
            Dataset
            <select
              className="input mt-1"
              value={datasetName}
              onChange={(e) => setDatasetName(e.target.value)}
            >
              <option value="breast_cancer">Breast Cancer Wisconsin</option>
              <option value="iris">Iris Plants</option>
            </select>
          </label>

          <label className="block text-sm">
            Algorithm
            <select
              className="input mt-1"
              value={algorithm}
              onChange={(e) => setAlgorithm(e.target.value as 'random_forest' | 'adaboost')}
            >
              <option value="random_forest">Random Forest</option>
              <option value="adaboost">AdaBoost</option>
            </select>
          </label>

          <label className="block text-sm">
            Estimators / trees: {nEstimators}
            <input
              type="range"
              min={10}
              max={300}
              value={nEstimators}
              onChange={(e) => setNEstimators(Number(e.target.value))}
              className="mt-1 w-full accent-[var(--accent)]"
            />
          </label>

          {algorithm === 'random_forest' ? (
            <>
              <label className="block text-sm">
                Max depth (empty = unlimited)
                <input
                  className="input mt-1"
                  type="number"
                  min={1}
                  value={maxDepth}
                  onChange={(e) => setMaxDepth(e.target.value === '' ? '' : Number(e.target.value))}
                />
              </label>
              <label className="block text-sm">
                Min samples split
                <input className="input mt-1" type="number" min={2} value={minSplit} onChange={(e) => setMinSplit(Number(e.target.value))} />
              </label>
              <label className="block text-sm">
                Min samples leaf
                <input className="input mt-1" type="number" min={1} value={minLeaf} onChange={(e) => setMinLeaf(Number(e.target.value))} />
              </label>
            </>
          ) : (
            <>
              <label className="block text-sm">
                Learning rate: {learningRate}
                <input
                  type="range"
                  min={0.05}
                  max={2}
                  step={0.05}
                  value={learningRate}
                  onChange={(e) => setLearningRate(Number(e.target.value))}
                  className="mt-1 w-full accent-[var(--accent)]"
                />
              </label>
              <label className="block text-sm">
                Weak learner max depth
                <input className="input mt-1" type="number" min={1} value={baseDepth} onChange={(e) => setBaseDepth(Number(e.target.value))} />
              </label>
            </>
          )}

          <label className="block text-sm">
            Test size: {testSize}
            <input
              type="range"
              min={0.1}
              max={0.4}
              step={0.05}
              value={testSize}
              onChange={(e) => setTestSize(Number(e.target.value))}
              className="mt-1 w-full accent-[var(--accent)]"
            />
          </label>
          <label className="block text-sm">
            Random seed
            <input className="input mt-1" type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value))} />
          </label>

          <button type="button" className="btn-primary" disabled={pending} onClick={() => void onTrain()}>
            {pending ? <Loader2 className="animate-spin" size={16} /> : null}
            {pending ? 'Training…' : 'Train model'}
          </button>
          {error && <div className="alert-error">{error}</div>}
        </section>

        <section className="card space-y-3">
          <h2 className="font-medium">Results</h2>
          {!lastModel && <p className="text-sm text-[var(--muted)]">Train a model to see metrics here.</p>}
          {lastModel && m && (
            <>
              <p className="text-sm text-[var(--muted)]">Model ID: <code className="text-[var(--ink)]">{lastModel.model_id}</code></p>
              <div className="grid grid-cols-2 gap-3">
                {[
                  ['Accuracy', m.accuracy],
                  ['Precision', m.precision],
                  ['Recall', m.recall],
                  ['F1', m.f1],
                  ['ROC-AUC', m.roc_auc],
                  ['Train (s)', lastModel.train_seconds],
                ].map(([label, val]) => (
                  <div key={String(label)} className="rounded-lg bg-[var(--surface)] p-3">
                    <p className="text-xs text-[var(--muted)]">{label}</p>
                    <p className="text-lg font-semibold">
                      {val == null ? '—' : typeof val === 'number' && Number(val) <= 1 && Number(val) >= 0 && label !== 'Train (s)'
                        ? `${(Number(val) * 100).toFixed(2)}%`
                        : typeof val === 'number'
                          ? Number(val).toFixed(4)
                          : String(val)}
                    </p>
                  </div>
                ))}
              </div>
              <p className="text-xs text-[var(--muted)]">
                Averaging: {m.average}
                {m.positive_class ? ` · Positive class: ${m.positive_class}` : ''}
              </p>
            </>
          )}
        </section>
      </div>
    </div>
  )
}
