import { createContext, useContext, useMemo, useState, type ReactNode } from 'react'
import type { CompareResponse, DatasetInfo, TrainedModel } from '../types/api'

interface ExperimentState {
  datasetName: string
  setDatasetName: (v: string) => void
  datasets: DatasetInfo[]
  setDatasets: (v: DatasetInfo[]) => void
  lastModel: TrainedModel | null
  setLastModel: (v: TrainedModel | null) => void
  comparison: CompareResponse | null
  setComparison: (v: CompareResponse | null) => void
}

const Ctx = createContext<ExperimentState | null>(null)

export function ExperimentProvider({ children }: { children: ReactNode }) {
  const [datasetName, setDatasetName] = useState('breast_cancer')
  const [datasets, setDatasets] = useState<DatasetInfo[]>([])
  const [lastModel, setLastModel] = useState<TrainedModel | null>(null)
  const [comparison, setComparison] = useState<CompareResponse | null>(null)

  const value = useMemo(
    () => ({
      datasetName,
      setDatasetName,
      datasets,
      setDatasets,
      lastModel,
      setLastModel,
      comparison,
      setComparison,
    }),
    [datasetName, datasets, lastModel, comparison],
  )

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}

export function useExperiment() {
  const ctx = useContext(Ctx)
  if (!ctx) throw new Error('useExperiment must be used within ExperimentProvider')
  return ctx
}
