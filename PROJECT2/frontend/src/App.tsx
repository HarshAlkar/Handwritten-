import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import { Layout } from './components/Layout'
import { ExperimentProvider } from './context/ExperimentContext'
import { OverviewPage } from './pages/OverviewPage'
import { DatasetPage } from './pages/DatasetPage'
import { TrainPage } from './pages/TrainPage'
import { PredictPage } from './pages/PredictPage'
import { ComparePage } from './pages/ComparePage'
import { ImportancePage } from './pages/ImportancePage'
import { VisualizerPage } from './pages/VisualizerPage'
import { DocsPage } from './pages/DocsPage'

export default function App() {
  return (
    <ExperimentProvider>
      <BrowserRouter>
        <Routes>
          <Route element={<Layout />}>
            <Route index element={<OverviewPage />} />
            <Route path="dataset" element={<DatasetPage />} />
            <Route path="train" element={<TrainPage />} />
            <Route path="predict" element={<PredictPage />} />
            <Route path="compare" element={<ComparePage />} />
            <Route path="importance" element={<ImportancePage />} />
            <Route path="visualizer" element={<VisualizerPage />} />
            <Route path="docs" element={<DocsPage />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </ExperimentProvider>
  )
}
