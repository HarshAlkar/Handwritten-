import { useRef, useState } from 'react'
import { Eraser, Loader2, Upload, Undo2, Wand2 } from 'lucide-react'
import { DigitCanvas, type DigitCanvasHandle } from '../components/DigitCanvas'
import { ProbabilityChart } from '../components/ProbabilityChart'
import { predictFromBlob } from '../api/client'
import type { PredictResponse } from '../types/api'

export function WorkspacePage() {
  const canvasRef = useRef<DigitCanvasHandle>(null)
  const fileRef = useRef<HTMLInputElement>(null)
  const [brushSize, setBrushSize] = useState(16)
  const [pending, setPending] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<PredictResponse | null>(null)
  const [sourceLabel, setSourceLabel] = useState<string>('canvas')

  const runPredict = async (blob: Blob, label: string) => {
    setPending(true)
    setError(null)
    setSourceLabel(label)
    try {
      const res = await predictFromBlob(blob, label === 'upload' ? 'upload.png' : 'canvas.png')
      setResult(res)
    } catch (err) {
      setResult(null)
      setError(err instanceof Error ? err.message : 'Prediction failed.')
    } finally {
      setPending(false)
    }
  }

  const onPredict = async () => {
    const handle = canvasRef.current
    if (!handle) return
    if (handle.isEmpty()) {
      setError('Empty canvas: please draw a digit before predicting.')
      return
    }
    const blob = await handle.toBlob()
    if (!blob) {
      setError('Could not export the canvas image.')
      return
    }
    await runPredict(blob, 'canvas')
  }

  const onUpload = async (file: File | undefined) => {
    if (!file) return
    const ok = ['image/png', 'image/jpeg', 'image/jpg'].includes(file.type)
    if (!ok) {
      setError('Unsupported file type. Upload a PNG or JPEG image.')
      return
    }
    await runPredict(file, 'upload')
  }

  return (
    <div className="space-y-8">
      <div>
        <h1 className="font-display text-3xl tracking-tight">Digit Recognition Workspace</h1>
        <p className="mt-2 text-[var(--muted)]">
          Draw a digit or upload an image, then submit it for CNN classification.
        </p>
      </div>

      <div className="grid gap-8 lg:grid-cols-2">
        <section className="space-y-4 rounded-2xl border border-[var(--border)] bg-white p-5 shadow-sm">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <h2 className="font-medium">Canvas</h2>
            <div className="flex items-center gap-2 text-sm text-[var(--muted)]">
              <label htmlFor="brush">Brush</label>
              <input
                id="brush"
                type="range"
                min={8}
                max={28}
                value={brushSize}
                onChange={(e) => setBrushSize(Number(e.target.value))}
                className="accent-[var(--accent)]"
              />
            </div>
          </div>

          <DigitCanvas ref={canvasRef} brushSize={brushSize} />

          <div className="flex flex-wrap gap-2">
            <button
              type="button"
              className="btn-primary"
              onClick={onPredict}
              disabled={pending}
            >
              {pending ? <Loader2 className="animate-spin" size={16} /> : <Wand2 size={16} />}
              {pending ? 'Predicting…' : 'Predict'}
            </button>
            <button
              type="button"
              className="btn-secondary"
              onClick={() => canvasRef.current?.undo()}
              disabled={pending}
            >
              <Undo2 size={16} />
              Undo
            </button>
            <button
              type="button"
              className="btn-secondary"
              onClick={() => {
                canvasRef.current?.clear()
                setResult(null)
                setError(null)
              }}
              disabled={pending}
            >
              <Eraser size={16} />
              Clear
            </button>
            <button
              type="button"
              className="btn-secondary"
              onClick={() => fileRef.current?.click()}
              disabled={pending}
            >
              <Upload size={16} />
              Upload
            </button>
            <input
              ref={fileRef}
              type="file"
              accept="image/png,image/jpeg"
              className="hidden"
              onChange={(e) => {
                void onUpload(e.target.files?.[0])
                e.target.value = ''
              }}
            />
          </div>

          {error && (
            <div
              role="alert"
              className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-800"
            >
              {error}
            </div>
          )}
        </section>

        <section className="space-y-4 rounded-2xl border border-[var(--border)] bg-white p-5 shadow-sm">
          <h2 className="font-medium">Prediction</h2>
          {!result && !pending && !error && (
            <p className="text-sm text-[var(--muted)]">
              Results will appear here after you submit a drawing or upload.
            </p>
          )}
          {pending && (
            <div className="flex items-center gap-2 text-[var(--muted)]">
              <Loader2 className="animate-spin" size={18} />
              Running inference…
            </div>
          )}
          {result && (
            <>
              <div className="flex items-end gap-4">
                <div>
                  <p className="text-sm text-[var(--muted)]">Predicted digit</p>
                  <p className="font-display text-6xl leading-none text-[var(--accent)]">
                    {result.predicted_digit}
                  </p>
                </div>
                <div className="pb-2">
                  <p className="text-sm text-[var(--muted)]">Confidence</p>
                  <p className="text-2xl font-semibold">
                    {(result.confidence * 100).toFixed(1)}%
                  </p>
                  <p className="text-xs text-[var(--muted)]">Source: {sourceLabel}</p>
                </div>
              </div>
              <div>
                <h3 className="mb-2 text-sm font-medium text-[var(--muted)]">
                  Class probabilities
                </h3>
                <ProbabilityChart probabilities={result.probabilities} />
              </div>
            </>
          )}
        </section>
      </div>
    </div>
  )
}
