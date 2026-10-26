import {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
} from 'react'

export interface DigitCanvasHandle {
  clear: () => void
  undo: () => void
  isEmpty: () => boolean
  toBlob: () => Promise<Blob | null>
  toDataURL: () => string
}

interface DigitCanvasProps {
  brushSize: number
  className?: string
}

const CANVAS_SIZE = 280

export const DigitCanvas = forwardRef<DigitCanvasHandle, DigitCanvasProps>(
  function DigitCanvas({ brushSize, className }, ref) {
    const canvasRef = useRef<HTMLCanvasElement>(null)
    const drawing = useRef(false)
    const lastPoint = useRef<{ x: number; y: number } | null>(null)
    const historyRef = useRef<ImageData[]>([])
    const [empty, setEmpty] = useState(true)

    const getCtx = useCallback(() => {
      const canvas = canvasRef.current
      if (!canvas) return null
      return canvas.getContext('2d', { willReadFrequently: true })
    }, [])

    const fillWhite = useCallback(() => {
      const canvas = canvasRef.current
      const ctx = getCtx()
      if (!canvas || !ctx) return
      ctx.fillStyle = '#ffffff'
      ctx.fillRect(0, 0, canvas.width, canvas.height)
    }, [getCtx])

    const pushHistory = useCallback(() => {
      const canvas = canvasRef.current
      const ctx = getCtx()
      if (!canvas || !ctx) return
      historyRef.current.push(ctx.getImageData(0, 0, canvas.width, canvas.height))
      if (historyRef.current.length > 30) historyRef.current.shift()
    }, [getCtx])

    const checkEmpty = useCallback(() => {
      const canvas = canvasRef.current
      const ctx = getCtx()
      if (!canvas || !ctx) return true
      const { data } = ctx.getImageData(0, 0, canvas.width, canvas.height)
      for (let i = 0; i < data.length; i += 4) {
        // Non-white pixel counts as ink
        if (data[i]! < 250 || data[i + 1]! < 250 || data[i + 2]! < 250) {
          setEmpty(false)
          return false
        }
      }
      setEmpty(true)
      return true
    }, [getCtx])

    useEffect(() => {
      const canvas = canvasRef.current
      if (!canvas) return
      canvas.width = CANVAS_SIZE
      canvas.height = CANVAS_SIZE
      fillWhite()
      pushHistory()
    }, [fillWhite, pushHistory])

    const pointerPos = (e: React.PointerEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current!
      const rect = canvas.getBoundingClientRect()
      const scaleX = canvas.width / rect.width
      const scaleY = canvas.height / rect.height
      return {
        x: (e.clientX - rect.left) * scaleX,
        y: (e.clientY - rect.top) * scaleY,
      }
    }

    const drawLine = (from: { x: number; y: number }, to: { x: number; y: number }) => {
      const ctx = getCtx()
      if (!ctx) return
      ctx.strokeStyle = '#111827'
      ctx.fillStyle = '#111827'
      ctx.lineWidth = brushSize
      ctx.lineCap = 'round'
      ctx.lineJoin = 'round'
      ctx.beginPath()
      ctx.moveTo(from.x, from.y)
      ctx.lineTo(to.x, to.y)
      ctx.stroke()
    }

    const onPointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
      e.preventDefault()
      const canvas = canvasRef.current
      if (!canvas) return
      canvas.setPointerCapture(e.pointerId)
      pushHistory()
      drawing.current = true
      const p = pointerPos(e)
      lastPoint.current = p
      const ctx = getCtx()
      if (ctx) {
        ctx.beginPath()
        ctx.fillStyle = '#111827'
        ctx.arc(p.x, p.y, brushSize / 2, 0, Math.PI * 2)
        ctx.fill()
      }
      setEmpty(false)
    }

    const onPointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
      if (!drawing.current || !lastPoint.current) return
      e.preventDefault()
      const p = pointerPos(e)
      drawLine(lastPoint.current, p)
      lastPoint.current = p
    }

    const endStroke = (e: React.PointerEvent<HTMLCanvasElement>) => {
      if (!drawing.current) return
      drawing.current = false
      lastPoint.current = null
      try {
        canvasRef.current?.releasePointerCapture(e.pointerId)
      } catch {
        /* ignore */
      }
      checkEmpty()
    }

    useImperativeHandle(ref, () => ({
      clear: () => {
        fillWhite()
        historyRef.current = []
        pushHistory()
        setEmpty(true)
      },
      undo: () => {
        const canvas = canvasRef.current
        const ctx = getCtx()
        if (!canvas || !ctx || historyRef.current.length < 2) return
        historyRef.current.pop()
        const prev = historyRef.current[historyRef.current.length - 1]
        if (prev) ctx.putImageData(prev, 0, 0)
        checkEmpty()
      },
      isEmpty: () => empty,
      toBlob: () =>
        new Promise((resolve) => {
          canvasRef.current?.toBlob((b) => resolve(b), 'image/png')
        }),
      toDataURL: () => canvasRef.current?.toDataURL('image/png') ?? '',
    }))

    return (
      <div className={`relative ${className ?? ''}`}>
        <canvas
          ref={canvasRef}
          className="w-full max-w-[320px] touch-none rounded-xl border border-[var(--border)] bg-white shadow-sm cursor-crosshair aspect-square"
          style={{ touchAction: 'none' }}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={endStroke}
          onPointerLeave={endStroke}
          onPointerCancel={endStroke}
          aria-label="Drawing canvas for handwritten digit"
          role="img"
        />
        {empty && (
          <p className="pointer-events-none absolute inset-0 flex items-center justify-center text-sm text-[var(--muted)]">
            Draw a digit (0–9)
          </p>
        )}
      </div>
    )
  },
)
