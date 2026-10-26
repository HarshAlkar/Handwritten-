import { ArrowDown } from 'lucide-react'

const steps = [
  {
    title: '1. Input image',
    body: 'You draw on the canvas or upload a PNG/JPEG. The frontend sends the image to the FastAPI backend.',
  },
  {
    title: '2. Preprocessing',
    body: 'The image is converted to grayscale, foreground/background polarity is detected, the digit is cropped and centered, then resized to 28×28 and normalized to [0, 1].',
  },
  {
    title: '3. CNN feature extraction',
    body: 'Convolutional layers with ReLU and max-pooling detect edges, loops, and local stroke patterns that distinguish digits.',
  },
  {
    title: '4. Classification',
    body: 'Flattened features pass through dense layers with dropout. A softmax layer produces a probability for each digit 0–9.',
  },
  {
    title: '5. Final prediction',
    body: 'The class with the highest probability is returned as the predicted digit, along with confidence and the full probability vector.',
  },
]

export function HowItWorksPage() {
  return (
    <div className="space-y-8">
      <div>
        <h1 className="font-display text-3xl tracking-tight">How It Works</h1>
        <p className="mt-2 max-w-2xl text-[var(--muted)]">
          From a handwritten stroke to a digit label: the end-to-end inference pipeline.
        </p>
      </div>

      <ol className="mx-auto max-w-2xl space-y-0">
        {steps.map((step, idx) => (
          <li key={step.title} className="relative">
            <div className="rounded-xl border border-[var(--border)] bg-white p-5 shadow-sm">
              <h2 className="font-medium text-[var(--accent)]">{step.title}</h2>
              <p className="mt-2 text-sm text-[var(--muted)]">{step.body}</p>
            </div>
            {idx < steps.length - 1 && (
              <div className="flex justify-center py-2 text-[var(--accent)]" aria-hidden>
                <ArrowDown size={18} />
              </div>
            )}
          </li>
        ))}
      </ol>

      <section className="rounded-2xl border border-[var(--border)] bg-white p-6 shadow-sm">
        <h2 className="font-display text-xl tracking-tight">Model sketch</h2>
        <pre className="mt-4 overflow-x-auto rounded-lg bg-[var(--surface)] p-4 text-xs leading-relaxed text-[var(--ink)] sm:text-sm">
{`Input 28×28×1
  → Conv2D(32) + ReLU + MaxPool
  → Conv2D(64) + ReLU + MaxPool
  → Dropout(0.25)
  → Flatten → Dense(128) + ReLU
  → Dropout(0.5)
  → Dense(10) + Softmax`}
        </pre>
      </section>
    </div>
  )
}
