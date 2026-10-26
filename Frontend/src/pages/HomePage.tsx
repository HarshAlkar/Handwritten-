import { Link } from 'react-router-dom'
import { ArrowRight, Layers, PenLine, Sparkles } from 'lucide-react'

export function HomePage() {
  return (
    <div className="space-y-12">
      <section className="grid gap-8 lg:grid-cols-[1.2fr_0.8fr] lg:items-center">
        <div className="space-y-5">
          <p className="text-sm font-medium uppercase tracking-wider text-[var(--accent)]">
            Deep Learning · MNIST
          </p>
          <h1 className="font-display text-4xl leading-tight tracking-tight text-[var(--ink)] sm:text-5xl">
            Handwritten Digit Recognition
          </h1>
          <p className="max-w-xl text-lg text-[var(--muted)]">
            Draw a digit and watch a convolutional neural network classify it in real time.
            Trained on the Hugging Face MNIST dataset with TensorFlow and Keras.
          </p>
          <div className="flex flex-wrap gap-3">
            <Link to="/workspace" className="btn-primary">
              Start Drawing
              <ArrowRight size={16} aria-hidden />
            </Link>
            <Link to="/how-it-works" className="btn-secondary">
              How it works
            </Link>
          </div>
        </div>
        <div className="rounded-2xl border border-[var(--border)] bg-white p-6 shadow-sm">
          <div className="grid grid-cols-5 gap-2">
            {Array.from({ length: 10 }, (_, i) => (
              <div
                key={i}
                className="flex aspect-square items-center justify-center rounded-lg bg-[var(--surface)] font-display text-2xl text-[var(--ink)]"
                aria-hidden
              >
                {i}
              </div>
            ))}
          </div>
          <p className="mt-4 text-sm text-[var(--muted)]">
            Classes 0–9 · 28×28 grayscale · CNN with softmax output
          </p>
        </div>
      </section>

      <section className="grid gap-4 sm:grid-cols-3">
        {[
          {
            icon: PenLine,
            title: 'Interactive canvas',
            body: 'Draw with mouse, touch, or stylus. Upload PNG/JPEG digits too.',
          },
          {
            icon: Layers,
            title: 'CNN predictions',
            body: 'See the predicted digit, confidence, and full probability distribution.',
          },
          {
            icon: Sparkles,
            title: 'Real metrics',
            body: 'Explore training curves, confusion matrix, and test-set scores from actual evaluation.',
          },
        ].map((card) => (
          <article
            key={card.title}
            className="rounded-xl border border-[var(--border)] bg-white p-5 shadow-sm"
          >
            <card.icon className="mb-3 text-[var(--accent)]" size={22} aria-hidden />
            <h2 className="font-medium text-[var(--ink)]">{card.title}</h2>
            <p className="mt-2 text-sm text-[var(--muted)]">{card.body}</p>
          </article>
        ))}
      </section>

      <section className="rounded-2xl border border-[var(--border)] bg-white p-6 sm:p-8">
        <h2 className="font-display text-2xl tracking-tight">About MNIST & CNNs</h2>
        <p className="mt-3 max-w-3xl text-[var(--muted)]">
          MNIST contains 70,000 labeled handwritten digits. A convolutional neural network learns
          local visual patterns—edges, loops, and strokes—then maps them to digit classes through
          dense layers and a softmax classifier. This project loads MNIST from Hugging Face
          (<code className="text-[var(--ink)]">ylecun/mnist</code>), trains a CNN, and serves
          live predictions through a FastAPI backend.
        </p>
      </section>
    </div>
  )
}
