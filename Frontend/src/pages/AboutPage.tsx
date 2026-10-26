export function AboutPage() {
  return (
    <div className="space-y-8">
      <div>
        <h1 className="font-display text-3xl tracking-tight">About & Documentation</h1>
        <p className="mt-2 max-w-2xl text-[var(--muted)]">
          An educational full-stack project for handwritten digit recognition with a real trained CNN.
        </p>
      </div>

      <section className="prose-section">
        <h2>Objectives</h2>
        <ul>
          <li>Train a CNN on MNIST loaded from Hugging Face (<code>ylecun/mnist</code>).</li>
          <li>Serve predictions through a typed FastAPI REST API.</li>
          <li>Provide an accessible React UI for drawing, upload, and analytics.</li>
          <li>Report genuine evaluation metrics—not placeholders.</li>
        </ul>
      </section>

      <section className="prose-section">
        <h2>Technologies</h2>
        <ul>
          <li>Python, TensorFlow/Keras, NumPy, Pillow, scikit-learn, Hugging Face Datasets</li>
          <li>FastAPI, Uvicorn, Pydantic</li>
          <li>React, Vite, TypeScript, Tailwind CSS, Recharts, Lucide</li>
          <li>pytest, Vitest, React Testing Library</li>
        </ul>
      </section>

      <section className="prose-section">
        <h2>Dataset</h2>
        <p>
          MNIST provides 60,000 training and 10,000 test grayscale digit images (28×28). This project
          downloads the official Hugging Face mirror and preserves the train/test split, carving a
          validation subset from training data.
        </p>
      </section>

      <section className="prose-section">
        <h2>Model architecture</h2>
        <p>
          A compact CNN with two convolutional blocks, dropout regularization, a dense hidden layer,
          and a 10-way softmax. Optimized with Adam and sparse categorical cross-entropy. Training
          uses early stopping and model checkpointing.
        </p>
      </section>

      <section className="prose-section">
        <h2>Limitations & future work</h2>
        <ul>
          <li>MNIST digits are centered and relatively clean; messy real-world photos may need stronger preprocessing.</li>
          <li>The model does not handle multi-digit images or letters.</li>
          <li>Possible improvements: data augmentation, deeper architectures, ONNX export, auth, and production monitoring.</li>
        </ul>
      </section>
    </div>
  )
}
