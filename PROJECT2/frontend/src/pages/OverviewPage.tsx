import { Link } from 'react-router-dom'
import { ArrowRight, Layers, Sparkles, Trees } from 'lucide-react'

export function OverviewPage() {
  return (
    <div className="space-y-10">
      <section className="grid gap-8 lg:grid-cols-[1.2fr_0.8fr] lg:items-center">
        <div className="space-y-4">
          <p className="text-sm font-medium uppercase tracking-wider text-[var(--accent)]">
            Supervised Learning
          </p>
          <h1 className="font-display text-4xl tracking-tight sm:text-5xl">
            AdaBoost vs Random Forest
          </h1>
          <p className="max-w-xl text-lg text-[var(--muted)]">
            Train, compare, and interpret two ensemble classifiers on real scikit-learn datasets —
            Breast Cancer Wisconsin (binary) and Iris (multiclass).
          </p>
          <Link to="/train" className="btn-primary inline-flex">
            Start Experiment <ArrowRight size={16} />
          </Link>
        </div>
        <div className="rounded-2xl border border-[var(--border)] bg-white p-6 shadow-sm">
          <h2 className="font-medium">Learning objectives</h2>
          <ul className="mt-3 space-y-2 text-sm text-[var(--muted)]">
            <li>Understand bagging (Random Forest) vs boosting (AdaBoost)</li>
            <li>Configure hyperparameters and evaluate fairly on one split</li>
            <li>Inspect confusion matrices, ROC-AUC, and feature importance</li>
            <li>Make live predictions from a feature form</li>
          </ul>
        </div>
      </section>

      <section className="grid gap-4 sm:grid-cols-3">
        {[
          {
            icon: Trees,
            title: 'Random Forest',
            body: 'Many trees trained on bootstrap samples with random feature subsets; predictions are aggregated by vote.',
          },
          {
            icon: Layers,
            title: 'AdaBoost',
            body: 'Sequential weak learners. Misclassified samples get higher weight; final prediction is a weighted vote.',
          },
          {
            icon: Sparkles,
            title: 'Fair comparison',
            body: 'Same stratified train/test split, same metrics. No hardcoded winner — results depend on data and tuning.',
          },
        ].map((c) => (
          <article key={c.title} className="rounded-xl border border-[var(--border)] bg-white p-5 shadow-sm">
            <c.icon className="mb-3 text-[var(--accent)]" size={22} />
            <h3 className="font-medium">{c.title}</h3>
            <p className="mt-2 text-sm text-[var(--muted)]">{c.body}</p>
          </article>
        ))}
      </section>
    </div>
  )
}
