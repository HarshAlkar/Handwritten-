import { NavLink, Outlet } from 'react-router-dom'
import { GitCompareArrows } from 'lucide-react'

const links = [
  { to: '/', label: 'Overview', end: true },
  { to: '/dataset', label: 'Dataset' },
  { to: '/train', label: 'Train' },
  { to: '/predict', label: 'Predict' },
  { to: '/compare', label: 'Compare' },
  { to: '/importance', label: 'Importance' },
  { to: '/visualizer', label: 'Visualizer' },
  { to: '/docs', label: 'Docs' },
]

export function Layout() {
  return (
    <div className="min-h-screen bg-[var(--bg)] text-[var(--ink)]">
      <header className="sticky top-0 z-40 border-b border-[var(--border)] bg-white/90 backdrop-blur">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-3 px-4 py-3 sm:px-6">
          <NavLink to="/" className="flex items-center gap-2 font-semibold">
            <span className="flex h-9 w-9 items-center justify-center rounded-lg bg-[var(--accent-soft)] text-[var(--accent)]">
              <GitCompareArrows size={18} />
            </span>
            <span className="hidden sm:inline">AdaBoost vs Random Forest</span>
          </NavLink>
          <nav className="flex flex-wrap justify-end gap-1">
            {links.map((l) => (
              <NavLink
                key={l.to}
                to={l.to}
                end={l.end}
                className={({ isActive }) =>
                  `rounded-md px-2.5 py-1.5 text-sm ${
                    isActive
                      ? 'bg-[var(--accent-soft)] font-medium text-[var(--accent)]'
                      : 'text-[var(--muted)] hover:bg-neutral-100'
                  }`
                }
              >
                {l.label}
              </NavLink>
            ))}
          </nav>
        </div>
      </header>
      <main className="mx-auto max-w-6xl px-4 py-8 sm:px-6">
        <Outlet />
      </main>
    </div>
  )
}
