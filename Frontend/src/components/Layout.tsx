import { NavLink, Outlet } from 'react-router-dom'
import { BrainCircuit } from 'lucide-react'

const links = [
  { to: '/', label: 'Home', end: true },
  { to: '/workspace', label: 'Workspace' },
  { to: '/analytics', label: 'Analytics' },
  { to: '/how-it-works', label: 'How It Works' },
  { to: '/about', label: 'About' },
]

export function Layout() {
  return (
    <div className="min-h-screen bg-[var(--bg)] text-[var(--ink)]">
      <header className="border-b border-[var(--border)] bg-white/90 backdrop-blur-sm sticky top-0 z-40">
        <div className="mx-auto flex max-w-6xl items-center justify-between gap-4 px-4 py-3 sm:px-6">
          <NavLink to="/" className="flex items-center gap-2 font-semibold tracking-tight">
            <span className="flex h-9 w-9 items-center justify-center rounded-lg bg-[var(--accent-soft)] text-[var(--accent)]">
              <BrainCircuit size={20} aria-hidden />
            </span>
            <span className="hidden sm:inline">Digit Recognition</span>
          </NavLink>
          <nav className="flex flex-wrap items-center justify-end gap-1 sm:gap-2" aria-label="Main">
            {links.map((link) => (
              <NavLink
                key={link.to}
                to={link.to}
                end={link.end}
                className={({ isActive }) =>
                  [
                    'rounded-md px-2.5 py-1.5 text-sm transition-colors',
                    isActive
                      ? 'bg-[var(--accent-soft)] text-[var(--accent)] font-medium'
                      : 'text-[var(--muted)] hover:text-[var(--ink)] hover:bg-neutral-100',
                  ].join(' ')
                }
              >
                {link.label}
              </NavLink>
            ))}
          </nav>
        </div>
      </header>
      <main className="mx-auto max-w-6xl px-4 py-8 sm:px-6 sm:py-10">
        <Outlet />
      </main>
      <footer className="border-t border-[var(--border)] py-6 text-center text-sm text-[var(--muted)]">
        Educational MNIST CNN project · Hugging Face dataset
      </footer>
    </div>
  )
}
