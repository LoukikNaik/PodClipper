import { useEffect, useState } from 'react'

function preference() {
  try {
    const saved = localStorage.getItem('podclipper-theme')
    if (saved === 'light' || saved === 'dark') return saved
  } catch {}
  return null
}

export default function ThemeToggle() {
  const [choice, setChoice] = useState(preference)
  const theme = choice || 'light'
  useEffect(() => {
    const stored = event => {
      if (event.key === 'podclipper-theme' || event.key === null) setChoice(preference())
    }
    window.addEventListener('storage', stored)
    return () => window.removeEventListener('storage', stored)
  }, [])
  useEffect(() => { document.documentElement.dataset.theme = theme }, [theme])
  const toggle = () => {
    const next = theme === 'dark' ? 'light' : 'dark'
    setChoice(next)
    try { localStorage.setItem('podclipper-theme', next) } catch {}
  }
  const label = `Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`
  return <button className="theme-toggle" type="button" onClick={toggle} aria-label={label} title={label}>
    <svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      {theme === 'dark' ? <><circle cx="12" cy="12" r="4" /><path d="M12 2v2m0 16v2M2 12h2m16 0h2M5 5l1.5 1.5m11 11L19 19M5 19l1.5-1.5m11-11L19 5" /></> : <path d="M21 13a9 9 0 0 1-10-10A9 9 0 1 0 21 13Z" />}
    </svg>
  </button>
}
