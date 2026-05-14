// 1200x630 Open Graph card for link previews (LinkedIn, X, Slack, iMessage, etc).
//
// To regenerate the static PNG:
//   1. Temporarily render <OGCard /> from main.jsx instead of <App />.
//   2. `npm run dev`, open the page in Chrome.
//   3. DevTools → Device Mode → Responsive 1200x630, DPR 2.
//   4. Toolbar `⋮` → Capture screenshot.
//   5. Save as landing/public/og.png and revert main.jsx.
//
// The current landing/public/og.png was AI-generated to match this design.
import { useEffect } from 'react'

const REELS = [
  { title: 'Why Ravan Could\nNever Break Sita', score: '0.89', tint: '#243a6a' },
  { title: 'She Reached for\nDeath, Found Hope', score: '0.83', tint: '#1c3358' },
  { title: 'How Hanuman Knew\nIt Was Sita',     score: '0.80', tint: '#27406b' },
]

export default function OGCard() {
  useEffect(() => {
    document.body.style.margin = '0'
    document.body.style.background = '#0a101e'
    document.body.style.display = 'grid'
    document.body.style.placeItems = 'center'
    document.body.style.minHeight = '100vh'
  }, [])

  return (
    <div style={S.canvas}>
      <div style={S.grid} />

      <div style={S.brand}>
        <span style={S.brandMark}>✂</span>
        <span>PodClipper</span>
      </div>

      <div style={S.left}>
        <div style={S.badge}>
          <span style={S.badgeDot} />
          OPEN SOURCE · PYTHON · CLAUDE AI
        </div>
        <h1 style={S.h1}>
          Drop a podcast.<br />
          Get <em style={S.em}>viral reels.</em>
        </h1>
        <p style={S.tag}>
          AI finds your most shareable moments — speaker-tracked,
          karaoke-captioned, 9:16 vertical. Local. One command.
        </p>
        <div style={S.chips}>
          {['faster-whisper', 'Claude AI', 'YOLOv8', 'FFmpeg'].map(c => (
            <span key={c} style={S.chip}>{c}</span>
          ))}
        </div>
      </div>

      <div style={S.phones}>
        {REELS.map((r, i) => (
          <div key={i} style={{
            ...S.phone,
            transform: `rotate(${(i - 1) * 5}deg) translateY(${Math.abs(i - 1) * 14}px)`,
            zIndex: i === 1 ? 3 : 1,
            background: `linear-gradient(180deg, ${r.tint} 0%, #0a101e 100%)`,
          }}>
            <div style={S.notch} />
            <div style={S.phoneScore}>★ AI {r.score} · PUBLISH</div>
            <div style={S.play}>▶</div>
            <div style={S.sub}>
              <span style={{ color: '#94A3B8' }}>scored </span>
              <span style={{ color: '#fff', fontWeight: 700 }}>shareable</span>
              <span style={{ color: '#94A3B8' }}> by AI</span>
            </div>
            <div style={S.phoneTitle}>{r.title}</div>
          </div>
        ))}
      </div>

      <div style={S.url}>
        <span style={S.urlDot} />
        podclipper.loukik.dev
      </div>
    </div>
  )
}

const S = {
  canvas: {
    width: 1200, height: 630, position: 'relative', overflow: 'hidden',
    background: 'linear-gradient(135deg, #EBF3FF 0%, #F4F8FF 55%, #EEF4FF 100%)',
    fontFamily: "'Instrument Sans', system-ui, sans-serif",
    color: '#0F172A', boxSizing: 'border-box',
  },
  grid: {
    position: 'absolute', inset: 0,
    backgroundImage: 'radial-gradient(circle, rgba(37,99,235,0.14) 1px, transparent 1px)',
    backgroundSize: '26px 26px',
    WebkitMaskImage: 'linear-gradient(180deg, rgba(0,0,0,0.55), rgba(0,0,0,0.1))',
            maskImage: 'linear-gradient(180deg, rgba(0,0,0,0.55), rgba(0,0,0,0.1))',
  },
  brand: {
    position: 'absolute', top: 38, left: 60,
    display: 'flex', alignItems: 'center', gap: 12,
    fontFamily: "'Fraunces', serif", fontWeight: 700, fontSize: 26,
    letterSpacing: '-0.01em', color: '#0F172A',
  },
  brandMark: {
    width: 40, height: 40, borderRadius: 9, background: '#2563EB',
    color: '#fff', display: 'inline-flex', alignItems: 'center',
    justifyContent: 'center', fontSize: 22,
    boxShadow: '0 4px 14px rgba(37,99,235,0.35)',
  },

  left: { position: 'absolute', left: 60, top: 138, width: 640 },
  badge: {
    display: 'inline-flex', alignItems: 'center', gap: 8,
    padding: '7px 14px', background: '#fff', border: '1.5px solid #D1E2FF',
    borderRadius: 999, boxShadow: '0 1px 3px rgba(15,23,42,0.06)',
    fontSize: 12, fontWeight: 700, letterSpacing: '0.12em',
    color: '#64748B', marginBottom: 28,
  },
  badgeDot: { width: 7, height: 7, borderRadius: '50%', background: '#2563EB' },
  h1: {
    fontFamily: "'Fraunces', serif", fontWeight: 800, fontSize: 88,
    lineHeight: 0.95, letterSpacing: '-0.038em', margin: '0 0 22px',
  },
  em: { color: '#2563EB', fontStyle: 'italic' },
  tag: {
    fontSize: 22, lineHeight: 1.45, color: '#475569',
    margin: '0 0 28px', maxWidth: 580,
  },
  chips: { display: 'flex', gap: 8, flexWrap: 'wrap' },
  chip: {
    padding: '7px 14px', background: 'rgba(37,99,235,0.08)',
    border: '1.5px solid #D1E2FF', borderRadius: 999,
    fontSize: 13, fontWeight: 600, color: '#1D4ED8',
  },

  phones: {
    position: 'absolute', right: 72, top: 105,
    width: 470, height: 440, display: 'flex',
    gap: 8, alignItems: 'center', justifyContent: 'center',
  },
  phone: {
    width: 142, height: 296, borderRadius: 30, border: '3px solid #1a1d24',
    boxShadow: '0 24px 60px rgba(15,23,42,0.25), 0 8px 24px rgba(15,23,42,0.15)',
    position: 'relative', padding: 12, boxSizing: 'border-box', overflow: 'hidden',
  },
  notch: {
    position: 'absolute', top: 3, left: '50%', transform: 'translateX(-50%)',
    width: 46, height: 13, background: '#0a0c12', borderRadius: '0 0 10px 10px',
  },
  phoneScore: {
    marginTop: 18, background: 'rgba(37,99,235,0.92)', color: '#fff',
    fontSize: 9, fontWeight: 700, letterSpacing: '0.06em',
    padding: '4px 8px', borderRadius: 4, display: 'inline-block',
  },
  play: {
    position: 'absolute', top: '42%', left: '50%',
    transform: 'translate(-50%, -50%)', width: 44, height: 44,
    borderRadius: '50%', background: 'rgba(255,255,255,0.16)',
    border: '1.5px solid rgba(255,255,255,0.5)', color: '#fff',
    fontSize: 14, display: 'flex', alignItems: 'center',
    justifyContent: 'center', paddingLeft: 3,
    backdropFilter: 'blur(4px)', WebkitBackdropFilter: 'blur(4px)',
  },
  sub: {
    position: 'absolute', bottom: 52, left: 8, right: 8,
    fontSize: 9, fontWeight: 600, textAlign: 'center',
    background: 'rgba(10,16,30,0.85)', padding: '5px 6px', borderRadius: 4,
  },
  phoneTitle: {
    position: 'absolute', bottom: 14, left: 10, right: 10,
    color: '#fff', fontFamily: "'Fraunces', serif",
    fontWeight: 600, fontSize: 11, lineHeight: 1.2,
    whiteSpace: 'pre-line', textAlign: 'center',
  },

  url: {
    position: 'absolute', bottom: 36, left: 60,
    display: 'inline-flex', alignItems: 'center', gap: 8,
    fontSize: 16, fontWeight: 600, color: '#2563EB',
    background: '#fff', border: '1.5px solid #D1E2FF',
    borderRadius: 999, padding: '8px 18px',
    boxShadow: '0 1px 3px rgba(15,23,42,0.06)',
  },
  urlDot: { width: 8, height: 8, borderRadius: '50%', background: '#10B981' },
}
