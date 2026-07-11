import { useState, useEffect, useRef, useCallback } from 'react'

const GH = 'https://github.com/LoukikNaik/PodClipper'

// ── Intersection-based reveal ────────────────────────────────────
function useReveal(threshold = 0.1) {
  const ref = useRef(null)
  const [visible, setVisible] = useState(false)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const obs = new IntersectionObserver(
      ([e]) => { if (e.isIntersecting) { setVisible(true); obs.disconnect() } },
      { threshold }
    )
    obs.observe(el)
    return () => obs.disconnect()
  }, [threshold])
  return [ref, visible]
}

// ── Animated counter ─────────────────────────────────────────────
function AnimCounter({ target, suffix = '' }) {
  const [count, setCount] = useState(0)
  const [started, setStarted] = useState(false)
  const ref = useRef(null)
  useEffect(() => {
    const obs = new IntersectionObserver(
      ([e]) => { if (e.isIntersecting) { setStarted(true); obs.disconnect() } },
      { threshold: 0.5 }
    )
    if (ref.current) obs.observe(ref.current)
    return () => obs.disconnect()
  }, [])
  useEffect(() => {
    if (!started) return
    let start = null
    const step = ts => {
      if (!start) start = ts
      const p = Math.min((ts - start) / 1100, 1)
      setCount(Math.floor(p * target))
      if (p < 1) requestAnimationFrame(step)
    }
    requestAnimationFrame(step)
  }, [started, target])
  return <span ref={ref}>{count}{suffix}</span>
}

// ── Icons ────────────────────────────────────────────────────────
const GithubIcon = ({ size = 16 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor" aria-hidden>
    <path d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0 1 12 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0 0 22 12.017C22 6.484 17.522 2 12 2z" />
  </svg>
)

const ArrowRight = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
    <line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/>
  </svg>
)

// ── Nav ──────────────────────────────────────────────────────────
function Nav() {
  const [scrolled, setScrolled] = useState(false)
  useEffect(() => {
    const fn = () => setScrolled(window.scrollY > 16)
    window.addEventListener('scroll', fn, { passive: true })
    return () => window.removeEventListener('scroll', fn)
  }, [])
  return (
    <nav className={`nav${scrolled ? ' nav--up' : ''}`}>
      <a href="/" className="nav-logo">
        <span className="nav-logo-icon" aria-hidden>✂</span>
        PodClipper
      </a>
      <div className="nav-right">
        <a href="#how" className="nav-text-link">How it works</a>
        <a href={GH} className="nav-gh" target="_blank" rel="noopener noreferrer">
          <GithubIcon size={14} />
          GitHub
        </a>
      </div>
    </nav>
  )
}

// ── Hero ─────────────────────────────────────────────────────────
function Hero() {
  return (
    <section className="hero">
      <div className="hero-grid" aria-hidden />
      <div className="hero-inner">
        <div className="hero-badge">
          <span className="hero-badge-dot" />
          Open source · Python · Claude AI
        </div>
        <h1 className="hero-h1">
          {/* Computer vision plus LLM reasoning.<br /> */}
          Long podcasts. <span className="hero-highlight">Short reels.</span>
        </h1>
        <p className="hero-p">
          We merged computer vision and LLM reasoning to convert long
          podcasts into digestible short, interesting reels. PodClipper
          picks the moments worth sharing, follows the active speaker
          through camera cuts, and burns in karaoke captions. Ready to
          post to TikTok, Reels, and Shorts.
        </p>
        <div className="hero-ctas">
          <a href={GH} className="btn-primary" target="_blank" rel="noopener noreferrer">
            <GithubIcon size={15} />
            Star on GitHub
          </a>
          <a href="#demo" className="btn-outline">
            See it in action <ArrowRight />
          </a>
        </div>
      </div>
    </section>
  )
}

// ── Stats bar ────────────────────────────────────────────────────
const STATS = [
  { n: 5,   suffix: '',   label: 'Reels from a single episode' },
  { n: 0,   suffix: '',   label: 'Manual editing steps required' },
  { n: 100, suffix: '%',  label: 'Local. Audio never leaves your machine' },
  { n: 1,   suffix: '',   label: 'Command to clip an entire episode' },
]

function StatsBar() {
  return (
    <div className="stats-bar">
      {STATS.map(s => (
        <div key={s.label} className="stat">
          <span className="stat-n">
            <AnimCounter target={s.n} suffix={s.suffix} />
          </span>
          <span className="stat-l">{s.label}</span>
        </div>
      ))}
    </div>
  )
}

// ── Demo ─────────────────────────────────────────────────────────
const EPISODES = [
  {
    youtubeId: 'RiHi4solYHE',
    label: 'Episode 1 · youtube.com/watch?v=RiHi4solYHE',
    reels: [
      {
        src: 'videos/reel_01_why-ravan-could-never-break-sita.mp4',
        score: 'AI score 4.5 / 5 · Publish',
        title: 'Why Ravan Could Never Break Sita',
        note: "Sita's blade-of-grass defiance. The AI surfaced this moment at 6 min 54 sec and rated it ready to publish.",
      },
      {
        src: 'videos/reel_02_she-reached-for-death-found-hope.mp4',
        score: 'AI score 4.2 / 5 · Publish',
        title: 'She Reached for Death, Found Hope',
        note: 'A full emotional arc with a twist ending. 47 seconds, surfaced from the 12-minute mark.',
      },
      {
        src: 'videos/reel_03_how-hanuman-knew-it-was-sita.mp4',
        score: 'AI score 3.8 / 5 · Review',
        title: 'How Hanuman Knew It Was Sita',
        note: 'Detective-logic payoff at the 3 min 28 sec mark. Flagged for review with feedback on adding context.',
      },
    ],
  },
  {
    youtubeId: '9FtradY1AI4',
    label: 'Episode 2 · youtube.com/watch?v=9FtradY1AI4',
    reels: [
      {
        src: 'videos/reel_01_confidence-is-an-output-not-input.mp4',
        score: 'AI score 3.3 / 5 · Review',
        title: 'Confidence Is an Output, Not Input',
        note: 'A contrarian reframe with a vivid personal-story payoff. Surfaced from the 1:49 mark.',
      },
      {
        src: 'videos/reel_02_make-anxiety-sit-in-the-back-seat.mp4',
        score: 'AI score 3.3 / 5 · Review',
        title: 'Make Anxiety Sit in the Back Seat',
        note: 'A memorable car-seat metaphor for handling anxiety in unknown territory. Surfaced from the 11:21 mark.',
      },
      {
        src: 'videos/reel_03_the-market-wont-kill-your-business.mp4',
        score: 'AI score 3.2 / 5 · Review',
        title: "The Market Won't Kill Your Business",
        note: 'Counterintuitive founder advice backed by a concrete COVID example. Surfaced from the 16:42 mark.',
      },
    ],
  },
  {
    youtubeId: 'KdTGjYFu2RQ',
    label: 'Stand-up set · comedian mode',
    reels: [
      {
        src: 'videos/reel_01_you-cant-sell-chairs-to-dogs.mp4',
        score: 'AI score 4.5 / 5 · Publish',
        title: "You Can't Sell Chairs to Dogs",
        note: 'A clueless boss sets a 50-crore target with no plan; it lands on a mic-drop dog punchline. Comedian mode held the frame on the performer the whole way. From the 6:50 mark.',
      },
      {
        src: 'videos/reel_02_b-schools-teach-just-one-thing.mp4',
        score: 'AI score 4.5 / 5 · Publish',
        title: 'B-Schools Teach Just One Thing',
        note: 'Four years of buildup pays off on one sharp line about the answer being inside the box. Single-performer framing throughout. From the 7:39 mark.',
      },
      {
        src: 'videos/reel_03_18-lakhs-to-become-my-boss.mp4',
        score: 'AI score 3.7 / 5 · Review',
        title: '18 Lakhs to Become My Boss',
        note: 'A self-contained bit on a fresh grad paying a fortune just to outrank a veteran. Flagged for review. From the 1:36 mark.',
      },
    ],
  },
]

function Phone({ src, score, title, note, delay = 0 }) {
  const [muted, setMuted] = useState(true)
  const [hovered, setHovered] = useState(false)
  const videoRef = useRef(null)
  const [ref, visible] = useReveal()

  const toggle = useCallback(() => {
    const v = videoRef.current
    if (!v) return
    v.muted = !v.muted
    setMuted(v.muted)
  }, [])

  return (
    <div
      ref={ref}
      className={`phone-wrap${visible ? ' visible' : ''}`}
      style={{ '--delay': `${delay}ms` }}
    >
      <button
        className={`phone${hovered ? ' phone--hover' : ''}`}
        onMouseEnter={() => setHovered(true)}
        onMouseLeave={() => setHovered(false)}
        onClick={toggle}
        aria-label={`${title}: ${muted ? 'unmute' : 'mute'}`}
      >
        <span className="phone-notch" aria-hidden />
        <div className="phone-screen">
          <video ref={videoRef} src={src} autoPlay muted loop playsInline />
          <div className={`phone-hint${hovered ? ' phone-hint--show' : ''}`}>
            {muted ? '🔇 Tap to unmute' : '🔊 Tap to mute'}
          </div>
        </div>
      </button>
      <div className="phone-meta">
        <span className="phone-score">★ {score}</span>
        <p className="phone-title">{title}</p>
        <p className="phone-note">{note}</p>
      </div>
    </div>
  )
}

function EpisodeBlock({ episode, index }) {
  const [yRef, yVisible] = useReveal()
  return (
    <div className="episode-block">
      <div ref={yRef} className={`yt-card${yVisible ? ' visible' : ''}`}>
        <div className="yt-label">
          <span className="yt-dot" />
          {episode.label}
        </div>
        <div className="yt-frame">
          <iframe
            src={`https://www.youtube.com/embed/${episode.youtubeId}`}
            title={`Source podcast episode ${index + 1}`}
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
            allowFullScreen
          />
        </div>
      </div>

      <div className="connector" aria-hidden>
        <div className="connector-track">
          <div className="connector-line" />
          <span className="connector-pill">PodClipper clipped this episode</span>
          <div className="connector-line" />
        </div>
        <span className="connector-arrow">↓</span>
      </div>

      <div className="phones">
        {episode.reels.map((r, i) => (
          <Phone key={r.src} {...r} delay={i * 90} />
        ))}
      </div>
    </div>
  )
}

function Demo() {
  const [hRef, hVisible] = useReveal()

  return (
    <section className="demo-section" id="demo">
      <div className="container">
        <div ref={hRef} className={`demo-head${hVisible ? ' visible' : ''}`}>
          <p className="eyebrow">See it in action</p>
          <h2 className="h2">Three episodes in.<br /><em>Nine reels out.</em></h2>
          <p className="lead">
            Three distinct sources fed into PodClipper — two podcasts and a
            stand-up set. Different topics, different speakers, same automated
            pipeline. Every clip selection, crop decision, caption, and title
            came from the AI. No human touched the timeline.
          </p>
        </div>
        {EPISODES.map((ep, idx) => (
          <EpisodeBlock key={ep.youtubeId} episode={ep} index={idx} />
        ))}
      </div>
    </section>
  )
}

// ── Step visualizations ───────────────────────────────────────────

const BARS = [22,45,60,80,48,90,65,78,52,85,38,70,58,92,44,76,55,68,88,35,62,78,42,95]

function TranscribeViz() {
  return (
    <div className="viz-screen">
      <div className="viz-wave">
        {BARS.map((peak, i) => (
          <div key={i} className="viz-bar" style={{
            '--peak': `${peak}px`,
            '--dur':  `${0.55 + (i % 6) * 0.09}s`,
            '--del':  `${(i % 9) * 0.06}s`,
          }} />
        ))}
      </div>
      <div className="viz-lines">
        {[
          ['00:03', 'standing at the edge of the lake...'],
          ['00:11', 'every joy reminded him of her'],
          ['00:19', 'the lotus, the wind, the birds...'],
        ].map(([ts, txt], i) => (
          <div key={i} className="viz-line" style={{ '--del': `${i * 0.55}s` }}>
            <span className="viz-ts">{ts}</span>
            <span className="viz-lt">{txt}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

function AnalyzeViz() {
  return (
    <div className="viz-screen">
      <div className="viz-segs">
        <div className="viz-seg viz-seg-dim">He stood at the edge of the lake watching...</div>
        <div className="viz-seg viz-seg-active">
          every joy reminded him of her
          <div className="viz-seg-hl" />
        </div>
        <div className="viz-seg viz-seg-dim">the lotus, the peacocks, the wind...</div>
      </div>
      <div className="viz-meter-row">
        <span className="viz-meter-lbl">Hook</span>
        <div className="viz-meter-track">
          <div className="viz-meter-fill" />
        </div>
        <span className="viz-meter-val">0.88</span>
      </div>
      <div className="viz-gen-row">
        <span className="viz-gen-tag">✦ Title</span>
        <span className="viz-gen-text">Every Joy Reminded Him of Her</span>
      </div>
    </div>
  )
}

function EditViz() {
  // bbox rect: x=92,y=22,w=96,h=144 → perimeter = 2*(96+144) = 480
  return (
    <div className="viz-screen viz-screen--pad0">
      <svg className="viz-svg" viewBox="0 0 280 172" xmlns="http://www.w3.org/2000/svg">
        {/* Rule-of-thirds grid */}
        <line x1="0" y1="57"  x2="280" y2="57"  stroke="#151f2e" strokeWidth="0.6"/>
        <line x1="0" y1="114" x2="280" y2="114" stroke="#151f2e" strokeWidth="0.6"/>
        <line x1="93"  y1="0" x2="93"  y2="172" stroke="#151f2e" strokeWidth="0.6"/>
        <line x1="186" y1="0" x2="186" y2="172" stroke="#151f2e" strokeWidth="0.6"/>

        {/* ── Person silhouette ── */}
        {/* Soft ambient glow behind the person */}
        <ellipse cx="140" cy="100" rx="46" ry="62" fill="rgba(255,255,255,0.03)"/>

        {/* Head */}
        <circle cx="140" cy="52" r="22" fill="#3a5068"/>
        {/* Face highlight, adds depth */}
        <ellipse cx="140" cy="49" rx="12" ry="14" fill="#4d6880" opacity="0.6"/>
        {/* Neck */}
        <rect x="133" y="72" width="14" height="10" rx="3" fill="#3a5068"/>
        {/* Shoulders + torso, classic bust icon shape */}
        <path
          d="M100 168 C100 122 120 106 140 106 C160 106 180 122 180 168 Z"
          fill="#3a5068"
        />
        {/* Subtle shoulder highlight */}
        <path
          d="M114 130 C116 116 126 108 140 108 C154 108 164 116 166 130"
          fill="none" stroke="#4d6880" strokeWidth="1.5" opacity="0.5"
        />

        {/* Detection bounding box, draws itself */}
        <rect
          className="viz-bbox"
          x="92" y="22" width="96" height="144"
          fill="none" stroke="#3B82F6" strokeWidth="1.5" rx="4"
          strokeDasharray="480" strokeDashoffset="480"
        />
        {/* Corner accent squares (give it a pro detection-UI look) */}
        <rect className="viz-chip-bg" x="92"  y="22"  width="8" height="8" fill="#3B82F6" rx="1"/>
        <rect className="viz-chip-bg" x="180" y="22"  width="8" height="8" fill="#3B82F6" rx="1"/>
        <rect className="viz-chip-bg" x="92"  y="158" width="8" height="8" fill="#3B82F6" rx="1"/>
        <rect className="viz-chip-bg" x="180" y="158" width="8" height="8" fill="#3B82F6" rx="1"/>

        {/* Confidence label chip */}
        <rect  className="viz-chip-bg"  x="92" y="9" width="75" height="16" rx="3" fill="#2563EB"/>
        <text  className="viz-chip-txt" x="129" y="20.5" textAnchor="middle" fill="white" fontSize="8.5" fontWeight="700">person · 0.97</text>

        {/* ── Crop shade panels (9:16 window) ── */}
        <rect className="viz-shade" x="0"   y="0" width="60"  height="172" fill="rgba(0,0,0,0.68)"/>
        <rect className="viz-shade" x="220" y="0" width="60"  height="172" fill="rgba(0,0,0,0.68)"/>
        {/* Crop guide lines */}
        <line className="viz-crop-line" x1="60"  y1="0" x2="60"  y2="172" stroke="#10B981" strokeWidth="1.5" strokeDasharray="5 3"/>
        <line className="viz-crop-line" x1="220" y1="0" x2="220" y2="172" stroke="#10B981" strokeWidth="1.5" strokeDasharray="5 3"/>
        {/* 9:16 label */}
        <rect className="viz-crop-line" x="66" y="4" width="26" height="11" rx="2" fill="#10B981" opacity="0.15"/>
        <text className="viz-crop-line" x="79" y="13" textAnchor="middle" fill="#10B981" fontSize="7" fontWeight="700">9:16</text>

        {/* ── Karaoke subtitle ── */}
        <rect  className="viz-sub-bg"  x="63"  y="150" width="154" height="16" rx="4" fill="rgba(10,16,30,0.9)"/>
        <text  className="viz-sub-txt" x="140"  y="161" textAnchor="middle" fill="#94A3B8" fontSize="7.5">every joy reminded</text>
        {/* Highlighted word */}
        <text  className="viz-sub-txt" x="140"  y="161" textAnchor="middle" fill="#94A3B8" fontSize="7.5">
          <tspan fill="#94A3B8">every joy </tspan>
          <tspan fill="#FFFFFF" fontWeight="700">reminded</tspan>
          <tspan fill="#94A3B8"> him...</tspan>
        </text>
      </svg>
    </div>
  )
}

// ── How it works ─────────────────────────────────────────────────
const STEPS = [
  {
    n: '01',
    viz: TranscribeViz,
    title: 'Transcribe',
    body: 'Your episode is transcribed in parallel. Audio chunks process concurrently so even a two-hour podcast finishes in the background while you keep working. Every word gets a precise timestamp the AI can reason over.',
    tag: 'faster-whisper',
  },
  {
    n: '02',
    viz: AnalyzeViz,
    title: 'Analyze',
    body: "Claude reads your entire transcript and finds moments that are genuinely shareable: a strong opening hook, a clear arc, and a satisfying close. It writes a scroll-stopping title for each clip, concise and curiosity-driven, grounded in what was actually said.",
    tag: 'Claude AI',
  },
  {
    n: '03',
    viz: EditViz,
    title: 'Edit',
    body: "Every person is detected and tracked per frame. The crop mirrors the editor's cuts — one panel for close-ups, a stacked split when the shot goes wide — or, in comedian mode, stays locked on a single performer and never cuts to the audience. Word-by-word captions, an optional vibe-matched music bed, a fading title card, and a clean audio fade round out each reel, ready to upload.",
    tag: 'YOLOv8 + MediaPipe + FFmpeg',
  },
]

function HowItWorks() {
  return (
    <section className="how-section" id="how">
      <div className="container">
        <div className="how-head">
          <p className="eyebrow">How it works</p>
          <h2 className="h2">Three stages.<br /><em>Fully automated.</em></h2>
          <p className="lead">Point PodClipper at any podcast episode and walk away. Transcription, clip selection, cropping, captions, and export, all handled.</p>
        </div>
        <div className="steps">
          {STEPS.map((s, i) => <StepCard key={s.n} step={s} delay={i * 80} />)}
        </div>
      </div>
    </section>
  )
}

function StepCard({ step, delay }) {
  const [ref, visible] = useReveal()
  const Viz = step.viz
  return (
    <div ref={ref} className={`step${visible ? ' visible' : ''}`} style={{ '--delay': `${delay}ms` }}>
      <Viz />
      <div className="step-body">
        <div className="step-top">
          <span className="step-n">{step.n}</span>
        </div>
        <h3 className="step-h">{step.title}</h3>
        <p className="step-p">{step.body}</p>
        <span className="step-tag">{step.tag}</span>
      </div>
    </div>
  )
}

// ── Stack ─────────────────────────────────────────────────────────
const STACK = [
  { label: 'Transcription', title: 'faster-whisper (CTranslate2)', desc: 'One model, multiple threads. A full hour of podcast audio transcribes in the background with no per-worker memory overhead. Every word lands with a precise timestamp.' },
  { label: 'AI Analysis',   title: 'Anthropic Claude',             desc: "Claude reads the full transcript and selects clips with a strong hook, a clear arc, and a satisfying close, then writes a scroll-stopping title for each. After rendering, a second LLM pass scores every reel on hook, arc, payoff, and standalone clarity, with a publish, review, or skip verdict." },
  { label: 'Speaker Tracking', title: 'YOLOv8 + MediaPipe Pose', desc: "YOLOv8 finds every person and MediaPipe Pose anchors the framing. The renderer mirrors the editor's cuts — a single 9:16 panel for close-ups, a stacked dual-panel when the shot goes wide — with body-IoU locking so it re-frames only on real movement, not landmark jitter. Comedian mode instead locks one performer, using brightness and full-body geometry to reject the audience sitting in shadow." },
  { label: 'Music & Export', title: 'FFmpeg + OpenCV + PIL', desc: 'OpenCV computes the crop window and pipes raw frames to FFmpeg; PIL burns the karaoke word highlights and the fading title. Optionally, an LLM scores a curated music library section-by-section for the clip’s vibe and sidechain-ducks the best match under the speech. Cached intermediates make re-runs near-instant.' },
]

function Stack() {
  const [ref, visible] = useReveal()
  return (
    <section className="stack-section">
      <div className="container">
        <div ref={ref} className={`stack-head${visible ? ' visible' : ''}`}>
          <p className="eyebrow">Under the hood</p>
          <h2 className="h2">Every component chosen<br /><em>for a reason.</em></h2>
        </div>
        <div className="stack-grid">
          {STACK.map((c, i) => <StackCard key={c.label} card={c} delay={i * 70} />)}
        </div>
      </div>
    </section>
  )
}

function StackCard({ card, delay }) {
  const [ref, visible] = useReveal()
  return (
    <div ref={ref} className={`scard${visible ? ' visible' : ''}`} style={{ '--delay': `${delay}ms` }}>
      <span className="scard-label">{card.label}</span>
      <h3 className="scard-h">{card.title}</h3>
      <p className="scard-p">{card.desc}</p>
    </div>
  )
}

// ── CTA Banner ────────────────────────────────────────────────────
function CTABanner() {
  const [ref, visible] = useReveal()
  return (
    <section className="cta-section" ref={ref}>
      <div className={`cta-inner${visible ? ' visible' : ''}`}>
        <h2 className="cta-h">Your episode archive is a goldmine.</h2>
        <p className="cta-p">Every podcast you've ever recorded has shareable moments in it. PodClipper finds them for you. Free, open source, and runs entirely on your machine.</p>
        <a href={GH} className="btn-cta" target="_blank" rel="noopener noreferrer">
          <GithubIcon size={16} />
          View on GitHub
        </a>
      </div>
    </section>
  )
}

// ── Footer ────────────────────────────────────────────────────────
function Footer() {
  return (
    <footer className="footer">
      <div className="footer-left">
        <span className="footer-logo">✂ PodClipper</span>
        <span className="footer-sub">by <strong>Loukik Naik</strong></span>
      </div>
      <a href={GH} className="footer-gh" target="_blank" rel="noopener noreferrer">
        <GithubIcon size={14} />
        github.com/LoukikNaik/PodClipper
      </a>
    </footer>
  )
}

// ── App ───────────────────────────────────────────────────────────
export default function App() {
  return (
    <>
      <Nav />
      <Hero />
      <StatsBar />
      <Demo />
      <HowItWorks />
      <Stack />
      <CTABanner />
      <Footer />
    </>
  )
}
