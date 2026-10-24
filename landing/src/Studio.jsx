import { useCallback, useEffect, useRef, useState } from 'react'
import './Studio.css'

const API = (import.meta.env.VITE_API_URL || 'https://podclipper-api.loukik.dev').replace(/\/$/, '')
const stages = ['queued', 'downloading', 'transcribing', 'selecting', 'exporting']
const labels = { queued: 'Queued', downloading: 'Downloading', transcribing: 'Transcribing', selecting: 'Selecting moments', exporting: 'Making reels', completed: 'Ready', partial: 'Some reels ready', failed: 'Needs a retry', expired: 'Results expired' }
let googleScript

function loadGoogle() {
  if (window.google?.accounts?.id) return Promise.resolve()
  if (!googleScript) googleScript = new Promise((resolve, reject) => {
    const script = document.createElement('script')
    script.src = 'https://accounts.google.com/gsi/client'
    script.async = true
    script.onload = resolve
    script.onerror = () => { script.remove(); googleScript = null; reject(new Error('Google sign-in could not load. Check your connection and try again.')) }
    document.head.appendChild(script)
  })
  return googleScript
}

async function request(path, { body, csrf, ...options } = {}) {
  const response = await fetch(`${API}${path}`, {
    credentials: 'include', ...options,
    headers: { ...(body ? { 'Content-Type': 'application/json', 'X-Podclipper-Request': '1' } : {}), ...(csrf ? { 'X-CSRF-Token': csrf } : {}) },
    ...(body ? { method: 'POST', body: JSON.stringify(body) } : {}),
  })
  const data = await response.json().catch(() => ({}))
  if (!response.ok) {
    const error = new Error(typeof data.detail === 'string' ? data.detail : 'This request could not be completed. Please try again.')
    error.status = response.status
    throw error
  }
  return data
}

function GoogleLogin({ config, onLogin }) {
  const button = useRef(null)
  const [error, setError] = useState('')
  const [attempt, setAttempt] = useState(0)
  const [busy, setBusy] = useState(false)
  useEffect(() => {
    let active = true
    setError('')
    Promise.all([loadGoogle(), request('/auth/challenge')]).then(([, challenge]) => {
      if (!active || !button.current) return
      window.google.accounts.id.initialize({
        client_id: config.google_client_id, nonce: challenge.nonce,
        auto_select: false,
        callback: async ({ credential }) => {
          setBusy(true)
          try {
            await request('/auth/google', { body: { credential } })
            await onLogin()
          } catch (e) { if (active) setError(e.message) }
          finally { if (active) setBusy(false) }
        },
      })
      button.current.replaceChildren()
      window.google.accounts.id.renderButton(button.current, { theme: 'outline', size: 'large', text: 'continue_with', shape: 'pill', width: 280 })
    }).catch(e => { if (active) setError(e.message) })
    return () => { active = false }
  }, [config.google_client_id, onLogin, attempt])
  return <section className="studio-login">
    <div className="studio-tape" aria-hidden><span /><span /><span /><span /><span /></div>
    <h1>Your next reel starts<br />with a conversation.</h1>
    <p>Paste a podcast. We’ll find the moments, follow the speakers, and add captions.</p>
    <div className="studio-google" ref={button} />
    {busy && <p role="status">Signing you in…</p>}
    {error && <div className="studio-error" role="alert">{error} <button onClick={() => setAttempt(x => x + 1)}>Try sign-in again</button></div>}
    <p className="studio-login-note">5 videos each month. Your jobs and reels stay in your account.<br />By continuing, you acknowledge our <a href="/privacy/">privacy policy</a>.</p>
  </section>
}

function JobDetail({ job, csrf, refresh, admin = false }) {
  const [error, setError] = useState('')
  const [retrying, setRetrying] = useState(false)
  const active = ['queued', 'running'].includes(job.state)
  const stage = stages.indexOf(job.stage)
  async function retry() {
    setRetrying(true); setError('')
    try { await request(`/jobs/${job.id}/retry`, { body: {}, csrf }); await refresh() }
    catch (e) { setError(e.message) }
    finally { setRetrying(false) }
  }
  return <section className="studio-detail" aria-labelledby="podcast-title">
    <a href={admin ? '#/admin' : '#/try'} className="studio-back">← {admin ? 'Admin dashboard' : 'All podcasts'}</a>
    {admin && <p className="studio-muted">Submitted by {job.owner.name || job.owner.email} · {job.owner.email}</p>}
    <div className="studio-detail-heading">
      <div><span className={`studio-status studio-status--${job.state}`}>{labels[job.state]}</span><h1 id="podcast-title">{job.title || 'Your podcast'}</h1><a className="studio-source" href={job.url} target="_blank" rel="noopener noreferrer">Watch source on YouTube ↗</a></div>
      <span className="studio-ready-count"><strong>{job.reels.length}</strong> / {job.clips} reels ready</span>
    </div>
    {active && <>
      <ol className="studio-progress" aria-label="Processing progress">{stages.map((value, i) => <li key={value} className={i < stage ? 'done' : i === stage ? 'current' : ''}><span aria-hidden>{i < stage ? '✓' : i + 1}</span>{labels[value]}</li>)}</ol>
      <p className="studio-muted" role="status">{job.stage === 'queued' ? 'Waiting for a processing slot.' : 'Processing your podcast.'} You can close this tab and return later. Finished reels appear below.</p>
    </>}
    {job.error && <div className="studio-error" role="status">{job.error}{!admin && ['failed', 'partial'].includes(job.state) && <button disabled={retrying} onClick={retry}>{retrying ? 'Queuing retry…' : 'Retry this job'}</button>}</div>}
    {error && <p className="studio-error" role="alert">{error}</p>}
    <div className="studio-reels">{job.reels.map((reel, i) => {
      const url = `${API}${admin ? '/admin' : ''}/jobs/${job.id}/reels/${reel.id}`
      return <article className="studio-reel" key={reel.id}>
        <video controls playsInline preload="none" crossOrigin="use-credentials" poster={`${url}?poster=true`} src={url} aria-label={reel.title} />
        <div className="studio-reel-info"><span>Reel {i + 1} · {Math.round(reel.duration)} sec</span><h2>{reel.title}</h2>{reel.verdict && <p className="studio-muted">Quality check: {({ publish: 'Recommended', review: 'Needs review', skip: 'Marked skip — available for you to review' })[reel.verdict]}</p>}<a href={`${url}?download=true`}>Download MP4 ↓</a></div>
      </article>
    })}</div>
    {job.state === 'expired' && <p className="studio-muted">These files have passed their retention period. Your podcast history is still saved.</p>}
  </section>
}

function AdminDashboard({ hash }) {
  const jobId = hash.startsWith('#/admin/jobs/') ? hash.slice('#/admin/jobs/'.length) : null
  const [offset, setOffset] = useState(0)
  const viewKey = jobId ? `job:${jobId}` : `list:${offset}`
  const [result, setResult] = useState(null)
  const view = result?.key === viewKey ? result.data : null
  const [error, setError] = useState('')
  useEffect(() => {
    let active = true
    let busy = false
    setResult(null); setError('')
    async function update() {
      if (busy) return
      busy = true
      try {
        const result = await request(jobId ? `/admin/jobs/${encodeURIComponent(jobId)}` : `/admin/jobs?offset=${offset}`)
        if (active) { setResult({ key: viewKey, data: result }); setError('') }
      } catch (e) {
        if (active) { setError(e.message); if ([401, 403, 404].includes(e.status)) setResult(null) }
      } finally { busy = false }
    }
    update()
    const timer = setInterval(update, 5000)
    return () => { active = false; clearInterval(timer) }
  }, [jobId, offset, viewKey])
  return <>
    {error && <p className="studio-error" role="alert">{error}</p>}
    {!view && !error && <p className="studio-loading" role="status">Opening admin dashboard…</p>}
    {view && (jobId ? <JobDetail job={view} admin /> : <section className="studio-admin">
      <a href="#/try" className="studio-back">← My workspace</a>
      <div className="studio-heading"><div><h1>Every podcast.<br />One place to follow along.</h1><p>Admin dashboard · All accounts · Refreshes every 5 seconds</p></div></div>
      <dl className="studio-admin-counts">{[['Total videos', view.total], ['Processing', view.counts.running || 0], ['Queued', view.counts.queued || 0], ['Ready', view.counts.completed || 0]].map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}</dl>
      <div className="studio-library-heading"><h2>All podcasts</h2><span>{view.total ? `${offset + 1}–${offset + view.jobs.length} of ${view.total}` : 'No submissions yet'}</span></div>
      <div className="studio-job-list">{view.jobs.map(job => <a className="studio-job" href={`#/admin/jobs/${job.id}`} key={job.id}><div><span className={`studio-status studio-status--${job.state}`}>{labels[job.state]}{job.state === 'running' ? ` · ${labels[job.stage] || job.stage}` : ''}</span><h3>{job.title || 'YouTube video'}</h3><p className="studio-admin-owner">{job.owner.name || job.owner.email} · {job.owner.email}</p><p>{new Date(job.created_at * 1000).toLocaleString()} · {job.reels.length} of {job.clips} reels ready</p></div><span aria-hidden>↗</span></a>)}</div>
      {view.total > view.limit && <nav className="studio-admin-pages" aria-label="Admin results pages"><button disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - view.limit))}>← Previous</button><button disabled={offset + view.limit >= view.total} onClick={() => setOffset(offset + view.limit)}>Next →</button></nav>}
    </section>)}
  </>
}

export default function Studio() {
  const [config, setConfig] = useState(null)
  const [user, setUser] = useState(null)
  const [jobs, setJobs] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [url, setUrl] = useState('')
  const [clips, setClips] = useState(5)
  const [sending, setSending] = useState(false)
  const [hash, setHash] = useState(window.location.hash)
  const key = useRef(null)
  const refresh = useCallback(async () => {
    try {
      const me = await request('/me')
      const list = await request('/jobs')
      setUser(me); setJobs(list)
    } catch (e) {
      if (e.status === 401) { setUser(null); setJobs([]); return }
      throw e
    }
  }, [])
  useEffect(() => {
    let active = true
    async function init() {
      try { const conf = await request('/config'); if (active) setConfig(conf); await refresh() }
      catch { if (active) setError('The processing server is unavailable. Your saved jobs are safe. Please try again shortly.') }
      finally { if (active) setLoading(false) }
    }
    init()
    const change = () => setHash(window.location.hash)
    window.addEventListener('hashchange', change)
    return () => { active = false; window.removeEventListener('hashchange', change) }
  }, [refresh])
  useEffect(() => {
    if (!user) return
    const interval = setInterval(() => refresh().then(() => setError(previous => previous.startsWith('Connection interrupted') ? '' : previous)).catch(() => setError('Connection interrupted. Reconnecting to your saved jobs…')), 5000)
    return () => clearInterval(interval)
  }, [!!user, refresh])
  async function submit(e) {
    e.preventDefault(); setSending(true); setError('')
    const signature = `${url.trim()}:${clips}`
    if (!key.current || key.current.signature !== signature) key.current = { signature, id: crypto.randomUUID() }
    try {
      const job = await request('/jobs', { body: { url: url.trim(), clips: Number(clips), request_key: key.current.id }, csrf: user.csrf })
      await refresh(); key.current = null; setUrl(''); window.location.hash = `/jobs/${job.id}`
    } catch (e) { setError(e.message) }
    finally { setSending(false) }
  }
  async function logout() {
    try { await request('/auth/logout', { body: {}, csrf: user.csrf }); window.google?.accounts.id.disableAutoSelect(); setUser(null); setJobs([]); window.location.hash = '/try' }
    catch (e) { setError(e.message) }
  }
  const selectedId = hash.startsWith('#/jobs/') ? hash.slice('#/jobs/'.length) : null
  const selected = jobs.find(job => job.id === selectedId)
  const adminRoute = hash === '#/admin' || hash.startsWith('#/admin/')
  return <div className="studio">
    <header className="studio-header"><a href="/" className="studio-brand">✂ PodClipper</a><div>{user?.is_admin && <a href={adminRoute ? '#/try' : '#/admin'}>{adminRoute ? 'My workspace' : 'Admin dashboard'}</a>}<a href="/privacy/">Privacy</a>{user && <><span>{user.name || user.email}</span><button onClick={logout}>Sign out</button></>}</div></header>
    <main className="studio-main">
      {loading ? <p className="studio-loading" role="status">Opening your workspace…</p> : <>
        {error && <div className="studio-error" role="alert">{error}{!config && <button onClick={() => window.location.reload()}>Reconnect</button>}</div>}
        {!user && config && <GoogleLogin config={config} onLogin={refresh} />}
        {user && (adminRoute ? (user.is_admin ? <AdminDashboard hash={hash} /> : <div className="studio-error" role="alert">This dashboard is only available to the administrator. <a href="#/try">Go to my workspace</a></div>) : selected ? <JobDetail job={selected} csrf={user.csrf} refresh={refresh} /> : <>
          {selectedId && <p className="studio-error">This podcast was not found in your account.</p>}
          <div className="studio-heading"><div><h1>Good conversations.<br />Ready for a second audience.</h1><p>Your podcasts, from first word to final reel.</p></div><div className="studio-allowance"><strong>{user.used}<span> / {user.limit}</span></strong><p>videos used this month</p><small>Resets on the 1st · UTC</small></div></div>
          <form className="studio-submit" onSubmit={submit}>
            <label htmlFor="podcast-url">Start with a YouTube link</label>
            <div className="studio-submit-row"><input id="podcast-url" type="url" placeholder="https://www.youtube.com/watch?v=…" required value={url} onChange={e => setUrl(e.target.value)} /><label className="studio-select">Reels<select value={clips} onChange={e => setClips(Number(e.target.value))}>{[5, 10, 15].map(n => <option key={n} value={n}>{n}</option>)}</select></label><button type="submit" disabled={sending || user.used >= user.limit}>{sending ? 'Adding podcast…' : 'Make reels'}</button></div>
            <p>Public videos up to {config.max_video_minutes} minutes. We aim for your chosen reel count; quality may mean fewer. Results stay available for {config.retention_days} days.</p>
          </form>
          <section className="studio-library" aria-labelledby="library-title"><div className="studio-library-heading"><h2 id="library-title">My podcasts</h2><span>{jobs.length} {jobs.length === 1 ? 'podcast' : 'podcasts'}</span></div>
            {jobs.length === 0 ? <div className="studio-empty"><span aria-hidden>✂</span><h3>Your first podcast belongs here.</h3><p>Add a link above. You can leave the page while we work.</p></div> : <div className="studio-job-list">{jobs.map(job => <a className="studio-job" href={`#/jobs/${job.id}`} key={job.id}><div><span className={`studio-status studio-status--${job.state}`}>{labels[job.state]}</span><h3>{job.title || 'YouTube podcast'}</h3><p>{new Date(job.created_at * 1000).toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' })} · {job.reels.length} of {job.clips} reels ready</p></div><span aria-hidden>↗</span></a>)}</div>}
          </section>
        </>)}
      </>}
    </main>
    <footer className="studio-footer">Made with PodClipper. Your workspace is private; the administrator can access results for support.<a href={`${API}/source`}>Source code · AGPL-3.0</a></footer>
  </div>
}
