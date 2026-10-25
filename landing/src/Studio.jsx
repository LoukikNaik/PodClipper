import { useCallback, useEffect, useRef, useState } from 'react'
import './Studio.css'
import StudioIcon from './StudioIcon.jsx'
import UploadForm from './UploadForm.jsx'
import { navigate } from './navigation.js'

const API = (import.meta.env.VITE_API_URL || 'https://podclipper-api.loukik.dev').replace(/\/$/, '')
const stages = ['queued', 'downloading', 'transcribing', 'selecting', 'exporting']
const labels = { validating: 'Checking video', running: 'Processing', uploading: 'Uploading video', cancelled: 'Cancelled', queued: 'Queued', downloading: 'Downloading', transcribing: 'Transcribing', selecting: 'Selecting moments', exporting: 'Making reels', completed: 'Ready', partial: 'Some reels ready', failed: 'Needs a retry', expired: 'Results expired' }
function reelSummary(job) {
  const count = job.ready_count ?? job.reels.length
  if (count) return `${count} ${count === 1 ? 'reel' : 'reels'} ready for review`
  return ['uploading', 'queued', 'running'].includes(job.state) ? 'Preparing reels…' : 'No reels ready'
}

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

async function request(path, { body, raw, csrf, ...options } = {}) {
  const response = await fetch(`${API}${path}`, {
    credentials: 'include', ...options,
    headers: { ...((body || raw) ? { 'Content-Type': raw ? 'application/octet-stream' : 'application/json', 'X-Podclipper-Request': '1' } : {}), ...(csrf ? { 'X-CSRF-Token': csrf } : {}) },
    ...((body || raw) ? { method: 'POST', body: raw || JSON.stringify(body) } : {}),
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
    <p>Paste a YouTube link or upload your video. We’ll find the moments, follow the speakers, and add captions.</p>
    <div className="studio-google" ref={button} />
    {busy && <p role="status">Signing you in…</p>}
    {error && <div className="studio-error" role="alert">{error} <button onClick={() => setAttempt(x => x + 1)}>Try sign-in again</button></div>}
    <p className="studio-login-note">5 videos each month. Your jobs and reels stay in your account.<br />By continuing, you acknowledge our <a href="/privacy/">privacy policy</a>.</p>
  </section>
}

function JobDetail({ job, csrf, refresh, admin = false, resumeUpload }) {
  const [error, setError] = useState('')
  const [retrying, setRetrying] = useState(false)
  const active = ['queued', 'running'].includes(job.state)
  const jobStages = job.source_type === 'upload' ? ['queued', 'validating', 'transcribing', 'selecting', 'exporting'] : stages
  const stage = jobStages.indexOf(job.stage)
  async function retry() {
    setRetrying(true); setError('')
    try { await request(`/jobs/${job.id}/retry`, { body: {}, csrf }); await refresh() }
    catch (e) { setError(e.message) }
    finally { setRetrying(false) }
  }
  async function cancelUpload() {
    setRetrying(true); setError('')
    try { await request(`/uploads/${job.id}/cancel`, { body: {}, csrf }); await refresh() }
    catch (e) { setError(e.message) }
    finally { setRetrying(false) }
  }
  return <section className="studio-detail" aria-labelledby="podcast-title">
    <a href={admin ? '/admin' : '/try'} className="studio-back">← {admin ? 'Admin dashboard' : 'All podcasts'}</a>
    {admin && <p className="studio-muted">Submitted by {job.owner.name || job.owner.email} · {job.owner.email}</p>}
    <div className="studio-detail-heading">
      <div><span className={`studio-status studio-status--${job.state}`}>{labels[job.state]}</span><h1 id="podcast-title">{job.title || 'Your podcast'}</h1>{job.source_type === 'upload' ? <span className="studio-muted">Uploaded video</span> : <a className="studio-source" href={job.url} target="_blank" rel="noopener noreferrer">Watch source on YouTube ↗</a>}</div>
      <span className="studio-ready-count"><StudioIcon name={job.reels.length ? "check" : "clock"} size={20} />{reelSummary(job)}</span>
    </div>
    {job.state === 'uploading' && <p className="studio-muted" role="status">{admin ? 'Waiting for the account owner to finish uploading.' : <>Uploaded {Math.round((job.upload?.received_bytes || 0) / 1e6)} MB of {Math.round((job.upload?.size || 0) / 1e6)} MB. <a href="/try" onClick={resumeUpload}>Choose the same file to resume it.</a> <button type="button" disabled={retrying} onClick={cancelUpload}>Cancel upload</button></>}</p>}
    {active && <>
      <div className="studio-processing"><ol className="studio-progress" aria-label="Processing progress">{jobStages.map((value, i) => <li key={value} aria-current={i === stage ? 'step' : undefined} className={i < stage ? 'done' : i === stage ? 'current' : ''}><span aria-hidden>{i < stage ? '✓' : i + 1}</span>{labels[value]}</li>)}</ol>
      <p className="studio-muted" role="status">{job.stage === 'queued' ? 'Waiting for a processing slot.' : job.stage === 'exporting' && job.discovery_complete === false ? 'Making reels while we keep finding more moments.' : 'Processing your podcast.'} You can close this tab and return later. Finished reels appear below.</p></div>
    </>}
    {job.error && <div className="studio-error" role="status">{job.error}{!admin && ['failed', 'partial'].includes(job.state) && (job.source_type === 'upload' && job.state === 'failed' && job.upload?.completed_chunks?.length === 0 ? <a href="/try" onClick={resumeUpload}>Choose another file</a> : <button disabled={retrying} onClick={retry}>{retrying ? 'Queuing retry…' : 'Retry this job'}</button>)}</div>}
    {error && <p className="studio-error" role="alert">{error}</p>}
    <div className="studio-reels">{job.reels.map((reel, i) => {
      const url = `${API}${admin ? '/admin' : ''}/jobs/${job.id}/reels/${reel.id}`
      return <article className="studio-reel" key={reel.id}>
        <video controls playsInline preload="none" crossOrigin="use-credentials" poster={`${url}?poster=true`} src={url} aria-label={reel.title} />
        <div className="studio-reel-info"><span>Reel {i + 1} · {Math.round(reel.duration)} sec</span><h2>{reel.title}</h2>{reel.verdict && <p className="studio-muted">Quality check: {({ publish: 'Recommended', review: 'Needs review', skip: 'Marked skip — available for you to review' })[reel.verdict]}</p>}<a href={`${url}?download=true`}><StudioIcon name="download" size={16} />Download MP4</a></div>
      </article>
    })}</div>
    {job.state === 'expired' && <p className="studio-muted">These files have passed their retention period. Your podcast history is still saved.</p>}
  </section>
}

function PodcastCard({ job, admin = false }) {
  const reel = job.state !== 'expired' && job.reels[0]
  const poster = reel ? `${API}${admin ? '/admin' : ''}/jobs/${job.id}/reels/${reel.id}?poster=true` : null
  const [failedPoster, setFailedPoster] = useState(null)
  return <a className="studio-job" href={`${admin ? '/admin' : ''}/jobs/${job.id}`}>
    <div className={`studio-job-art ${poster && failedPoster !== poster ? 'has-preview' : ''}`} aria-hidden="true">
      <div className="studio-job-frame"><StudioIcon name="video" size={32} /></div>
      {poster && failedPoster !== poster && <img src={poster} alt="" loading="lazy" crossOrigin="use-credentials" onError={() => setFailedPoster(poster)} />}
      <span className="studio-job-source">{job.source_type === 'upload' ? 'Uploaded video' : 'YouTube'}</span>
    </div>
    <div className="studio-job-body">
      <span className={`studio-status studio-status--${job.state}`}>{labels[job.state]}{job.state === 'running' ? ` · ${labels[job.stage] || job.stage}` : ''}</span>
      <h3>{job.title || 'Your podcast'}</h3>
      {admin && <p className="studio-admin-owner">{job.owner.name || job.owner.email} · {job.owner.email}</p>}
      <div className="studio-job-bottom"><div><p>{reelSummary(job)}</p><time dateTime={new Date(job.created_at * 1000).toISOString()}>{new Date(job.created_at * 1000).toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' })}</time></div><span className="studio-job-open"><StudioIcon name="arrow" size={18} /></span></div>
    </div>
  </a>
}

function AdminDashboard({ route }) {
  const jobId = route.startsWith('/admin/jobs/') ? route.slice('/admin/jobs/'.length) : null
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
        const result = await request(jobId ? `/admin/jobs/${encodeURIComponent(jobId)}` : `/admin/jobs?offset=${offset}&summary=true`)
        if (active) { setResult({ key: viewKey, data: result }); setError('') }
      } catch (e) {
        if (active) { setError(e.message); if ([401, 403, 404].includes(e.status)) setResult(null) }
      } finally { busy = false }
    }
    update()
    const timer = setInterval(() => { if (!document.hidden) update() }, 5000)
    return () => { active = false; clearInterval(timer) }
  }, [jobId, offset, viewKey])
  return <>
    {error && <p className="studio-error" role="alert">{error}</p>}
    {!view && !error && <p className="studio-loading" role="status">Opening admin dashboard…</p>}
    {view && (jobId ? <JobDetail job={view} admin /> : <section className="studio-admin">
      <a href="/try" className="studio-back">← My workspace</a>
      <div className="studio-heading"><div><h1>Every podcast.<br />One place to follow along.</h1><p>Admin dashboard · All accounts · Refreshes every 5 seconds</p></div></div>
      <dl className="studio-admin-counts">{[['Total videos', view.total], ['Processing', view.counts.running || 0], ['Queued', view.counts.queued || 0], ['Ready', view.counts.completed || 0]].map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}</dl>
      <div className="studio-library-heading"><h2>All podcasts</h2><span>{view.total ? `${offset + 1}–${offset + view.jobs.length} of ${view.total}` : 'No submissions yet'}</span></div>
      <div className="studio-job-list">{view.jobs.map(job => <PodcastCard job={job} admin key={job.id} />)}</div>
      {view.total > view.limit && <nav className="studio-admin-pages" aria-label="Admin results pages"><button disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - view.limit))}>← Previous</button><button disabled={offset + view.limit >= view.total} onClick={() => setOffset(offset + view.limit)}>Next →</button></nav>}
    </section>)}
  </>
}

export default function Studio({ route }) {
  const [config, setConfig] = useState(null)
  const [user, setUser] = useState(null)
  const [jobs, setJobs] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [url, setUrl] = useState('')
  const [sending, setSending] = useState(false)
  const [mode, setMode] = useState('youtube')
  const [uploading, setUploading] = useState(false)
  const key = useRef(null)
  const [detail, setDetail] = useState(null)
  const selectedId = route.startsWith('/jobs/') ? route.slice('/jobs/'.length) : null
  const adminRoute = route === '/admin' || route.startsWith('/admin/')
  const refresh = useCallback(async ({ poll = false } = {}) => {
    try {
      if (!poll) setUser(await request('/me'))
      if (adminRoute) return
      if (selectedId) {
        try { setDetail(await request(`/jobs/${encodeURIComponent(selectedId)}`)) }
        catch (e) { if (e.status === 404) setDetail({ id: selectedId, missing: true }); else throw e }
      } else {
        setJobs(await request('/jobs?summary=true'))
      }
    } catch (e) {
      if (e.status === 401) { setUser(null); setJobs([]); setDetail(null); return }
      throw e
    }
  }, [selectedId, adminRoute])
  useEffect(() => {
    let active = true
    async function init() {
      try { const conf = await request('/config'); if (active) setConfig(conf); await refresh() }
      catch { if (active) setError('The processing server is unavailable. Your saved jobs are safe. Please try again shortly.') }
      finally { if (active) setLoading(false) }
    }
    init()
    return () => { active = false }
  }, [refresh])
  const selected = detail?.id === selectedId && !detail.missing ? detail : null
  const active = item => ['queued', 'running', 'uploading'].includes(item?.state)
  const busy = selectedId ? active(selected) : jobs.some(active)
  useEffect(() => {
    if (!user || adminRoute || (selected && !busy)) return
    let pending = false
    const interval = setInterval(async () => {
      if (document.hidden || pending) return
      pending = true
      try { await refresh({ poll: true }); setError(previous => previous.startsWith('Connection interrupted') ? '' : previous) }
      catch { setError('Connection interrupted. Reconnecting to your saved jobs…') }
      finally { pending = false }
    }, busy ? 5000 : 30000)
    return () => clearInterval(interval)
  }, [!!user, refresh, busy, adminRoute, selected?.state])
  async function submit(e) {
    e.preventDefault(); setSending(true); setError('')
    const signature = url.trim()
    if (!key.current || key.current.signature !== signature) key.current = { signature, id: crypto.randomUUID() }
    try {
      const job = await request('/jobs', { body: { url: url.trim(), request_key: key.current.id }, csrf: user.csrf })
      await refresh(); key.current = null; setUrl(''); navigate(`/jobs/${job.id}`)
    } catch (e) { setError(e.message) }
    finally { setSending(false) }
  }
  async function logout() {
    try { await request('/auth/logout', { body: {}, csrf: user.csrf }); window.google?.accounts.id.disableAutoSelect(); setUser(null); setJobs([]); navigate('/try') }
    catch (e) { setError(e.message) }
  }
  return <div className="studio">
    <header className="studio-header"><a href="/" className="studio-brand"><span className="studio-brand-mark"><StudioIcon name="scissors" size={21} /></span>PodClipper</a><div>{user?.is_admin && <a href={adminRoute ? '/try' : '/admin'}>{adminRoute ? 'My workspace' : 'Admin dashboard'}</a>}<a href="/privacy/">Privacy</a>{user && <><span>{user.name || user.email}</span><button onClick={logout}>Sign out</button></>}</div></header>
    <main className="studio-main">
      {loading ? <p className="studio-loading" role="status">Opening your workspace…</p> : <>
        {error && <div className="studio-error" role="alert">{error}{!config && <button onClick={() => window.location.reload()}>Reconnect</button>}</div>}
        {!user && config && <GoogleLogin config={config} onLogin={refresh} />}
        {user && (adminRoute ? (user.is_admin ? <AdminDashboard route={route} /> : <div className="studio-error" role="alert">This dashboard is only available to the administrator. <a href="/try">Go to my workspace</a></div>) : selectedId && detail?.id !== selectedId ? <p className="studio-loading" role="status">Opening podcast…</p> : selected ? <JobDetail job={selected} csrf={user.csrf} refresh={refresh} resumeUpload={() => setMode('upload')} /> : <>
          {selectedId && <p className="studio-error">This podcast was not found in your account.</p>}
          <div className="studio-heading"><div><h1>Long conversations.<br />Great little moments.</h1><p>Find the moments worth sharing. We’ll handle the edit.</p></div><div className="studio-allowance">{user.limit === null ? <><strong>Unlimited videos</strong><p>{user.used} used this month</p></> : <><strong>{user.used}<span> / {user.limit}</span></strong><p>videos used this month</p><div className="studio-allowance-track" aria-hidden="true">{Array.from({ length: user.limit }, (_, i) => <span className={i < user.used ? 'used' : ''} key={i} />)}</div><small>Resets on the 1st · UTC</small></>}</div></div>
          <section className="studio-composer" aria-label="Create reels"><div className="studio-input-modes" role="group" aria-label="Video source">
            <button type="button" aria-pressed={mode === 'youtube'} disabled={uploading || sending} onClick={() => setMode('youtube')}><StudioIcon name="link" size={17} />YouTube link</button>
            <button type="button" aria-pressed={mode === 'upload'} disabled={uploading || sending} onClick={() => setMode('upload')}><StudioIcon name="upload" size={17} />Upload video</button>
          </div>
          {mode === 'upload' ? <UploadForm api={request} config={config} user={user} onBusy={setUploading} onRefresh={refresh} onDone={async job => { await refresh(); navigate(`/jobs/${job.id}`) }} /> : <form className="studio-submit" onSubmit={submit}>
            <label htmlFor="podcast-url">Start with a YouTube link</label><p className="studio-form-intro">One video in. A collection of captioned, vertical reels out.</p>
            <div className="studio-submit-row"><input id="podcast-url" type="url" placeholder="https://www.youtube.com/watch?v=…" required value={url} onChange={e => setUrl(e.target.value)} /><button type="submit" disabled={sending || (user.limit !== null && user.used >= user.limit)}>{sending ? 'Adding podcast…' : 'Find reels'}<StudioIcon name="arrow" size={18} /></button></div>
            <p>We find strong moments automatically, up to {config.max_reels || 30} reels depending on video length.</p>
          </form>}<div className="studio-composer-notes"><span><StudioIcon name="clock" size={15} />Up to {config.max_video_minutes} minutes</span><span><StudioIcon name="video" size={15} />Vertical video + captions</span><span><StudioIcon name="download" size={15} />{user.retention_days === null ? 'No automatic expiry' : `Download within ${user.retention_days ?? config.retention_days} days`}</span></div></section>
          <section className="studio-library" aria-labelledby="library-title"><div className="studio-library-heading"><h2 id="library-title">My podcasts</h2><span>{jobs.length} {jobs.length === 1 ? 'podcast' : 'podcasts'}</span></div>
            {jobs.length === 0 ? <div className="studio-empty"><span className="studio-empty-icon"><StudioIcon name="scissors" size={28} /></span><h3>Your first podcast belongs here.</h3><p>Add a link above. You can leave the page while we work.</p></div> : <div className="studio-job-list">{jobs.map(job => <PodcastCard job={job} key={job.id} />)}</div>}
          </section>
        </>)}
      </>}
    </main>
    <footer className="studio-footer">Made with PodClipper. {user?.retention_days === null ? 'Your results are stored without automatic deletion.' : `Results are stored for ${user?.retention_days ?? config?.retention_days ?? 7} days, then automatically deleted.`}<a href={`${API}/source`}>Source code · AGPL-3.0</a></footer>
  </div>
}
