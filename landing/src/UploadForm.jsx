import StudioIcon from './StudioIcon.jsx'
import { useEffect, useRef, useState } from 'react'

const pause = ms => new Promise(resolve => setTimeout(resolve, ms))
const readBlob = blob => new Promise((resolve, reject) => {
  const reader = new FileReader()
  reader.onload = () => resolve(reader.result)
  reader.onerror = () => reject(new Error('This video could not be read. Choose the file again.'))
  reader.readAsArrayBuffer(blob)
})

async function signatureFor(file) {
  const sample = new Blob([file.slice(0, 65536), file.slice(Math.max(0, file.size - 65536))])
  const digest = await crypto.subtle.digest('SHA-256', new Uint8Array(await readBlob(sample)))
  return `${file.name}:${file.size}:${file.lastModified}:${Array.from(new Uint8Array(digest), b => b.toString(16).padStart(2, '0')).join('')}`
}

function savedUploads(key) {
  try {
    const value = JSON.parse(localStorage.getItem(key) || '[]')
    return Array.isArray(value) ? value : value?.signature ? [value] : []
  } catch { return [] }
}
function savedUpload(key, signature) {
  const uploads = savedUploads(key)
  return signature ? uploads.find(item => item.signature === signature) : uploads.filter(item => !item.submitted).at(-1)
}
function saveUpload(key, value, signature) {
  try {
    const target = value?.signature || signature || savedUpload(key)?.signature
    const uploads = savedUploads(key).filter(item => item.signature !== target)
    if (value) uploads.push(value)
    uploads.length ? localStorage.setItem(key, JSON.stringify(uploads)) : localStorage.removeItem(key)
  } catch { /* Resume still works within this tab. */ }
}

export default function UploadForm({ api, user, config, onDone, onRefresh = () => {}, onBusy = () => {} }) {
  const [file, setFile] = useState(null)
  const [phase, setPhase] = useState('idle')
  const [error, setError] = useState('')
  const [sent, setSent] = useState(0)
  const [uploadId, setUploadId] = useState(null)
  const controller = useRef(null)
  const session = useRef(null)
  const cancelled = useRef(false)
  const input = useRef(null)
  const storageKey = `podclipper-upload:${user.email}`
  const busy = phase !== 'idle'
  useEffect(() => { onBusy(busy) }, [busy, onBusy])
  useEffect(() => () => controller.current?.abort(), [])

  async function retry(operation, signal) {
    for (let attempt = 0; ; attempt++) {
      if (signal?.aborted) throw new DOMException('Upload stopped', 'AbortError')
      try { return await operation() }
      catch (e) {
        if (e.name === 'AbortError' || attempt >= 2 || (e.status && ![408, 409, 429].includes(e.status) && e.status < 500)) throw e
        await pause(500 * (attempt + 1))
      }
    }
  }

  async function upload(e) {
    e.preventDefault(); setError('')
    if (!file || file.size <= 0) { setError('Choose a video file first.'); return }
    if (file.size >= (config.max_upload_bytes || 1_000_000_000)) { setError('Choose a video under 1 GB.'); return }
    if (!/\.(mp4|mov|m4v|mkv|webm)$/i.test(file.name)) { setError('Choose an MP4, MOV, M4V, MKV, or WebM video.'); return }
    cancelled.current = false
    controller.current = new AbortController()
    const signal = controller.current.signal
    setPhase('starting'); setSent(0)
    try {
      const signature = await signatureFor(file)
      const previous = savedUpload(storageKey, signature)
      session.current = previous?.signature === signature ? previous : { signature, key: crypto.randomUUID() }
      saveUpload(storageKey, session.current)
      let job = await retry(() => api('/uploads', { body: { filename: file.name, size: file.size, request_key: session.current.key }, csrf: user.csrf, signal }), signal)
      if (['failed', 'cancelled', 'expired'].includes(job.state)) {
        session.current = { signature, key: crypto.randomUUID() }
        saveUpload(storageKey, session.current)
        job = await api('/uploads', { body: { filename: file.name, size: file.size, request_key: session.current.key }, csrf: user.csrf, signal })
      }
      session.current.id = job.id; setUploadId(job.id); saveUpload(storageKey, session.current)
      if (job.state !== 'uploading') { if (job.stage === 'validating') saveUpload(storageKey, { ...session.current, submitted: true }); else saveUpload(storageKey, null, signature); session.current = null; setUploadId(null); await onDone(job); return }
      const info = job.upload
      const complete = new Set(info.completed_chunks)
      const partSize = index => Math.min(info.chunk_bytes, file.size - index * info.chunk_bytes)
      const updateProgress = () => setSent(Array.from(complete).reduce((total, index) => total + partSize(index), 0))
      updateProgress(); setPhase('uploading')
      const missing = Array.from({ length: info.total_chunks }, (_, i) => i).filter(i => !complete.has(i))
      let cursor = 0
      async function sendParts() {
        while (!signal.aborted && cursor < missing.length) {
          const index = missing[cursor++]
          const body = file.slice(index * info.chunk_bytes, Math.min(file.size, (index + 1) * info.chunk_bytes))
          await retry(() => api(`/uploads/${job.id}/chunks/${index}`, { raw: body, csrf: user.csrf, signal }), signal)
          complete.add(index); updateProgress()
        }
      }
      await Promise.all(Array.from({ length: Math.min(info.parallel_requests, 4, missing.length) }, sendParts))
      if (signal.aborted) throw new DOMException('Upload stopped', 'AbortError')
      setPhase('validating')
      const ready = await retry(() => api(`/uploads/${job.id}/complete`, { body: {}, csrf: user.csrf, signal }), signal)
      if (ready.stage === 'validating') saveUpload(storageKey, { ...session.current, submitted: true }); else saveUpload(storageKey, null, signature)
      session.current = null; setUploadId(null); setSent(file.size)
      await onDone(ready)
    } catch (e) {
      controller.current?.abort()
      if (!cancelled.current) {
        if (e.status === 422) saveUpload(storageKey, null, session.current?.signature)
        setError(e.name === 'AbortError' ? 'Upload paused. Choose the same file to resume.' : e.status === 422 ? `${e.message} Your allowance has been returned.` : e.message + (session.current?.id ? ' Completed upload parts are saved for retry.' : ''))
        await Promise.resolve(onRefresh()).catch(() => {})
      }
    } finally { if (!cancelled.current) setPhase('idle') }
  }

  async function cancel() {
    cancelled.current = true; controller.current?.abort(); setPhase('cancelling'); setError('')
    try {
      const pending = session.current || savedUpload(storageKey)
      await retry(() => api(`/uploads/${pending.id}/cancel`, { body: {}, csrf: user.csrf }))
      saveUpload(storageKey, null, pending.signature); session.current = null; setUploadId(null); setSent(0); setFile(null)
      if (input.current) input.current.value = ''
      await onRefresh()
    } catch (e) { setError(e.message) }
    finally { setPhase('idle') }
  }

  const percent = file?.size ? Math.min(100, Math.round(sent / file.size * 100)) : 0
  return <form className="studio-submit studio-upload" onSubmit={upload}>
    <label htmlFor="video-file">Choose a video</label><p className="studio-form-intro">Bring your own recording. We’ll find the moments worth keeping.</p>
    <div className="studio-submit-row"><div className="studio-file-picker"><StudioIcon name="upload" size={28} /><span className="studio-file-description" aria-hidden="true">{file ? file.name : 'Select a video from your device'}<small>{file ? `${(file.size / 1e6).toFixed(1)} MB` : 'MP4, MOV, M4V, MKV, or WebM · Under 1 GB'}</small></span><input ref={input} id="video-file" type="file" accept=".mp4,.mov,.m4v,.mkv,.webm,video/mp4,video/quicktime,video/webm" disabled={busy} onChange={e => { setFile(e.target.files?.[0] || null); setError(''); setSent(0) }} /></div>
      <button type="submit" disabled={busy || !file}>{phase === 'validating' ? 'Checking video…' : busy ? 'Uploading…' : 'Upload and find reels'}</button></div>
    <p>Video under 1 GB, up to {config.max_video_minutes} minutes. MP4, MOV, M4V, MKV, or WebM with audio. Uses one of your {user.limit} monthly videos.</p>
    {busy && <div className="studio-upload-progress" role="status"><label htmlFor="upload-progress">{phase === 'validating' ? 'Upload complete. Checking your video before processing.' : phase === 'cancelling' ? 'Cancelling upload…' : `Uploading ${percent}%`}</label><progress id="upload-progress" value={sent} max={file?.size || 1} />
      <p>Keep this tab open during the upload. Afterward, processing continues in the background.</p>
      {uploadId && phase === 'uploading' && <button type="button" className="studio-upload-cancel" onClick={cancel}>Cancel upload</button>}</div>}
    {!busy && savedUpload(storageKey) && <div className="studio-muted"><p>An unfinished upload is saved. Choose the same file to resume it.</p>{savedUpload(storageKey)?.id && <button type="button" className="studio-upload-cancel" onClick={cancel}>Discard unfinished upload</button>}</div>}
    {error && <p className="studio-error" role="alert">{error}</p>}
  </form>
}
