// @vitest-environment jsdom
import { webcrypto } from 'node:crypto'
import { cleanup, fireEvent, render, screen, waitFor, act } from '@testing-library/react'
import { afterEach, beforeEach, expect, test, vi } from 'vitest'
import UploadForm from './UploadForm.jsx'

beforeEach(() => { localStorage.clear(); vi.stubGlobal('crypto', webcrypto) })
afterEach(() => { cleanup(); vi.unstubAllGlobals() })
const user = { email: 'owner@example.com', csrf: 'csrf', used: 1, limit: 5 }
const config = { max_upload_bytes: 1_000_000_000, max_video_minutes: 180 }

test('resumes missing chunks with at most four parallel requests before queueing', async () => {
  let active = 0, peak = 0
  const pending = []
  const onDone = vi.fn()
  const api = vi.fn(async (path, options) => {
    if (path === '/uploads') return { id: 'upload-one', state: 'uploading', upload: { size: 20, chunk_bytes: 4, parallel_requests: 4, total_chunks: 5, completed_chunks: [0] } }
    if (path.endsWith('/complete')) return { id: 'upload-one', state: 'queued' }
    if (path.includes('/chunks/')) {
      active++; peak = Math.max(peak, active)
      return await new Promise(resolve => pending.push(() => { active--; resolve({}) }))
    }
    throw new Error('Unexpected request ' + path)
  })
  render(<UploadForm api={api} user={user} config={config} onDone={onDone} />)
  fireEvent.change(screen.getByLabelText('Choose a video'), { target: { files: [new File(['abcdefghijklmnopqrst'], 'podcast.mp4', { type: 'video/mp4', lastModified: 10 })] } })
  fireEvent.click(screen.getByRole('button', { name: 'Upload and find reels' }))
  await waitFor(() => expect(pending.length).toBe(4))
  expect(peak).toBe(4)
  expect(api.mock.calls.filter(([path]) => path.includes('/chunks/')).map(([path]) => path)).toEqual([1,2,3,4].map(i => `/uploads/upload-one/chunks/${i}`))
  expect(api.mock.calls.some(([path]) => path.endsWith('/complete'))).toBe(false)
  await act(async () => { pending.forEach(resolve => resolve()) })
  await waitFor(() => expect(onDone).toHaveBeenCalledWith({ id: 'upload-one', state: 'queued' }))
})

test('rejects a file at the 1 GB limit before starting an upload', async () => {
  const api = vi.fn()
  const file = new File(['x'], 'too-big.mp4', { type: 'video/mp4' })
  Object.defineProperty(file, 'size', { value: 1_000_000_000 })
  render(<UploadForm api={api} user={user} config={config} onDone={vi.fn()} />)
  fireEvent.change(screen.getByLabelText('Choose a video'), { target: { files: [file] } })
  fireEvent.click(screen.getByRole('button', { name: 'Upload and find reels' }))
  expect(await screen.findByRole('alert')).toHaveProperty('textContent', expect.stringContaining('under 1 GB'))
  expect(api).not.toHaveBeenCalled()
})

test('a transient chunk failure retries only that chunk', async () => {
  const api = vi.fn()
  let tries = 0
  api.mockImplementation(async path => {
    if (path === '/uploads') return { id: 'resume-one', state: 'uploading', upload: { size: 3, chunk_bytes: 32_000_000, parallel_requests: 4, total_chunks: 1, completed_chunks: [] } }
    if (path.includes('/chunks/')) { if (++tries === 1) throw new TypeError('Network interrupted'); return {} }
    return { id: 'resume-one', state: 'queued' }
  })
  const onDone = vi.fn()
  render(<UploadForm api={api} user={user} config={config} onDone={onDone} />)
  fireEvent.change(screen.getByLabelText('Choose a video'), { target: { files: [new File(['abc'], 'retry.mp4', { lastModified: 20 })] } })
  fireEvent.click(screen.getByRole('button', { name: 'Upload and find reels' }))
  await waitFor(() => expect(onDone).toHaveBeenCalled(), { timeout: 3000 })
  expect(tries).toBe(2)
  expect(api.mock.calls.filter(([path]) => path === '/uploads').length).toBe(1)
})

test('background validation retains the resume identity without showing an unfinished upload', async () => {
  const api=vi.fn(async path => path==='/uploads' ? {id:'validate-one',state:'uploading',upload:{size:3,chunk_bytes:32_000_000,parallel_requests:4,total_chunks:1,completed_chunks:[]}} : path.endsWith('/complete') ? {id:'validate-one',state:'queued',stage:'validating'} : {})
  const onDone=vi.fn()
  render(<UploadForm api={api} user={user} config={config} onDone={onDone} />)
  fireEvent.change(screen.getByLabelText('Choose a video'),{target:{files:[new File(['abc'],'validate.mp4',{lastModified:20})]}})
  fireEvent.click(screen.getByRole('button',{name:'Upload and find reels'}))
  await waitFor(()=>expect(onDone).toHaveBeenCalled())
  const saved=JSON.parse(localStorage.getItem(`podclipper-upload:${user.email}`))
  expect(saved).toHaveLength(1)
  expect(saved[0]).toMatchObject({id:'validate-one',submitted:true})
  expect(screen.queryByText(/An unfinished upload is saved/)).toBeNull()
})
