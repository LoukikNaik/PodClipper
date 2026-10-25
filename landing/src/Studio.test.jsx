// @vitest-environment jsdom
import { Component, StrictMode } from 'react'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, expect, test, vi } from 'vitest'
import App from './App.jsx'

const job = {
  id: 'other-job', title: 'A podcast from another account', state: 'running', stage: 'exporting',
  clips: 5, reels: [], url: 'https://youtu.be/abcdefghijk', created_at: 1700000000,
  owner: { name: 'Another listener', email: 'listener@example.com' },
}
const list = { jobs: [job], total: 1, counts: { running: 1 }, offset: 0, limit: 50 }
const response = data => ({ ok: true, json: async () => data })
let detail
let retention

class RenderCheck extends Component {
  state = { failed: false }
  static getDerivedStateFromError() { return { failed: true } }
  render() { return this.state.failed ? <p role="alert">Page crashed</p> : this.props.children }
}

beforeEach(() => {
  window.history.replaceState({}, '', '/admin')
  retention = 7
  detail = Promise.resolve(response(job))
  vi.stubGlobal('fetch', vi.fn(async (url, options = {}) => {
    const path = new URL(url).pathname
    if (path === '/config') return response({ max_video_minutes: 180, retention_days: 7, max_reels: 30 })
    if (path === '/me') return response({ is_admin: true, name: 'Owner', email: 'owner@example.com', used: 1, limit: 5, retention_days: retention })
    if (path === '/jobs' && options.method === 'POST') return response({ id: 'own-job' })
    if (path === '/jobs') return response([{ ...job, id: 'own-job', title: 'My own podcast', owner: undefined }])
    if (path === '/jobs/own-job') return response({ ...job, id: 'own-job', title: 'My own podcast', owner: undefined })
    if (path === '/admin/jobs') return response(list)
    if (path === '/admin/jobs/other-job') return detail
    throw new Error(`Unexpected request: ${path}`)
  }))
})

afterEach(() => { cleanup(); vi.unstubAllGlobals() })

test('admin list → job → browser Back → Forward → dashboard link works without refresh', async () => {
  let resolveDetail
  detail = new Promise(resolve => { resolveDetail = resolve })
  render(<RenderCheck><StrictMode><App /></StrictMode></RenderCheck>)
  await screen.findByRole('heading', { name: 'All podcasts' })
  fireEvent.click(screen.getByRole('link', { name: /A podcast from another account/ }))
  await waitFor(() => expect(window.location.pathname).toBe('/admin/jobs/other-job'))
  expect(screen.queryByText('Page crashed')).toBeNull()
  expect(await screen.findByText('Opening admin dashboard…')).toBeTruthy()
  await act(async () => { resolveDetail(response(job)) })
  await screen.findByRole('heading', { name: job.title, level: 1 })

  act(() => window.history.back())
  await screen.findByRole('heading', { name: 'All podcasts' })
  expect(window.location.pathname).toBe('/admin')
  act(() => window.history.forward())
  await screen.findByRole('heading', { name: job.title, level: 1 })
  fireEvent.click(screen.getByRole('link', { name: '← Admin dashboard' }))
  await screen.findByRole('heading', { name: 'All podcasts' })
  expect(screen.queryByText('Page crashed')).toBeNull()
})

test('opening an admin job directly then using its Back link returns to the list', async () => {
  window.history.replaceState({}, '', '/admin/jobs/other-job')
  render(<RenderCheck><StrictMode><App /></StrictMode></RenderCheck>)
  await screen.findByRole('heading', { name: job.title, level: 1 })
  fireEvent.click(screen.getByRole('link', { name: '← Admin dashboard' }))
  await screen.findByRole('heading', { name: 'All podcasts' })
  expect(screen.queryByText('Page crashed')).toBeNull()
})

test('workspace job Back link returns to the user’s podcast list', async () => {
  window.history.replaceState({}, '', '/try')
  render(<RenderCheck><StrictMode><App /></StrictMode></RenderCheck>)
  await screen.findByRole('heading', { name: 'My podcasts' })
  fireEvent.click(screen.getByRole('link', { name: /My own podcast/ }))
  await screen.findByRole('heading', { name: 'My own podcast', level: 1 })
  fireEvent.click(screen.getByRole('link', { name: '← All podcasts' }))
  await screen.findByRole('heading', { name: 'My podcasts' })
  expect(screen.queryByText('Page crashed')).toBeNull()
})

test('submission finds reels automatically without asking for or sending a count', async () => {
  window.history.replaceState({}, '', '/try')
  render(<RenderCheck><StrictMode><App /></StrictMode></RenderCheck>)
  await screen.findByRole('heading', { name: 'My podcasts' })
  expect(screen.queryByRole('combobox')).toBeNull()
  fireEvent.change(screen.getByLabelText('Start with a YouTube link'), { target: { value: 'https://youtu.be/abcdefghijk' } })
  fireEvent.click(screen.getByRole('button', { name: 'Find reels' }))
  await screen.findByRole('heading', { name: 'My own podcast', level: 1 })
  const submission = fetch.mock.calls.find(([url, options]) => url.endsWith('/jobs') && options.method === 'POST')
  expect(Object.keys(JSON.parse(submission[1].body)).sort()).toEqual(['request_key', 'url'])
})

test('a processing job does not present its duration-based cap as a promised reel count', async () => {
  window.history.replaceState({}, '', '/admin/jobs/other-job')
  detail = Promise.resolve(response({ ...job, selected_count: 28 }))
  render(<App />)
  await screen.findByRole('heading', { name: job.title, level: 1 })
  expect(screen.queryByText(/0.*28.*reels ready/)).toBeNull()
  expect(await screen.findByText('Preparing reels…')).toBeTruthy()
})

test('finished exports show their actual ready-for-review count without a denominator', async () => {
  window.history.replaceState({}, '', '/admin/jobs/other-job')
  detail = Promise.resolve(response({ ...job, selected_count: 28, reels: [
    { id: 'reel-one', title: 'A finished moment', duration: 30 },
  ] }))
  render(<App />)
  await screen.findByRole('heading', { name: job.title, level: 1 })
  expect(await screen.findByText('1 reel ready for review')).toBeTruthy()
  expect(screen.queryByText(/\/ 28/)).toBeNull()
})

test('old shared hash links open the same job at a clean URL', async () => {
  window.history.replaceState({}, '', '/#/admin/jobs/other-job')
  render(<App />)
  await screen.findByRole('heading', { name: job.title, level: 1 })
  expect(window.location.pathname).toBe('/admin/jobs/other-job')
  expect(window.location.hash).toBe('')
})

 test('podcast cards preview a ready reel through the authenticated poster endpoint', async () => {
  window.history.replaceState({}, '', '/admin')
  list.jobs = [{ ...job, reels: [{ id: 'ready-reel', title: 'A finished moment' }] }]
  try {
    render(<App />)
    const card = await screen.findByRole('link', { name: /A podcast from another account/ })
    expect(card.textContent).toContain('Processing · Making reels')
    const preview = card.querySelector('img')
    expect(preview).not.toBeNull()
    expect(preview.getAttribute('src')).toBe('https://podclipper-api.loukik.dev/admin/jobs/other-job/reels/ready-reel?poster=true')
    expect(preview.getAttribute('crossorigin')).toBe('use-credentials')
  } finally { list.jobs = [job] }
})

test('the owner workspace shows that its reels have no automatic expiry', async () => {
  retention = null
  window.history.replaceState({}, '', '/try')
  render(<App />)
  expect(await screen.findByText('No automatic expiry')).toBeTruthy()
  expect(screen.queryByText('Download within 7 days')).toBeNull()
})

test('opening a podcast fetches its details rather than repeatedly loading the whole library', async () => {
  window.history.replaceState({}, '', '/jobs/own-job')
  const original=fetch.getMockImplementation()
  fetch.mockImplementation(async (url, options) => new URL(url).pathname==='/jobs/own-job' ? response({...job,id:'own-job',title:'My own podcast'}) : original(url,options))
  render(<App />)
  await screen.findByRole('heading',{name:'My own podcast',level:1})
  expect(fetch.mock.calls.some(([url])=>new URL(url).pathname==='/jobs/own-job')).toBe(true)
})

test('unlimited owner can submit beyond five videos without a monthly cap display', async () => {
  window.history.replaceState({}, '', '/try')
  const original=fetch.getMockImplementation()
  fetch.mockImplementation(async (url, options) => new URL(url).pathname==='/me' ? response({is_admin:true,name:'Owner',used:8,limit:null,retention_days:null}) : original(url,options))
  render(<App />)
  expect(await screen.findByText('Unlimited videos')).toBeTruthy()
  expect(screen.getByRole('button',{name:'Find reels'}).disabled).toBe(false)
  expect(screen.queryByText('Resets on the 1st · UTC')).toBeNull()
})
