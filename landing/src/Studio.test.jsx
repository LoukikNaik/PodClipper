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

class RenderCheck extends Component {
  state = { failed: false }
  static getDerivedStateFromError() { return { failed: true } }
  render() { return this.state.failed ? <p role="alert">Page crashed</p> : this.props.children }
}

beforeEach(() => {
  window.history.replaceState({}, '', '/#/admin')
  detail = Promise.resolve(response(job))
  vi.stubGlobal('fetch', vi.fn(async url => {
    const path = new URL(url).pathname
    if (path === '/config') return response({ max_video_minutes: 180, retention_days: 7 })
    if (path === '/me') return response({ is_admin: true, name: 'Owner', email: 'owner@example.com', used: 1, limit: 5 })
    if (path === '/jobs') return response([{ ...job, id: 'own-job', title: 'My own podcast', owner: undefined }])
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
  await waitFor(() => expect(window.location.hash).toBe('#/admin/jobs/other-job'))
  expect(screen.queryByText('Page crashed')).toBeNull()
  expect(await screen.findByText('Opening admin dashboard…')).toBeTruthy()
  await act(async () => { resolveDetail(response(job)) })
  await screen.findByRole('heading', { name: job.title, level: 1 })

  act(() => window.history.back())
  await screen.findByRole('heading', { name: 'All podcasts' })
  expect(window.location.hash).toBe('#/admin')
  act(() => window.history.forward())
  await screen.findByRole('heading', { name: job.title, level: 1 })
  fireEvent.click(screen.getByRole('link', { name: '← Admin dashboard' }))
  await screen.findByRole('heading', { name: 'All podcasts' })
  expect(screen.queryByText('Page crashed')).toBeNull()
})

test('opening an admin job directly then using its Back link returns to the list', async () => {
  window.history.replaceState({}, '', '/#/admin/jobs/other-job')
  render(<RenderCheck><StrictMode><App /></StrictMode></RenderCheck>)
  await screen.findByRole('heading', { name: job.title, level: 1 })
  fireEvent.click(screen.getByRole('link', { name: '← Admin dashboard' }))
  await screen.findByRole('heading', { name: 'All podcasts' })
  expect(screen.queryByText('Page crashed')).toBeNull()
})

test('workspace job Back link returns to the user’s podcast list', async () => {
  window.history.replaceState({}, '', '/#/try')
  render(<RenderCheck><StrictMode><App /></StrictMode></RenderCheck>)
  await screen.findByRole('heading', { name: 'My podcasts' })
  fireEvent.click(screen.getByRole('link', { name: /My own podcast/ }))
  await screen.findByRole('heading', { name: 'My own podcast', level: 1 })
  fireEvent.click(screen.getByRole('link', { name: '← All podcasts' }))
  await screen.findByRole('heading', { name: 'My podcasts' })
  expect(screen.queryByText('Page crashed')).toBeNull()
})
