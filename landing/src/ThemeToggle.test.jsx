// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, expect, test, vi } from 'vitest'
import ThemeToggle from './ThemeToggle.jsx'

beforeEach(() => {
  localStorage.clear()
  delete document.documentElement.dataset.theme
  vi.stubGlobal('matchMedia',vi.fn(() => ({matches:true,addEventListener:vi.fn(),removeEventListener:vi.fn()})))
})
afterEach(() => {cleanup();vi.restoreAllMocks();vi.unstubAllGlobals()})

test('defaults to light even on a dark device and remembers explicit dark mode across pages', () => {
  const view=render(<ThemeToggle />)
  expect(document.documentElement.dataset.theme).toBe('light')
  fireEvent.click(screen.getByRole('button',{name:'Switch to dark mode'}))
  expect(document.documentElement.dataset.theme).toBe('dark')
  expect(localStorage.getItem('podclipper-theme')).toBe('dark')
  view.unmount()
  render(<ThemeToggle />)
  expect(document.documentElement.dataset.theme).toBe('dark')
})

test('theme switching works when saving preferences is blocked', () => {
  vi.spyOn(Storage.prototype,'setItem').mockImplementation(()=>{throw new Error('blocked')})
  render(<ThemeToggle />)
  fireEvent.click(screen.getByRole('button',{name:'Switch to dark mode'}))
  expect(document.documentElement.dataset.theme).toBe('dark')
})

test('defaults to light when reading saved preferences is blocked', () => {
  vi.spyOn(Storage.prototype,'getItem').mockImplementation(()=>{throw new Error('blocked')})
  render(<ThemeToggle />)
  expect(document.documentElement.dataset.theme).toBe('light')
})
