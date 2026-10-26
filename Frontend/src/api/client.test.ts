import { afterEach, describe, expect, it, vi } from 'vitest'
import { getHealth, predictFromBlob } from './client'

describe('api client', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('getHealth returns parsed JSON on success', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({ status: 'ok', model_loaded: true, model_path_configured: true }),
      }),
    )
    const health = await getHealth()
    expect(health.model_loaded).toBe(true)
    expect(health.status).toBe('ok')
  })

  it('predictFromBlob throws detail message on error', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: false,
        status: 400,
        json: async () => ({ detail: 'Empty canvas: please draw a digit before predicting.' }),
      }),
    )
    await expect(predictFromBlob(new Blob(['x'], { type: 'image/png' }))).rejects.toThrow(
      /Empty canvas/,
    )
  })
})
