import { afterEach, describe, expect, it, vi } from 'vitest'
import { getHealth, trainModel } from './client'

describe('api client', () => {
  afterEach(() => vi.unstubAllGlobals())

  it('getHealth parses JSON', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({ status: 'ok' }),
      }),
    )
    await expect(getHealth()).resolves.toEqual({ status: 'ok' })
  })

  it('trainModel surfaces API errors', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: false,
        status: 400,
        json: async () => ({ detail: 'n_estimators must be between 1 and 500.' }),
      }),
    )
    await expect(
      trainModel({ dataset: 'breast_cancer', algorithm: 'random_forest' }),
    ).rejects.toThrow(/n_estimators/)
  })
})
