import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { DigitCanvas } from './DigitCanvas'

describe('DigitCanvas', () => {
  it('shows empty-canvas hint initially', () => {
    render(<DigitCanvas brushSize={16} />)
    expect(screen.getByText(/Draw a digit/i)).toBeInTheDocument()
    expect(screen.getByRole('img', { name: /Drawing canvas/i })).toBeInTheDocument()
  })
})
