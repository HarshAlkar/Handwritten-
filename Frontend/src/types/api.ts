export interface PredictResponse {
  predicted_digit: number
  confidence: number
  probabilities: Record<string, number>
}

export interface HealthResponse {
  status: string
  model_loaded: boolean
  model_path_configured: boolean
}

export interface ModelInfoResponse {
  name: string
  input_shape: number[]
  num_classes: number
  class_labels: number[]
  total_params: number | null
  layers: Array<{ name: string; type: string; output_shape: string }> | null
  framework: string | null
  evaluation: EvaluationMetrics | null
  model_available: boolean
}

export interface EvaluationMetrics {
  dataset?: string
  test_samples?: number
  test_loss?: number
  test_accuracy?: number
  precision_macro?: number
  recall_macro?: number
  f1_macro?: number
  confusion_matrix?: number[][]
  per_class?: Record<string, { precision?: number; recall?: number; 'f1-score'?: number; support?: number }>
  macro_avg?: Record<string, number>
  weighted_avg?: Record<string, number>
}

export interface MetricsResponse {
  history: {
    history?: {
      accuracy?: number[]
      val_accuracy?: number[]
      loss?: number[]
      val_loss?: number[]
    }
    epochs_trained?: number
    dataset?: string
    split_sizes?: Record<string, number>
    quick_test?: { accuracy: number; loss: number }
    config?: Record<string, unknown>
  } | null
  evaluation: EvaluationMetrics | null
  available: boolean
  message: string | null
}
