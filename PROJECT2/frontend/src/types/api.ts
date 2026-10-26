export interface DatasetInfo {
  name: string
  display_name: string
  task_type: string
  n_samples: number
  n_features: number
  n_classes: number
  feature_names: string[]
  target_names: string[]
  class_distribution: Record<string, number>
  feature_ranges: Record<string, { min: number; max: number; mean: number; std: number }>
  description: string
  positive_class: string | null
}

export interface Metrics {
  accuracy: number
  precision: number
  recall: number
  f1: number
  average: string
  positive_class: string | null
  confusion_matrix: number[][]
  roc_auc: number | null
  roc_curve: { fpr: number[]; tpr: number[]; thresholds: (number | null)[] } | null
  predict_seconds: number
  n_test_samples: number
  classification_report?: Record<string, unknown>
}

export interface TrainedModel {
  model_id: string
  algorithm: 'random_forest' | 'adaboost'
  dataset: string
  dataset_display_name?: string
  task_type: string
  hyperparameters: Record<string, unknown>
  test_size: number
  random_seed: number
  n_train: number
  n_test: number
  feature_names: string[]
  target_names: string[]
  train_seconds: number
  metrics: Metrics
  feature_importance?: { feature: string; importance: number }[] | null
  adaboost_diagnostics?: {
    estimator_weights?: number[]
    estimator_errors?: number[]
  } | null
}

export interface PredictResponse {
  model_id: string
  dataset: string
  algorithm: string
  predicted_class: string
  predicted_index: number
  probabilities: Record<string, number> | null
  confidence: number | null
}

export interface CompareResponse {
  dataset: string
  task_type: string
  test_size: number
  random_seed: number
  n_train: number
  n_test: number
  target_names: string[]
  models: {
    random_forest: TrainedModel
    adaboost: TrainedModel
  }
}
