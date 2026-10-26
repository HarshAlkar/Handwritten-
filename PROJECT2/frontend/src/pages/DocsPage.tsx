export function DocsPage() {
  return (
    <div className="space-y-6">
      <h1 className="font-display text-3xl tracking-tight">Documentation</h1>

      <section className="card prose-block">
        <h2>Datasets</h2>
        <ul>
          <li>
            <strong>Breast Cancer Wisconsin</strong> — binary (malignant / benign), 569 samples, 30 features.
            Positive class for binary metrics: <em>benign</em> (label index 1 in scikit-learn).
          </li>
          <li>
            <strong>Iris</strong> — multiclass (setosa / versicolor / virginica), 150 samples, 4 features.
            Multiclass precision/recall/F1 use macro averaging.
          </li>
        </ul>
      </section>

      <section className="card prose-block">
        <h2>Algorithms</h2>
        <p>
          <strong>Random Forest</strong> builds many decision trees on bootstrap samples and random feature
          subsets, then aggregates predictions.
        </p>
        <p>
          <strong>AdaBoost</strong> trains shallow trees sequentially, up-weighting misclassified samples, and
          combines them with estimator weights (scikit-learn <code>AdaBoostClassifier</code> with
          <code>estimator=DecisionTreeClassifier</code>).
        </p>
      </section>

      <section className="card prose-block">
        <h2>Evaluation methodology</h2>
        <ul>
          <li>Stratified train/test split with a fixed random seed.</li>
          <li>
            <code>StandardScaler</code> fit on training data only, then applied to test/prediction inputs.
          </li>
          <li>Comparison mode uses one shared split for both algorithms.</li>
          <li>ROC-AUC for binary uses positive-class probabilities; multiclass uses OvR macro when available.</li>
        </ul>
      </section>

      <section className="card prose-block">
        <h2>Limitations & future work</h2>
        <ul>
          <li>Default datasets only; no arbitrary CSV upload.</li>
          <li>Importance is impurity-based, not SHAP/permutation importance.</li>
          <li>Future: cross-validation curves, more datasets, exportable experiment reports.</li>
        </ul>
      </section>
    </div>
  )
}
