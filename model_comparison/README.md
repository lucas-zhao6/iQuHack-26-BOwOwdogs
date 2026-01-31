Model Comparison Workspace

Use this folder for alternative modeling pipelines. The key reuse points are:

- Feature extraction can be different per model.
- Evaluation is shared: use `src/evaluation.py` for leave-one-circuit-out CV
  and result analysis.

Suggested pattern:

1) Build your own feature matrix `X` and targets
2) Implement a `fit_predict_fn` that matches the signature in
   `src/evaluation.leave_one_circuit_out_cv`
3) Call the CV function and optionally `analyze_cv_results`

Example outline:

```
from src.evaluation import leave_one_circuit_out_cv, analyze_cv_results

# Build df, X, y_threshold, y_total_time, y_setup_time, y_per_shot_time

def fit_predict_fn(X_train, X_test, y_thr_train, y_time_train, y_setup_train, y_per_shot_train, test_families):
    # train models, return predictions
    return pred_thresholds, pred_times

cv = leave_one_circuit_out_cv(df, X, y_threshold, y_total_time, y_setup_time, y_per_shot_time, fit_predict_fn)
analyze_cv_results(cv, df)
```
