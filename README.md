# Circuit Fingerprint Challenge – Training Pipeline

This repository contains a feature pipeline, shared CV split generation, and model training/evaluation scripts for the Circuit Fingerprint Challenge.

## Quickstart

### 1) Prepare features + CV splits

```
python src/data/scripts/prep_features.py
python src/evaluation/scripts/prep_cv_splits.py
```

`src/evaluation/scripts/prep_cv_splits.py` is the script that writes `outputs/cv_splits.json`, which is shared by **all** model training/evaluation.

### 2) Train models

**Threshold models**

```
python src/threshold_models/lgbm/train.py
python src/threshold_models/naive_bucket/train.py
```

**Runtime models**

```
python src/runtime_models/lgbm/train.py
python src/runtime_models/naive_bucket/train.py
```

### 3) Compare models

```
python run_compare_threshold.py
python run_compare_runtime.py
```

These load the pretrained models and evaluate them on the fixed CV splits in `outputs/cv_splits.json`.

## Script Map

- Data prep
  - `src/data/scripts/prep_features.py`
- CV splits (shared across all models)
  - `src/evaluation/scripts/prep_cv_splits.py`
- Threshold models
  - `src/threshold_models/lgbm/train.py`
  - `src/threshold_models/naive_bucket/train.py`
- Runtime models
  - `src/runtime_models/lgbm/train.py`
  - `src/runtime_models/naive_bucket/train.py`
- Evaluation
  - `run_compare_threshold.py`
  - `run_compare_runtime.py`

## Outputs

Key artifacts written to `outputs/`:
- `training_features.pkl` / `training_features.csv`
- `cv_splits.json` (shared CV splits)
- `threshold_models/threshold_lgbm/` (LGBM fold + full models)
- `threshold_models/threshold_naive/` (Naive fold + full models)
- `runtime_models/runtime_lgbm/` (LGBM fold + full models)
- `runtime_models/runtime_naive/` (Naive fold + full models)

## Notes

- The CV split generator ensures consistent withheld circuits across models.
- Runtime scoring uses `-abs(log2(pred/true))`.
