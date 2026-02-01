# Circuit Fingerprint Challenge – Training Pipeline

This repository contains a feature pipeline, shared CV split generation, and model training/evaluation scripts for the Circuit Fingerprint Challenge.

## Quickstart

### 1) Prepare features + CV splits

```
python src/data/scripts/prep_features.py
python src/evaluation/scripts/prep_cv_splits.py
```

`src/evaluation/scripts/prep_cv_splits.py` is the script that writes `outputs/cv_splits.json`, which is shared by **all** model training/evaluation.

### 2) Prepare per-model data (features only)

Each model has a `prep.py` that regenerates features and **expects** CV splits to already exist.

```
python src/models/lgbm/prep.py
python src/models/naive_bucket/prep.py
python src/models/lgbm_curve/prep.py
```

### 3) Train models

**Main LGBM model**

```
python src/models/lgbm/train.py
```

This trains per-fold and full models and saves them to:
- `outputs/combined_model/fold_*.pkl`
- `outputs/combined_model/full_model.pkl`

**Naive bucket baseline**

```
python src/models/naive_bucket/train.py
```

This trains per-fold and full models and saves them to:
- `outputs/naive_bucket/fold_*.pkl`
- `outputs/naive_bucket/full_model.pkl`

**LGBM curve-fit fidelity model**

```
python src/models/lgbm_curve/train.py
```

This trains per-fold and full models and saves them to:
- `outputs/lgbm_curve/fold_*.pkl`
- `outputs/lgbm_curve/full_model.pkl`

### 4) Compare models

```
python run_compare.py
```

This loads the pretrained models and evaluates them on the fixed CV splits in `outputs/cv_splits.json`.

## Script Map

- Data prep
  - `src/data/scripts/prep_features.py`
- CV splits (shared across all models)
  - `src/evaluation/scripts/prep_cv_splits.py`
- LGBM model
  - `src/models/lgbm/prep.py`
  - `src/models/lgbm/train.py`
- Naive baseline
  - `src/models/naive_bucket/prep.py`
  - `src/models/naive_bucket/train.py`
- LGBM curve-fit fidelity model
  - `src/models/lgbm_curve/prep.py`
  - `src/models/lgbm_curve/train.py`
- Evaluation
  - `run_compare.py`

## Outputs

Key artifacts written to `outputs/`:
- `training_features.pkl` / `training_features.csv`
- `cv_splits.json` (shared CV splits)
- `combined_model/` (LGBM fold + full models)
- `naive_bucket/` (Naive fold + full models)
- `lgbm_curve/` (curve-fit fold + full models)

## Notes

- The CV split generator ensures consistent withheld circuits across models.
- Runtime scoring uses `-abs(log2(pred/true))`.
