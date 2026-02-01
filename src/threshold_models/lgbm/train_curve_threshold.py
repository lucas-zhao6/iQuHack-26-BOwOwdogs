#!/usr/bin/env python3
"""
Train the LGBM curve-based threshold model.
"""

from __future__ import annotations

import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
import warnings

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="X does not have valid feature names*",
)

from src.threshold_models.features import get_feature_columns
from src.evaluation.cv_splits import load_cv_splits
from src.evaluation.scoring import THRESHOLD_RUNGS, compute_threshold_metrics
from src.threshold_models.lgbm.curve_threshold_model import (
    LGBMCurveThresholdModel,
    TARGET_FIDELITY,
)

OUTPUT_DIR = PROJECT_ROOT / "outputs"
FEATURES_PATH = OUTPUT_DIR / "training_features.pkl"
CV_SPLITS_PATH = OUTPUT_DIR / "cv_splits.json"
MODEL_DIR = OUTPUT_DIR / "threshold_models" / "threshold_lgbm_curve"
OPTUNA_TRIALS = 120


def load_data() -> pd.DataFrame:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Feature file not found: {FEATURES_PATH}\n"
            "Run 'python src/data/scripts/prep_features.py' first."
        )
    return pd.read_pickle(FEATURES_PATH)


def build_feature_matrix(df: pd.DataFrame):
    feature_cols = get_feature_columns(df.columns.tolist())
    valid_cols = []
    for col in feature_cols:
        if col in df.columns:
            series = df[col]
            if series.dtype in [np.float64, np.int64, float, int]:
                if series.notna().sum() > 0 and series.std() > 1e-10:
                    valid_cols.append(col)
    feature_cols = valid_cols

    X = df[feature_cols].copy()
    X["is_gpu"] = (df["backend"] == "GPU").astype(float)
    X["is_double"] = (df["precision"] == "double").astype(float)
    feature_cols = feature_cols + ["is_gpu", "is_double"]

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    return X.values, feature_cols


def extract_curve_targets(df: pd.DataFrame) -> dict[int, np.ndarray]:
    curves = {}
    for rung in THRESHOLD_RUNGS:
        col = f"sweep_fid_{rung}"
        if col in df.columns:
            curves[rung] = np.asarray(df[col].values.astype(float))
    return curves


def suggest_lgbm_params(
    trial: optuna.Trial,
    prefix: str,
) -> dict:
    max_depth = trial.suggest_int(f"{prefix}max_depth", 3, 6)
    max_leaves = min(2 ** max_depth, 63)
    num_leaves = trial.suggest_int(f"{prefix}num_leaves", max(8, 2 ** (max_depth - 1)), max_leaves)
    return {
        "objective": "quantile",
        "alpha": 1.0 / 6.0,
        "max_depth": max_depth,
        "num_leaves": num_leaves,
        "learning_rate": trial.suggest_float(f"{prefix}learning_rate", 0.01, 0.08, log=True),
        "min_child_samples": trial.suggest_int(f"{prefix}min_child_samples", 8, 40),
        "min_gain_to_split": trial.suggest_float(f"{prefix}min_gain_to_split", 0.0, 0.1),
        "feature_fraction": trial.suggest_float(f"{prefix}feature_fraction", 0.6, 0.95),
        "bagging_fraction": trial.suggest_float(f"{prefix}bagging_fraction", 0.6, 0.95),
        "bagging_freq": trial.suggest_int(f"{prefix}bagging_freq", 1, 5),
        "lambda_l1": trial.suggest_float(f"{prefix}lambda_l1", 0.0, 6.0),
        "lambda_l2": trial.suggest_float(f"{prefix}lambda_l2", 0.0, 12.0),
        "max_bin": trial.suggest_int(f"{prefix}max_bin", 63, 255),
        "extra_trees": trial.suggest_categorical(f"{prefix}extra_trees", [True, False]),
        "verbosity": -1,
        "random_state": 42,
    }


def tune_hyperparameters(
    X: np.ndarray,
    y_curves: dict[int, np.ndarray],
    y_threshold: np.ndarray,
    splits: list[dict],
    n_trials: int = 120,
) -> dict:
    n_splits = len(splits)
    print("\n  Hyperparameter search (Optuna TPE, group-aware CV):")
    print(f"  Trials: {n_trials}, CV splits: {n_splits}")

    def objective(trial: optuna.Trial) -> float:
        lgb_params = suggest_lgbm_params(trial, "curve_")
        all_pred: list[int] = []
        all_true: list[int] = []
        for split in splits:
            train_idx = split["train_idx"]
            test_idx = split["test_idx"]
            val_idx = split.get("val_idx", np.array([], dtype=int))

            model = LGBMCurveThresholdModel(lgb_params=lgb_params)
            eval_set = (
                (np.asarray(X[val_idx]), {r: np.asarray(y_curves[r][val_idx]) for r in y_curves})
                if val_idx.size > 0
                else None
            )
            model.fit(
                np.asarray(X[train_idx]),
                {r: np.asarray(y_curves[r][train_idx]) for r in y_curves},
                eval_set=eval_set,
            )
            pred_thr = model.predict_threshold(
                np.asarray(X[test_idx]),
                target_fidelity=TARGET_FIDELITY,
                thresholds=THRESHOLD_RUNGS,
            )
            all_pred.extend([int(v) for v in pred_thr])
            all_true.extend([int(v) for v in y_threshold[test_idx]])

        scores = compute_threshold_metrics(all_pred, all_true)
        return scores["mean_threshold_score"]

    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=max(5, n_splits))
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_trial.params
    best_lgb_params = {k.replace("curve_", ""): v for k, v in best_params.items() if k.startswith("curve_")}

    print(f"\n  Best curve configuration (thr_score={study.best_value:.5f})")
    return {
        "lgb_params": best_lgb_params,
        "best_curve_threshold_score": study.best_value,
        "n_trials": len(study.trials),
    }


def main() -> None:
    if not CV_SPLITS_PATH.exists():
        raise FileNotFoundError(
            f"CV splits not found: {CV_SPLITS_PATH}\n"
            "Run 'python src/evaluation/scripts/prep_cv_splits.py' first."
        )

    df = load_data()
    valid = df["selected_threshold"].notna() & df["forward_wall_s"].notna()
    df = df[valid].reset_index(drop=True)

    splits, meta = load_cv_splits(CV_SPLITS_PATH)
    if meta.get("n_samples") is not None and meta["n_samples"] != len(df):
        raise ValueError(
            "CV splits do not match current dataset size. "
            f"Splits have n_samples={meta['n_samples']} but data has {len(df)} rows."
        )

    X, feature_cols = build_feature_matrix(df)
    X = np.asarray(X)
    y_curves = extract_curve_targets(df)

    best_params = tune_hyperparameters(
        X,
        y_curves,
        df["selected_threshold"].values.astype(float),
        splits,
        n_trials=OPTUNA_TRIALS,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for fold, split in enumerate(splits):
        train_idx = split["train_idx"]
        val_idx = split.get("val_idx", np.array([], dtype=int))

        model = LGBMCurveThresholdModel(lgb_params=best_params["lgb_params"])
        eval_set = (
            (np.asarray(X[val_idx]), {r: np.asarray(y_curves[r][val_idx]) for r in y_curves})
            if val_idx.size > 0
            else None
        )
        model.fit(
            np.asarray(X[train_idx]),
            {r: np.asarray(y_curves[r][train_idx]) for r in y_curves},
            eval_set=eval_set,
        )

        model_path = MODEL_DIR / f"fold_{fold}.pkl"
        payload = {
            "model": model,
            "feature_cols": feature_cols,
            "rungs": THRESHOLD_RUNGS,
            "target_fidelity": TARGET_FIDELITY,
        }
        with model_path.open("wb") as f:
            pickle.dump(payload, f)
        print(f"Saved fold {fold} model: {model_path}")

    full_model = LGBMCurveThresholdModel(lgb_params=best_params["lgb_params"])
    full_model.fit(np.asarray(X), {r: np.asarray(y_curves[r]) for r in y_curves})

    full_path = MODEL_DIR / "full_model.pkl"
    payload = {
        "model": full_model,
        "feature_cols": feature_cols,
        "rungs": THRESHOLD_RUNGS,
        "target_fidelity": TARGET_FIDELITY,
    }
    with full_path.open("wb") as f:
        pickle.dump(payload, f)
    print(f"Saved full model: {full_path}")


if __name__ == "__main__":
    main()
