#!/usr/bin/env python3
"""
Train the EBM curve-based threshold model.
"""

from __future__ import annotations

import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import optuna

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.threshold_models.features import get_feature_columns
from src.evaluation.cv_splits import load_cv_splits
from src.evaluation.scoring import THRESHOLD_RUNGS, compute_threshold_metrics
from src.threshold_models.ebm.curve_threshold_model import (
    EBMCurveThresholdModel,
    EBMConfig,
    TARGET_FIDELITY,
)

OUTPUT_DIR = PROJECT_ROOT / "outputs"
FEATURES_PATH = OUTPUT_DIR / "training_features.pkl"
CV_SPLITS_PATH = OUTPUT_DIR / "cv_splits.json"
MODEL_DIR = OUTPUT_DIR / "threshold_models" / "threshold_ebm_curve"
OPTUNA_TRIALS = 60


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


def suggest_ebm_params(trial: optuna.Trial) -> dict:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "max_bins": trial.suggest_int("max_bins", 32, 128, step=32),
        "max_rounds": trial.suggest_int("max_rounds", 50, 300, step=50),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 10),
        "outer_bags": trial.suggest_int("outer_bags", 4, 16, step=4),
        "interactions": 0,
    }


def tune_hyperparameters(
    X: np.ndarray,
    y_curves: dict[int, np.ndarray],
    y_threshold: np.ndarray,
    feature_cols: list[str],
    splits: list[dict],
    n_trials: int = 60,
) -> dict:
    n_splits = len(splits)
    print("\n  Hyperparameter search (Optuna TPE, group-aware CV):")
    print(f"  Trials: {n_trials}, CV splits: {n_splits}")

    def objective(trial: optuna.Trial) -> float:
        params = suggest_ebm_params(trial)
        all_pred: list[int] = []
        all_true: list[int] = []

        for split in splits:
            train_idx = split["train_idx"]
            test_idx = split["test_idx"]

            config = EBMConfig(
                ebm_params=params,
                use_monotone_constraints=False,
                enforce_monotone_projection=True,
                threshold_feature_name="threshold_rung",
                binary_feature_names=("is_gpu", "is_double"),
            )
            model = EBMCurveThresholdModel(config=config)
            model.fit(
                np.asarray(X[train_idx]),
                {r: np.asarray(y_curves[r][train_idx]) for r in y_curves},
                feature_names=feature_cols,
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

    print(f"\n  Best curve configuration (thr_score={study.best_value:.5f})")
    return {
        "ebm_params": best_params,
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
        feature_cols=feature_cols,
        splits=splits,
        n_trials=OPTUNA_TRIALS,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    config = EBMConfig(
        ebm_params=best_params["ebm_params"],
        use_monotone_constraints=False,
        enforce_monotone_projection=True,
        threshold_feature_name="threshold_rung",
        binary_feature_names=("is_gpu", "is_double"),
    )

    for fold, split in enumerate(splits):
        train_idx = split["train_idx"]

        model = EBMCurveThresholdModel(config=config)
        model.fit(
            np.asarray(X[train_idx]),
            {r: np.asarray(y_curves[r][train_idx]) for r in y_curves},
            feature_names=feature_cols,
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

    full_model = EBMCurveThresholdModel(config=config)
    full_model.fit(
        np.asarray(X),
        {r: np.asarray(y_curves[r]) for r in y_curves},
        feature_names=feature_cols,
    )

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
