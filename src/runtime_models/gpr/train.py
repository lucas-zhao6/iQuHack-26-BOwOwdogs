#!/usr/bin/env python3
"""
GPR Runtime Model Training

Trains per-fold and full GPR runtime models using the standard feature pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    RBF,
    WhiteKernel,
    Matern,
    RationalQuadratic,
    DotProduct,
    ExpSineSquared,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.threshold_models.features import get_feature_columns
from src.runtime_models.gpr.predictor import GPRRuntimePredictor
from src.runtime_models.gpr.runtime_model import GPRRuntimeModel
from src.evaluation.cv_splits import load_cv_splits

OUTPUT_DIR = PROJECT_ROOT / "outputs"
FEATURES_PATH = OUTPUT_DIR / "training_features.pkl"
CV_SPLITS_PATH = OUTPUT_DIR / "cv_splits.json"
MODEL_DIR = OUTPUT_DIR / "runtime_models" / "runtime_gpr"
OPTUNA_TRIALS = 300


def load_data() -> pd.DataFrame:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Feature file not found: {FEATURES_PATH}\n"
            "Run 'python run_phase1.py' first."
        )
    return pd.read_pickle(FEATURES_PATH)


def prepare_features(df: pd.DataFrame):
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

    y_threshold = df["selected_threshold"].values.astype(float)
    y_total_time = df["forward_wall_s"].values.astype(float)

    return X.values, y_threshold, y_total_time, feature_cols


def apply_threshold_feature(
    X_train: np.ndarray,
    X_test: np.ndarray,
    thr_train: np.ndarray,
    thr_test: np.ndarray,
    threshold_feature: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if threshold_feature == "raw":
        thr_train_use = thr_train
        thr_test_use = thr_test
    elif threshold_feature == "both":
        thr_train_use = np.column_stack(
            [np.log2(np.clip(thr_train, 1, 512)), thr_train]
        )
        thr_test_use = np.column_stack(
            [np.log2(np.clip(thr_test, 1, 512)), thr_test]
        )
    else:
        thr_train_use = np.log2(np.clip(thr_train, 1, 512))
        thr_test_use = np.log2(np.clip(thr_test, 1, 512))

    return X_train, X_test, thr_train_use, thr_test_use


def suggest_gpr_params(trial: optuna.Trial) -> dict:
    const_lower = trial.suggest_float("const_lower", 1e-3, 1e-1, log=True)
    const_upper = trial.suggest_float("const_upper", 10.0, 2000.0, log=True)
    len_lower = trial.suggest_float("len_lower", 1e-3, 1e-1, log=True)
    len_upper = trial.suggest_float("len_upper", 10.0, 2000.0, log=True)
    noise_lower = trial.suggest_float("noise_lower", 1e-12, 1e-8, log=True)
    noise_upper = trial.suggest_float("noise_upper", 1e-4, 1e-1, log=True)
    alpha = trial.suggest_float("alpha", 1e-10, 1e-3, log=True)
    n_restarts = trial.suggest_int("n_restarts_optimizer", 0, 15)
    normalize_y = trial.suggest_categorical("normalize_y", [True, False])
    scaler_type = trial.suggest_categorical("scaler_type", ["standard", "robust", "none"])
    threshold_feature = trial.suggest_categorical("threshold_feature", ["log2", "raw", "both"])
    max_train_size = trial.suggest_categorical("max_train_size", [None, 500, 1000, 2000])

    kernel_family = trial.suggest_categorical(
        "kernel_family",
        ["rbf", "matern_1p5", "matern_2p5", "rq", "dot", "exp_sine"],
    )
    if kernel_family == "rbf":
        base_kernel = RBF(1.0, (len_lower, len_upper))
    elif kernel_family == "matern_1p5":
        base_kernel = Matern(1.0, (len_lower, len_upper), nu=1.5)
    elif kernel_family == "matern_2p5":
        base_kernel = Matern(1.0, (len_lower, len_upper), nu=2.5)
    elif kernel_family == "dot":
        base_kernel = DotProduct(sigma_0=1.0) ** 2
    elif kernel_family == "exp_sine":
        periodicity = trial.suggest_float("periodicity", 0.1, 10.0, log=True)
        base_kernel = ExpSineSquared(1.0, periodicity, (len_lower, len_upper))
    else:
        alpha_rq = trial.suggest_float("rq_alpha", 0.1, 10.0, log=True)
        base_kernel = RationalQuadratic(1.0, alpha_rq, (len_lower, len_upper))

    kernel = ConstantKernel(1.0, (const_lower, const_upper)) * base_kernel + WhiteKernel(
        1e-3, (noise_lower, noise_upper)
    )

    return {
        "kernel": kernel,
        "alpha": alpha,
        "normalize_y": normalize_y,
        "n_restarts_optimizer": n_restarts,
        "max_train_size": max_train_size,
        "scaler_type": scaler_type,
        "threshold_feature": threshold_feature,
    }


def tune_hyperparameters(
    X: np.ndarray,
    y_threshold: np.ndarray,
    y_total_time: np.ndarray,
    splits: list[dict],
    n_trials: int = 40,
) -> dict:
    n_splits = len(splits)
    print("\n  Hyperparameter search (Optuna TPE, group-aware CV):")
    print(f"  Trials: {n_trials}, CV splits: {n_splits}")

    def objective(trial: optuna.Trial) -> float:
        params = suggest_gpr_params(trial)
        all_pred = []
        all_true = []
        threshold_feature = params.pop("threshold_feature")

        for split in splits:
            train_idx = split["train_idx"]
            test_idx = split["test_idx"]

            X_train = X[train_idx]
            X_test = X[test_idx]

            X_train, X_test, thr_train, thr_test = apply_threshold_feature(
                X_train,
                X_test,
                y_threshold[train_idx],
                y_threshold[test_idx],
                threshold_feature,
            )

            try:
                model = GPRRuntimeModel(**params)
                model.fit(X_train, thr_train, y_total_time[train_idx])
                pred = model.predict(X_test, thr_test)
            except Exception:
                return float("-inf")
            all_pred.extend(pred.tolist())
            all_true.extend(y_total_time[test_idx].tolist())

        all_pred = np.asarray(all_pred)
        all_true = np.asarray(all_true)
        valid = np.isfinite(all_pred) & np.isfinite(all_true) & (all_pred > 0) & (all_true > 0)
        if valid.sum() == 0:
            return float("-inf")
        err = np.abs(np.log2(all_pred[valid] / all_true[valid]))
        return float(-np.mean(err))

    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=max(5, n_splits))
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_trial.params
    const_lower = best_params["const_lower"]
    const_upper = best_params["const_upper"]
    len_lower = best_params["len_lower"]
    len_upper = best_params["len_upper"]
    noise_lower = best_params["noise_lower"]
    noise_upper = best_params["noise_upper"]

    kernel_family = best_params["kernel_family"]
    if kernel_family == "rbf":
        base_kernel = RBF(1.0, (len_lower, len_upper))
    elif kernel_family == "matern_1p5":
        base_kernel = Matern(1.0, (len_lower, len_upper), nu=1.5)
    elif kernel_family == "matern_2p5":
        base_kernel = Matern(1.0, (len_lower, len_upper), nu=2.5)
    elif kernel_family == "dot":
        base_kernel = DotProduct(sigma_0=1.0) ** 2
    elif kernel_family == "exp_sine":
        base_kernel = ExpSineSquared(1.0, best_params["periodicity"], (len_lower, len_upper))
    else:
        base_kernel = RationalQuadratic(1.0, best_params["rq_alpha"], (len_lower, len_upper))

    tuned_kernel = ConstantKernel(1.0, (const_lower, const_upper)) * base_kernel + WhiteKernel(
        1e-3, (noise_lower, noise_upper)
    )

    print(f"\n  Best runtime configuration (score={study.best_value:.5f})")
    return {
        "kernel": tuned_kernel,
        "alpha": best_params["alpha"],
        "normalize_y": best_params["normalize_y"],
        "n_restarts_optimizer": best_params["n_restarts_optimizer"],
        "max_train_size": best_params["max_train_size"],
        "scaler_type": best_params["scaler_type"],
        "threshold_feature": best_params["threshold_feature"],
        "best_runtime_score": study.best_value,
        "n_trials": len(study.trials),
    }


def main() -> None:
    if not CV_SPLITS_PATH.exists():
        raise FileNotFoundError(
            f"CV splits not found: {CV_SPLITS_PATH}\n"
            "Run 'python src/evaluation/scripts/prep_cv_splits.py' first."
        )

    print("Loading training data...")
    df = load_data()
    valid = df["selected_threshold"].notna() & df["forward_wall_s"].notna()
    df = df[valid].reset_index(drop=True)

    print("Loading CV splits...")
    splits, meta = load_cv_splits(CV_SPLITS_PATH)
    if meta.get("n_samples") is not None and meta["n_samples"] != len(df):
        raise ValueError(
            "CV splits do not match current dataset size. "
            f"Splits have n_samples={meta['n_samples']} but data has {len(df)} rows."
        )

    print("Building feature matrix...")
    X, y_threshold, y_total_time, feature_cols = prepare_features(df)
    X = np.asarray(X)
    y_threshold = np.asarray(y_threshold)
    y_total_time = np.asarray(y_total_time)
    print(f"Features: {len(feature_cols)} | Samples: {len(df)}")

    best_params = tune_hyperparameters(
        X,
        y_threshold,
        y_total_time,
        splits,
        n_trials=OPTUNA_TRIALS,
    )
    scaler_type = best_params.get("scaler_type", "standard")
    threshold_feature = best_params.get("threshold_feature", "log2")
    model_params = {
        k: v
        for k, v in best_params.items()
        if k in {"kernel", "alpha", "normalize_y", "n_restarts_optimizer", "max_train_size", "scaler_type"}
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for fold, split in enumerate(splits):
        print(f"\nTraining fold {fold}...")
        train_idx = split["train_idx"]

        model = GPRRuntimeModel(**model_params)
        X_train, _X_unused, thr_train, _thr_unused = apply_threshold_feature(
            X[train_idx],
            X[train_idx],
            y_threshold[train_idx],
            y_threshold[train_idx],
            threshold_feature,
        )
        model.fit(X_train, thr_train, y_total_time[train_idx])

        predictor = GPRRuntimePredictor(runtime_model=model)
        predictor.feature_columns = feature_cols
        predictor.enriched_feature_columns = feature_cols
        predictor.scaler_type = scaler_type
        predictor.threshold_feature = threshold_feature

        model_path = MODEL_DIR / f"fold_{fold}.pkl"
        predictor.save(model_path)
        print(f"Saved fold {fold} model: {model_path}")

    print("\nTraining full model...")
    full_model = GPRRuntimeModel(**model_params)
    X_full, _X_unused, thr_full, _thr_unused = apply_threshold_feature(
        X,
        X,
        y_threshold,
        y_threshold,
        threshold_feature,
    )
    full_model.fit(X_full, thr_full, y_total_time)

    full_predictor = GPRRuntimePredictor(runtime_model=full_model)
    full_predictor.feature_columns = feature_cols
    full_predictor.enriched_feature_columns = feature_cols
    full_predictor.scaler_type = scaler_type
    full_predictor.threshold_feature = threshold_feature

    full_path = MODEL_DIR / "full_model.pkl"
    full_predictor.save(full_path)
    print(f"Saved full model: {full_path}")


if __name__ == "__main__":
    main()
