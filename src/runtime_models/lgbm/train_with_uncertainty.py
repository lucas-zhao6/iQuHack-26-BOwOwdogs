#!/usr/bin/env python3
"""
LGBM Runtime Model Training with Threshold Uncertainty Features

Assumes threshold_lgbm models are already trained.
Saves models to outputs/runtime_models/runtime_lgbm_uncertainty.
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import optuna

warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.threshold_models.features import get_feature_columns
from src.threshold_models.lgbm.predictor import ThresholdPredictor
from src.runtime_models.lgbm.predictor_uncertainty import RuntimePredictorWithThresholdUncertainty
from src.runtime_models.lgbm.runtime_model import RuntimeModel
from src.evaluation.scoring import compute_runtime_metrics
from src.evaluation.cv_splits import load_cv_splits, split_train_valid_groups
from src.data.feature_selection import FeatureSelector
from src.data.auxiliary_features import AuxiliaryFeaturePredictor, extract_auxiliary_targets

OUTPUT_DIR = PROJECT_ROOT / "outputs"
FEATURES_PATH = OUTPUT_DIR / "training_features.pkl"
CV_SPLITS_PATH = OUTPUT_DIR / "cv_splits.json"
THRESHOLD_DIR = OUTPUT_DIR / "threshold_models" / "threshold_lgbm"
MODEL_DIR = OUTPUT_DIR / "runtime_models" / "runtime_lgbm_uncertainty"
OPTUNA_TRIALS = 120
VALID_SIZE = 0.2


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
    y_setup_time = df["estimated_setup_s"].values.astype(float)
    y_per_shot_time = df["estimated_per_shot_s"].values.astype(float)

    return X.values, y_threshold, y_total_time, y_setup_time, y_per_shot_time, feature_cols


def enrich_with_auxiliary_features(
    X_train: np.ndarray,
    X_val: np.ndarray | None,
    X_test: np.ndarray,
    auxiliary_targets: dict,
    train_idx: np.ndarray,
):
    aux_predictor = AuxiliaryFeaturePredictor(n_estimators=150, max_depth=4, learning_rate=0.05)
    train_aux = {k: v[train_idx] for k, v in auxiliary_targets.items()}
    aux_predictor.fit(X_train, train_aux)

    X_train_aux = aux_predictor.predict_as_features(X_train)
    X_test_aux = aux_predictor.predict_as_features(X_test)
    X_train_enriched = np.hstack([X_train, X_train_aux])
    X_test_enriched = np.hstack([X_test, X_test_aux])

    X_val_enriched = None
    if X_val is not None:
        X_val_aux = aux_predictor.predict_as_features(X_val)
        X_val_enriched = np.hstack([X_val, X_val_aux])

    aux_feature_names = [f"pred_{t}" for t in aux_predictor.models.keys()]
    return X_train_enriched, X_val_enriched, X_test_enriched, aux_predictor, aux_feature_names


def compute_threshold_uncertainty_features(
    threshold_predictor: ThresholdPredictor,
    X_base: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    X_thr = X_base
    if threshold_predictor.aux_predictor is not None:
        aux_features = threshold_predictor.aux_predictor.predict_as_features(X_base)
        if aux_features.shape[1] > 0:
            X_thr = np.hstack([X_base, aux_features])

    if threshold_predictor.feature_selector is not None:
        X_thr = threshold_predictor.feature_selector.transform_threshold(X_thr)

    proba = threshold_predictor.threshold_model.predict_proba(X_thr)
    idxs = np.arange(proba.shape[1])
    expected = proba @ idxs
    entropy = -(proba * np.log(proba + 1e-9)).sum(axis=1)
    return expected, entropy


def append_runtime_extras(
    X_rt: np.ndarray,
    X_enriched: np.ndarray,
    X_base: np.ndarray,
    enriched_names: list[str],
    threshold_predictor: ThresholdPredictor,
) -> np.ndarray:
    extras = []
    expected, entropy = compute_threshold_uncertainty_features(threshold_predictor, X_base)
    extras.extend([expected, entropy])

    for flag_name in ["is_gpu", "is_double"]:
        if flag_name in enriched_names:
            flag_idx = enriched_names.index(flag_name)
            extras.append(X_enriched[:, flag_idx])

    extra_cols = [e.reshape(-1, 1) if e.ndim == 1 else e for e in extras]
    return np.hstack([X_rt] + extra_cols)


def suggest_lgbm_params(
    trial: optuna.Trial,
    prefix: str,
) -> dict:
    max_depth = trial.suggest_int(f"{prefix}max_depth", 3, 6)
    max_leaves = min(2 ** max_depth, 63)
    num_leaves = trial.suggest_int(f"{prefix}num_leaves", max(8, 2 ** (max_depth - 1)), max_leaves)
    return {
        "objective": "regression_l1",
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
    df: pd.DataFrame,
    X: np.ndarray,
    y_threshold: np.ndarray,
    y_total_time: np.ndarray,
    y_setup_time: np.ndarray,
    y_per_shot_time: np.ndarray,
    feature_names: list,
    splits: list[dict],
    n_trials: int = 120,
) -> dict:
    n_splits = len(splits)
    print("\n  Hyperparameter search (Optuna TPE, group-aware CV):")
    print(f"  Trials: {n_trials}, CV splits: {n_splits}, features: {len(feature_names)}")

    auxiliary_targets = extract_auxiliary_targets(df)

    def objective(trial: optuna.Trial) -> float:
        rt_params = suggest_lgbm_params(trial, "rt_")
        n_feat = trial.suggest_int("n_features", 20, min(80, len(feature_names)))
        feature_method = trial.suggest_categorical("feature_method", ["mi", "corr"])
        use_auxiliary_features = trial.suggest_categorical("use_auxiliary_features", [True, False])
        use_ensemble = trial.suggest_categorical("use_ensemble", [True, False])
        use_calibration = trial.suggest_categorical("use_calibration", [True, False])

        all_pred = []
        all_true = []

        for fold, split in enumerate(splits):
            threshold_path = THRESHOLD_DIR / f"fold_{fold}.pkl"
            if not threshold_path.exists():
                raise FileNotFoundError(f"Threshold model not found: {threshold_path}")
            threshold_predictor = ThresholdPredictor.load(threshold_path)

            train_idx = split["train_idx"]
            test_idx = split["test_idx"]
            val_idx = split.get("val_idx", np.array([], dtype=int))

            X_train, X_test = X[train_idx], X[test_idx]
            y_time_train, y_time_test = y_total_time[train_idx], y_total_time[test_idx]
            y_thr_train, y_thr_test = y_threshold[train_idx], y_threshold[test_idx]
            y_setup_train = y_setup_time[train_idx]
            y_per_shot_train = y_per_shot_time[train_idx]

            X_val = X[val_idx] if val_idx.size > 0 else None
            y_time_val = y_total_time[val_idx] if val_idx.size > 0 else None
            y_thr_val = y_threshold[val_idx] if val_idx.size > 0 else None
            y_setup_val = y_setup_time[val_idx] if val_idx.size > 0 else None
            y_per_shot_val = y_per_shot_time[val_idx] if val_idx.size > 0 else None

            X_train_base = X_train
            X_val_base = X_val
            X_test_base = X_test

            if use_auxiliary_features and auxiliary_targets:
                (
                    X_train,
                    X_val,
                    X_test,
                    _aux_predictor,
                    aux_feature_names,
                ) = enrich_with_auxiliary_features(
                    X_train, X_val, X_test, auxiliary_targets, train_idx
                )
                enriched_names = feature_names + aux_feature_names
            else:
                enriched_names = feature_names

            selector = FeatureSelector(k=n_feat, method=feature_method)
            selector.fit(X_train, y_thr_train, y_time_train, enriched_names)
            X_train_rt = selector.transform_runtime(X_train)
            X_test_rt = selector.transform_runtime(X_test)
            X_val_rt = selector.transform_runtime(X_val) if X_val is not None else None

            X_train_rt = append_runtime_extras(
                X_train_rt, X_train, X_train_base, enriched_names, threshold_predictor
            )
            X_test_rt = append_runtime_extras(
                X_test_rt, X_test, X_test_base, enriched_names, threshold_predictor
            )
            if X_val_rt is not None:
                X_val_rt = append_runtime_extras(
                    X_val_rt, X_val, X_val_base, enriched_names, threshold_predictor
                )

            rt_model = RuntimeModel(
                lgb_params=rt_params,
                use_ensemble=use_ensemble,
                use_calibration=use_calibration,
            )
            rt_eval = (
                (X_val_rt, y_time_val, y_thr_val, y_setup_val, y_per_shot_val)
                if X_val_rt is not None
                else None
            )
            rt_model.fit(
                X_train_rt,
                y_time_train,
                y_thr_train,
                y_setup_train,
                y_per_shot_train,
                eval_set=rt_eval,
            )
            preds = rt_model.predict(X_test_rt, y_thr_test)
            all_pred.extend([float(v) for v in preds])
            all_true.extend([float(v) for v in y_time_test])

        scores = compute_runtime_metrics(all_pred, all_true)
        return scores["mean_runtime_log_score"]

    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=max(5, n_splits))
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_trial.params
    best_rt_params = {k.replace("rt_", ""): v for k, v in best_params.items() if k.startswith("rt_")}

    print(f"\n  Best runtime configuration (log-score={study.best_value:.4f}):")
    print(f"    Feature method: {best_params['feature_method']}")
    print(f"    Number of features: {best_params['n_features']}")
    print(f"    Use auxiliary features: {best_params['use_auxiliary_features']}")
    print(f"    Use ensemble: {best_params['use_ensemble']}")
    print(f"    Use calibration: {best_params['use_calibration']}")

    return {
        "runtime_params": best_rt_params,
        "feature_method": best_params["feature_method"],
        "n_features": best_params["n_features"],
        "use_auxiliary_features": best_params["use_auxiliary_features"],
        "use_ensemble": best_params["use_ensemble"],
        "use_calibration": best_params["use_calibration"],
        "best_runtime_score": study.best_value,
        "n_trials": len(study.trials),
    }


def train_runtime_predictor(
    df: pd.DataFrame,
    X: np.ndarray,
    y_threshold: np.ndarray,
    y_total_time: np.ndarray,
    y_setup_time: np.ndarray,
    y_per_shot_time: np.ndarray,
    feature_cols: list[str],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    best_params: dict,
    threshold_predictor: ThresholdPredictor,
):
    X_train = X[train_idx]
    X_val = X[val_idx] if val_idx.size > 0 else None
    y_thr_train = y_threshold[train_idx]
    y_thr_val = y_threshold[val_idx] if val_idx.size > 0 else None
    y_time_train = y_total_time[train_idx]
    y_time_val = y_total_time[val_idx] if val_idx.size > 0 else None
    y_setup_train = y_setup_time[train_idx]
    y_setup_val = y_setup_time[val_idx] if val_idx.size > 0 else None
    y_per_shot_train = y_per_shot_time[train_idx]
    y_per_shot_val = y_per_shot_time[val_idx] if val_idx.size > 0 else None

    X_enriched_train = X_train
    X_enriched_val = X_val
    enriched_feature_cols = feature_cols.copy()
    aux_predictor = None

    if best_params.get("use_auxiliary_features", True):
        auxiliary_targets = extract_auxiliary_targets(df)
        if auxiliary_targets:
            aux_predictor = AuxiliaryFeaturePredictor(
                n_estimators=150, max_depth=4, learning_rate=0.05
            )
            train_aux = {k: v[train_idx] for k, v in auxiliary_targets.items()}
            aux_predictor.fit(X_train, train_aux)
            X_train_aux = aux_predictor.predict_as_features(X_train)
            X_enriched_train = np.hstack([X_train, X_train_aux])
            if X_val is not None:
                X_val_aux = aux_predictor.predict_as_features(X_val)
                X_enriched_val = np.hstack([X_val, X_val_aux])
            enriched_feature_cols = feature_cols + [
                f"pred_{t}" for t in aux_predictor.models.keys()
            ]

    selector = FeatureSelector(
        k=best_params.get("n_features", 40),
        method=best_params.get("feature_method", "mi"),
    )
    selector.fit(X_enriched_train, y_thr_train, y_time_train, enriched_feature_cols)
    X_rt_train = selector.transform_runtime(X_enriched_train)
    X_rt_val = (
        selector.transform_runtime(X_enriched_val)
        if X_enriched_val is not None
        else None
    )

    X_rt_train = append_runtime_extras(
        X_rt_train, X_enriched_train, X_train, enriched_feature_cols, threshold_predictor
    )
    if X_rt_val is not None:
        X_rt_val = append_runtime_extras(
            X_rt_val, X_enriched_val, X_val, enriched_feature_cols, threshold_predictor
        )

    rt_model = RuntimeModel(
        lgb_params=best_params["runtime_params"],
        use_ensemble=best_params.get("use_ensemble", True),
        use_calibration=best_params.get("use_calibration", True),
    )
    rt_model.fit(
        X_rt_train,
        y_time_train,
        y_thr_train,
        y_setup_train,
        y_per_shot_train,
        eval_set=(
            (X_rt_val, y_time_val, y_thr_val, y_setup_val, y_per_shot_val)
            if X_rt_val is not None
            else None
        ),
    )

    predictor = RuntimePredictorWithThresholdUncertainty(
        runtime_model=rt_model,
        threshold_predictor=threshold_predictor,
    )
    predictor.feature_columns = feature_cols
    predictor.enriched_feature_columns = enriched_feature_cols
    predictor.selected_feature_names = selector.get_selected_names(target="runtime")
    predictor.feature_selector = selector
    predictor.aux_predictor = aux_predictor
    return predictor


def main() -> None:
    start_time = time.time()
    print("=" * 70)
    print("PHASE 2: Runtime Model Training (Threshold Uncertainty)")
    print("=" * 70)

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

    X, y_threshold, y_total_time, y_setup_time, y_per_shot_time, feature_cols = prepare_features(df)

    print("\n[1/3] Tuning hyperparameters...")
    best_params = tune_hyperparameters(
        df,
        X,
        y_threshold,
        y_total_time,
        y_setup_time,
        y_per_shot_time,
        feature_names=feature_cols,
        splits=splits,
        n_trials=OPTUNA_TRIALS,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[2/3] Training per-fold models...")
    for fold, split in enumerate(splits):
        threshold_path = THRESHOLD_DIR / f"fold_{fold}.pkl"
        if not threshold_path.exists():
            raise FileNotFoundError(f"Threshold model not found: {threshold_path}")
        threshold_predictor = ThresholdPredictor.load(threshold_path)

        train_idx = split["train_idx"]
        val_idx = split.get("val_idx", np.array([], dtype=int))
        if val_idx.size == 0:
            train_idx, val_idx = split_train_valid_groups(
                train_idx, df["file"].values, valid_size=VALID_SIZE, seed=42 + fold
            )
        predictor = train_runtime_predictor(
            df,
            X,
            y_threshold,
            y_total_time,
            y_setup_time,
            y_per_shot_time,
            feature_cols,
            train_idx,
            val_idx,
            best_params,
            threshold_predictor,
        )
        model_path = MODEL_DIR / f"fold_{fold}.pkl"
        predictor.save(model_path)
        print(f"  Saved fold {fold} model to: {model_path}")

    print("\n[3/3] Training full model...")
    full_threshold_path = THRESHOLD_DIR / "full_model.pkl"
    if not full_threshold_path.exists():
        raise FileNotFoundError(f"Threshold model not found: {full_threshold_path}")
    full_threshold_predictor = ThresholdPredictor.load(full_threshold_path)

    full_predictor = train_runtime_predictor(
        df,
        X,
        y_threshold,
        y_total_time,
        y_setup_time,
        y_per_shot_time,
        feature_cols,
        train_idx=np.arange(len(df)),
        val_idx=np.array([], dtype=int),
        best_params=best_params,
        threshold_predictor=full_threshold_predictor,
    )
    full_model_path = MODEL_DIR / "full_model.pkl"
    full_predictor.save(full_model_path)
    print(f"  Saved full model to: {full_model_path}")

    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("Runtime model training COMPLETE")
    print(f"  Training time: {total_time:.1f}s")
    print(f"  Models saved to: {MODEL_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
