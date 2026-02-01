#!/usr/bin/env python3
"""
LGBM Threshold Model Training

This script:
1. Loads the prepared feature DataFrame
2. Tunes hyperparameters on fixed CV splits
3. Trains per-fold models and a full model
4. Saves trained models to outputs/threshold_models/threshold_lgbm
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.threshold_models.features import get_feature_columns
from src.threshold_models.lgbm.predictor import ThresholdPredictor
from src.threshold_models.lgbm.threshold_model import ThresholdModel, apply_family_floor
from src.evaluation.scoring import compute_threshold_metrics, threshold_to_idx
from src.evaluation.cv_splits import load_cv_splits, split_train_valid_groups
from src.data.feature_selection import FeatureSelector
from src.data.auxiliary_features import AuxiliaryFeaturePredictor, extract_auxiliary_targets

OUTPUT_DIR = PROJECT_ROOT / "outputs"
FEATURES_PATH = OUTPUT_DIR / "training_features.pkl"
CV_SPLITS_PATH = OUTPUT_DIR / "cv_splits.json"
MODEL_DIR = OUTPUT_DIR / "threshold_models" / "threshold_lgbm"
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

    return X.values, y_threshold, y_total_time, feature_cols


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


def can_use_threshold_eval_set(
    y_train: np.ndarray,
    y_val: np.ndarray | None,
) -> bool:
    if y_val is None or y_val.size == 0:
        return False
    train_labels = {threshold_to_idx(v) for v in y_train}
    val_labels = {threshold_to_idx(v) for v in y_val}
    return val_labels.issubset(train_labels)


def suggest_lgbm_params(
    trial: optuna.Trial,
    prefix: str,
) -> dict:
    max_depth = trial.suggest_int(f"{prefix}max_depth", 3, 6)
    max_leaves = min(2 ** max_depth, 63)
    num_leaves = trial.suggest_int(f"{prefix}num_leaves", max(8, 2 ** (max_depth - 1)), max_leaves)
    return {
        "objective": "multiclass",
        "num_class": 9,
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
    feature_names: list,
    splits: list[dict],
    n_trials: int = 120,
    valid_size: float = 0.2,
) -> dict:
    n_splits = len(splits)
    print("\n  Hyperparameter search (Optuna TPE, group-aware CV):")
    print(f"  Trials: {n_trials}, CV splits: {n_splits}, features: {len(feature_names)}")

    auxiliary_targets = extract_auxiliary_targets(df)

    def objective(trial: optuna.Trial) -> float:
        thr_params = suggest_lgbm_params(trial, "thr_")
        n_feat = trial.suggest_int("n_features", 20, min(80, len(feature_names)))
        use_family_floor = trial.suggest_categorical("use_family_floor", [True, False])
        feature_method = trial.suggest_categorical("feature_method", ["mi", "corr"])
        decision_policy = trial.suggest_categorical("decision_policy", ["expected_score", "argmax"])
        safety_margin = trial.suggest_float("safety_margin", 0.0, 2.0, step=0.5)
        use_auxiliary_features = trial.suggest_categorical("use_auxiliary_features", [True, False])

        all_pred = []
        all_true = []

        for fold, split in enumerate(splits):
            train_idx = split["train_idx"]
            test_idx = split["test_idx"]
            val_idx = split.get("val_idx", np.array([], dtype=int))

            X_train, X_test = X[train_idx], X[test_idx]
            y_train = y_threshold[train_idx]
            y_test = y_threshold[test_idx]
            y_time_train = y_total_time[train_idx]

            X_val = X[val_idx] if val_idx.size > 0 else None
            y_val = y_threshold[val_idx] if val_idx.size > 0 else None
            y_time_val = y_total_time[val_idx] if val_idx.size > 0 else None

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
            selector.fit(X_train, y_train, y_time_train, enriched_names)
            X_train_thr = selector.transform_threshold(X_train)
            X_test_thr = selector.transform_threshold(X_test)
            X_val_thr = selector.transform_threshold(X_val) if X_val is not None else None

            thr_model = ThresholdModel(
                lgb_params=thr_params,
                safety_margin=safety_margin,
                decision_policy=decision_policy,
            )
            thr_eval = None
            if X_val_thr is not None and can_use_threshold_eval_set(y_train, y_val):
                thr_eval = (X_val_thr, y_val)
            thr_model.fit(X_train_thr, y_train, eval_set=thr_eval)

            preds = thr_model.predict(X_test_thr)
            if use_family_floor:
                families = df.loc[test_idx, "predicted_family"].tolist()
                preds = apply_family_floor(preds, families)

            all_pred.extend([int(v) for v in preds])
            all_true.extend([int(v) for v in y_test])

        scores = compute_threshold_metrics(all_pred, all_true)
        return scores["mean_threshold_score"]

    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=max(5, n_splits))
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials)

    try:
        fig = optuna.visualization.matplotlib.plot_optimization_history(study)
        fig.tight_layout()
        plot_path = MODEL_DIR / "optuna_history.png"
        fig.savefig(plot_path, dpi=160)
        plt.close(fig)
        print(f"  Saved Optuna optimization history: {plot_path}")
    except Exception as exc:
        print(f"  WARNING: failed to save Optuna history plot ({exc}).")

    best_params = study.best_trial.params
    best_thr_params = {k.replace("thr_", ""): v for k, v in best_params.items() if k.startswith("thr_")}

    print(f"\n  Best threshold configuration (score={study.best_value:.4f}):")
    print(f"    Safety margin: {best_params['safety_margin']}")
    print(f"    Use family floors: {best_params['use_family_floor']}")
    print(f"    Feature method: {best_params['feature_method']}")
    print(f"    Decision policy: {best_params['decision_policy']}")
    print(f"    Number of features: {best_params['n_features']}")

    return {
        "threshold_params": best_thr_params,
        "safety_margin": best_params["safety_margin"],
        "use_family_floor": best_params["use_family_floor"],
        "feature_method": best_params["feature_method"],
        "decision_policy": best_params["decision_policy"],
        "n_features": best_params["n_features"],
        "use_auxiliary_features": best_params["use_auxiliary_features"],
        "best_threshold_score": study.best_value,
        "n_trials": len(study.trials),
    }


def train_threshold_predictor(
    df: pd.DataFrame,
    X: np.ndarray,
    y_threshold: np.ndarray,
    y_total_time: np.ndarray,
    feature_cols: list[str],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    best_params: dict,
):
    X_train = X[train_idx]
    X_val = X[val_idx] if val_idx.size > 0 else None
    y_thr_train = y_threshold[train_idx]
    y_thr_val = y_threshold[val_idx] if val_idx.size > 0 else None
    y_time_train = y_total_time[train_idx]
    y_time_val = y_total_time[val_idx] if val_idx.size > 0 else None

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
    X_thr_train = selector.transform_threshold(X_enriched_train)
    X_thr_val = (
        selector.transform_threshold(X_enriched_val)
        if X_enriched_val is not None
        else None
    )

    thr_model = ThresholdModel(
        lgb_params=best_params["threshold_params"],
        safety_margin=best_params.get("safety_margin", 0.0),
        decision_policy=best_params.get("decision_policy", "expected_score"),
    )
    thr_eval = None
    if X_thr_val is not None and can_use_threshold_eval_set(y_thr_train, y_thr_val):
        thr_eval = (X_thr_val, y_thr_val)
    thr_model.fit(X_thr_train, y_thr_train, eval_set=thr_eval)

    predictor = ThresholdPredictor(threshold_model=thr_model)
    predictor.feature_columns = feature_cols
    predictor.enriched_feature_columns = enriched_feature_cols
    predictor.selected_feature_names = selector.get_selected_names(target="threshold")
    predictor.feature_selector = selector
    predictor.aux_predictor = aux_predictor
    predictor.use_family_floor = best_params.get("use_family_floor", True)
    return predictor


def main() -> None:
    start_time = time.time()
    print("=" * 70)
    print("PHASE 2: Threshold Model Training")
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

    X, y_threshold, y_total_time, feature_cols = prepare_features(df)

    print("\n[1/3] Tuning hyperparameters...")
    best_params = tune_hyperparameters(
        df,
        X,
        y_threshold,
        y_total_time,
        feature_names=feature_cols,
        splits=splits,
        n_trials=OPTUNA_TRIALS,
        valid_size=VALID_SIZE,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[2/3] Training per-fold models...")
    for fold, split in enumerate(splits):
        train_idx = split["train_idx"]
        val_idx = split.get("val_idx", np.array([], dtype=int))
        if val_idx.size == 0:
            train_idx, val_idx = split_train_valid_groups(
                train_idx, df["file"].values, valid_size=VALID_SIZE, seed=42 + fold
            )
        predictor = train_threshold_predictor(
            df, X, y_threshold, y_total_time, feature_cols, train_idx, val_idx, best_params
        )
        model_path = MODEL_DIR / f"fold_{fold}.pkl"
        predictor.save(model_path)
        print(f"  Saved fold {fold} model to: {model_path}")

    print("\n[3/3] Training full model...")
    full_predictor = train_threshold_predictor(
        df,
        X,
        y_threshold,
        y_total_time,
        feature_cols,
        train_idx=np.arange(len(df)),
        val_idx=np.array([], dtype=int),
        best_params=best_params,
    )
    full_model_path = MODEL_DIR / "full_model.pkl"
    full_predictor.save(full_model_path)
    print(f"  Saved full model to: {full_model_path}")

    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("Threshold model training COMPLETE")
    print(f"  Training time: {total_time:.1f}s")
    print(f"  Models saved to: {MODEL_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
