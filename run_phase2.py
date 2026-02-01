#!/usr/bin/env python3
"""
Phase 2: Model Development and Training

This script:
1. Loads the Phase 1 feature DataFrame
2. Performs leave-one-circuit-out cross-validation
3. Tunes hyperparameters on the competition scoring metric
4. Trains final models on full data
5. Saves trained models to outputs/

Training time estimate: 30-90 seconds on M3 Pro CPU (no GPU needed)
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import LeaveOneGroupOut

# Suppress sklearn warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models import (
    CombinedPredictor,
    get_feature_columns,
)
from src.threshold_model import (
    ThresholdModel,
    apply_family_floor,
    FAMILY_THRESHOLD_FLOORS,
    HIGH_THRESHOLD_FAMILIES,
)
from src.runtime_model import RuntimeModel
from src.scoring import (
    compute_threshold_score,
    compute_runtime_log_score,
    compute_threshold_metrics,
    compute_runtime_metrics,
    threshold_to_idx,
    idx_to_threshold,
    find_optimal_safety_margin,
)
from src.cv_splits import build_cv_splits, load_cv_splits, split_train_valid_groups
from src.feature_selection import FeatureSelector
from src.auxiliary_features import AuxiliaryFeaturePredictor, extract_auxiliary_targets
from src.naive_model import NaiveBucketModel

# Paths
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FEATURES_PATH = OUTPUT_DIR / "training_features.pkl"
CV_SPLITS_PATH = OUTPUT_DIR / "cv_splits.json"
OPTUNA_TRIALS = 120
OPTUNA_SPLITS = 5
VALID_SIZE = 0.2


def load_data():
    """Load the Phase 1 feature DataFrame."""
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Feature file not found: {FEATURES_PATH}\n"
            "Run 'python run_phase1.py' first."
        )
    return pd.read_pickle(FEATURES_PATH)


def prepare_features(df: pd.DataFrame):
    """
    Prepare feature matrix X and target vectors y.

    Returns: X, y_threshold, y_total_time, y_setup_time, y_per_shot_time, feature_cols
    """
    # Get feature columns
    feature_cols = get_feature_columns(df.columns.tolist())

    # Remove columns with all NaN or zero variance
    valid_cols = []
    for col in feature_cols:
        if col in df.columns:
            series = df[col]
            if series.dtype in [np.float64, np.int64, float, int]:
                if series.notna().sum() > 0 and series.std() > 1e-10:
                    valid_cols.append(col)

    feature_cols = valid_cols

    # Build feature matrix
    X = df[feature_cols].copy()

    # Add backend/precision as features (important for runtime)
    X["is_gpu"] = (df["backend"] == "GPU").astype(float)
    X["is_double"] = (df["precision"] == "double").astype(float)
    feature_cols = feature_cols + ["is_gpu", "is_double"]

    # Fill NaN/inf with column median (more robust for small data)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))

    # Target vectors
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
    """Train auxiliary feature predictor and enrich feature matrices."""
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
    """Return True if validation labels are subset of train labels."""
    if y_val is None or y_val.size == 0:
        return False
    train_labels = {threshold_to_idx(v) for v in y_train}
    val_labels = {threshold_to_idx(v) for v in y_val}
    return val_labels.issubset(train_labels)


def append_runtime_extras(
    X_rt: np.ndarray,
    extras: list[np.ndarray],
) -> np.ndarray:
    """Append extra runtime features."""
    if not extras:
        return X_rt
    processed = []
    for extra in extras:
        if extra.ndim == 1:
            processed.append(extra.reshape(-1, 1))
        else:
            processed.append(extra)
    return np.hstack([X_rt] + processed)


def leave_one_circuit_out_cv(
    df: pd.DataFrame,
    X: np.ndarray,
    y_threshold: np.ndarray,
    y_total_time: np.ndarray,
    y_setup_time: np.ndarray,
    y_per_shot_time: np.ndarray,
    feature_names: list,
    threshold_params: dict,
    runtime_params: dict,
    safety_margin: float = 0.0,
    use_family_floor: bool = True,
    n_features: int = 40,
    feature_method: str = "mi",
    decision_policy: str = "expected_score",
    use_auxiliary_features: bool = True,
    valid_size: float = 0.2,
    splitter=None,
    splits: list[dict] | list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> dict:
    """
    Perform leave-one-circuit-out cross-validation.

    This is critical because rows from the same circuit are correlated
    (same circuit, different backend/precision configs).

    Key improvements:
    1. Group-aware splits (by circuit)
    2. Feature selection fitted on training data only (prevents leakage)
    3. Sequential prediction: threshold first, then runtime uses predicted threshold
    4. Optional auxiliary feature enrichment
    """
    groups = df["file"].values
    if splits is None:
        if splitter is None:
            splitter = LeaveOneGroupOut()
        splits = build_cv_splits(
            X, y_threshold, groups, splitter, valid_size=valid_size, seed=42
        )

    all_pred_thresholds = []
    all_true_thresholds = []
    all_pred_times = []
    all_true_times = []
    all_indices = []
    all_families = []

    auxiliary_targets = extract_auxiliary_targets(df) if use_auxiliary_features else None

    for fold, split in enumerate(splits):
        if isinstance(split, dict):
            train_idx = split["train_idx"]
            val_idx = split.get("val_idx", np.array([], dtype=int))
            test_idx = split["test_idx"]
        else:
            train_idx, test_idx = split
            train_idx, val_idx = split_train_valid_groups(
                train_idx, groups, valid_size=valid_size, seed=42 + fold
            )
        X_train, X_test = X[train_idx], X[test_idx]
        y_thr_train, y_thr_test = y_threshold[train_idx], y_threshold[test_idx]
        y_time_train, y_time_test = y_total_time[train_idx], y_total_time[test_idx]
        y_setup_train = y_setup_time[train_idx]
        y_per_shot_train = y_per_shot_time[train_idx]

        X_val = X[val_idx] if val_idx.size > 0 else None
        y_thr_val = y_threshold[val_idx] if val_idx.size > 0 else None
        y_time_val = y_total_time[val_idx] if val_idx.size > 0 else None
        y_setup_val = y_setup_time[val_idx] if val_idx.size > 0 else None
        y_per_shot_val = y_per_shot_time[val_idx] if val_idx.size > 0 else None

        test_families = df.loc[test_idx, "predicted_family"].tolist()

        # Optional: Enrich features with predicted auxiliary features
        if use_auxiliary_features and auxiliary_targets is not None:
            (
                X_train_enriched,
                X_val_enriched,
                X_test_enriched,
                _aux_predictor,
                aux_feature_names,
            ) = enrich_with_auxiliary_features(
                X_train, X_val, X_test, auxiliary_targets, train_idx
            )
            enriched_names = feature_names + aux_feature_names
        else:
            X_train_enriched, X_val_enriched, X_test_enriched = X_train, X_val, X_test
            enriched_names = feature_names

        # Step 1: Feature selection (fitted on training data only)
        selector = FeatureSelector(k=n_features, method=feature_method)
        selector.fit(X_train_enriched, y_thr_train, y_time_train, enriched_names)
        X_train_thr = selector.transform_threshold(X_train_enriched)
        X_test_thr = selector.transform_threshold(X_test_enriched)
        X_train_rt = selector.transform_runtime(X_train_enriched)
        X_test_rt = selector.transform_runtime(X_test_enriched)

        X_val_thr = selector.transform_threshold(X_val_enriched) if X_val_enriched is not None else None
        X_val_rt = selector.transform_runtime(X_val_enriched) if X_val_enriched is not None else None

        # Step 2: Train threshold model on selected features
        thr_model = ThresholdModel(
            lgb_params=threshold_params,
            safety_margin=safety_margin,
            decision_policy=decision_policy,
        )
        thr_eval_set = None
        if X_val_thr is not None and can_use_threshold_eval_set(y_thr_train, y_thr_val):
            thr_eval_set = (X_val_thr, y_thr_val)
        thr_model.fit(X_train_thr, y_thr_train, eval_set=thr_eval_set)

        # Step 3: Predict thresholds
        pred_thresholds = thr_model.predict(X_test_thr)

        # Runtime extras: threshold uncertainty + backend/precision flags
        train_proba = thr_model.predict_proba(X_train_thr)
        test_proba = thr_model.predict_proba(X_test_thr)
        val_proba = thr_model.predict_proba(X_val_thr) if X_val_thr is not None else None

        idxs = np.arange(train_proba.shape[1])
        train_expected = train_proba @ idxs
        test_expected = test_proba @ idxs
        val_expected = val_proba @ idxs if val_proba is not None else None

        train_entropy = -(train_proba * np.log(train_proba + 1e-9)).sum(axis=1)
        test_entropy = -(test_proba * np.log(test_proba + 1e-9)).sum(axis=1)
        val_entropy = (
            -(val_proba * np.log(val_proba + 1e-9)).sum(axis=1)
            if val_proba is not None
            else None
        )

        extra_train = [train_expected, train_entropy]
        extra_test = [test_expected, test_entropy]
        extra_val = [val_expected, val_entropy] if val_expected is not None else []

        for flag_name in ["is_gpu", "is_double"]:
            if flag_name in enriched_names:
                flag_idx = enriched_names.index(flag_name)
                extra_train.append(X_train_enriched[:, flag_idx])
                extra_test.append(X_test_enriched[:, flag_idx])
                if X_val_enriched is not None:
                    extra_val.append(X_val_enriched[:, flag_idx])

        # Apply family floor (Innovation #3)
        if use_family_floor:
            pred_thresholds = apply_family_floor(pred_thresholds, test_families)

        # Step 4: Train runtime model with TRUE thresholds from training data
        rt_model = RuntimeModel(lgb_params=runtime_params)
        X_train_rt = append_runtime_extras(X_train_rt, extra_train)
        X_test_rt = append_runtime_extras(X_test_rt, extra_test)
        if X_val_rt is not None:
            X_val_rt = append_runtime_extras(X_val_rt, extra_val)
        rt_eval_set = (
            (X_val_rt, y_time_val, y_thr_val, y_setup_val, y_per_shot_val)
            if X_val_rt is not None
            else None
        )
        rt_model.fit(
            X_train_rt, y_time_train, y_thr_train,
            y_setup_train, y_per_shot_train,
            eval_set=rt_eval_set,
        )

        # Step 5: Predict runtime using PREDICTED thresholds (sequential prediction)
        pred_times = rt_model.predict(X_test_rt, pred_thresholds)

        # Collect results
        for j, (pt, tt, ptr, ttr, fam) in enumerate(zip(
            pred_thresholds, y_thr_test, pred_times, y_time_test, test_families
        )):
            all_pred_thresholds.append(int(pt))
            all_true_thresholds.append(int(tt))
            all_pred_times.append(float(ptr))
            all_true_times.append(float(ttr))
            all_indices.append(test_idx[j])
            all_families.append(fam)

    scores = {}
    scores.update(compute_threshold_metrics(all_pred_thresholds, all_true_thresholds))
    scores.update(compute_runtime_metrics(all_pred_times, all_true_times))

    return {
        "scores": scores,
        "pred_thresholds": all_pred_thresholds,
        "true_thresholds": all_true_thresholds,
        "pred_times": all_pred_times,
        "true_times": all_true_times,
        "indices": all_indices,
        "families": all_families,
    }


def evaluate_naive_baseline(
    df: pd.DataFrame,
    splits: list[dict] | list[tuple[np.ndarray, np.ndarray]],
) -> dict:
    """Evaluate naive bucketed baseline using the provided group splits."""
    all_pred_thresholds: list[int] = []
    all_true_thresholds: list[int] = []
    all_pred_times: list[float] = []
    all_true_times: list[float] = []

    for split in splits:
        if isinstance(split, dict):
            train_df = df.iloc[split["train_idx"]]
            test_df = df.iloc[split["test_idx"]]
        else:
            train_df = df.iloc[split[0]]
            test_df = df.iloc[split[1]]

        model = NaiveBucketModel()
        model.fit(train_df, threshold_col="selected_threshold", runtime_col="forward_wall_s")
        pred_thr, pred_time = model.predict(test_df)

        all_pred_thresholds.extend([int(v) for v in pred_thr])
        all_true_thresholds.extend([int(v) for v in test_df["selected_threshold"].values])
        all_pred_times.extend([float(v) for v in pred_time])
        all_true_times.extend([float(v) for v in test_df["forward_wall_s"].values])

    scores = {}
    scores.update(compute_threshold_metrics(all_pred_thresholds, all_true_thresholds))
    scores.update(compute_runtime_metrics(all_pred_times, all_true_times))

    return {
        "scores": scores,
        "pred_thresholds": all_pred_thresholds,
        "true_thresholds": all_true_thresholds,
        "pred_times": all_pred_times,
        "true_times": all_true_times,
    }


def suggest_lgbm_params(
    trial: optuna.Trial,
    prefix: str,
    objective: str,
    num_class: int | None = None,
) -> dict:
    """Suggest LightGBM parameters with strong regularization."""
    max_depth = trial.suggest_int(f"{prefix}max_depth", 3, 6)
    max_leaves = min(2 ** max_depth, 63)
    num_leaves = trial.suggest_int(f"{prefix}num_leaves", max(8, 2 ** (max_depth - 1)), max_leaves)
    params = {
        "objective": objective,
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
    if num_class is not None:
        params["num_class"] = num_class
    return params


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
    valid_size: float = 0.2,
) -> dict:
    """
    Bayesian hyperparameter search (Optuna) using threshold score.

    Returns best parameters for threshold and runtime models.
    """
    n_splits = len(splits)

    print("\n  Hyperparameter search (Optuna TPE, group-aware CV):")
    print(f"  Trials: {n_trials}, CV splits: {n_splits}, features: {len(feature_names)}")

    def objective(trial: optuna.Trial) -> float:
        thr_params = suggest_lgbm_params(trial, "thr_", "multiclass", num_class=9)
        rt_params = suggest_lgbm_params(trial, "rt_", "regression")

        n_feat = trial.suggest_int("n_features", 20, min(80, len(feature_names)))
        use_family_floor = trial.suggest_categorical("use_family_floor", [True, False])
        feature_method = trial.suggest_categorical("feature_method", ["mi", "corr"])
        decision_policy = trial.suggest_categorical("decision_policy", ["expected_score", "argmax"])
        safety_margin = trial.suggest_float("safety_margin", 0.0, 2.0, step=0.5)

        result = leave_one_circuit_out_cv(
            df, X, y_threshold, y_total_time, y_setup_time, y_per_shot_time,
            feature_names, thr_params, rt_params,
            safety_margin=safety_margin,
            use_family_floor=use_family_floor,
            n_features=n_feat,
            feature_method=feature_method,
            decision_policy=decision_policy,
            use_auxiliary_features=True,
            valid_size=valid_size,
            splits=splits,
        )
        return result["scores"]["mean_threshold_score"]

    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=max(5, n_splits))
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_trial.params
    best_thr_params = {k.replace("thr_", ""): v for k, v in best_params.items() if k.startswith("thr_")}
    best_rt_params = {k.replace("rt_", ""): v for k, v in best_params.items() if k.startswith("rt_")}

    print(f"\n  Best threshold configuration (score={study.best_value:.4f}):")
    print(f"    Safety margin: {best_params['safety_margin']}")
    print(f"    Use family floors: {best_params['use_family_floor']}")
    print(f"    Feature method: {best_params['feature_method']}")
    print(f"    Decision policy: {best_params['decision_policy']}")
    print(f"    Number of features: {best_params['n_features']}")

    return {
        "threshold_params": best_thr_params,
        "runtime_params": best_rt_params,
        "safety_margin": best_params["safety_margin"],
        "use_family_floor": best_params["use_family_floor"],
        "feature_method": best_params["feature_method"],
        "decision_policy": best_params["decision_policy"],
        "n_features": best_params["n_features"],
        "use_auxiliary_features": True,
        "best_threshold_score": study.best_value,
        "n_trials": len(study.trials),
    }


def analyze_cv_results(cv_result: dict, df: pd.DataFrame):
    """Print detailed analysis of CV results."""
    scores = cv_result["scores"]

    print(f"\n  Mean threshold score: {scores['mean_threshold_score']:.4f}")
    print(f"  Threshold failures (score=0): {scores['n_threshold_failures']}/{scores['n_tasks']}")
    print(f"  Mean runtime log score: {scores['mean_runtime_log_score']:.4f}")
    if scores.get("n_runtime_invalid", 0) > 0:
        print(f"  Runtime invalid: {scores['n_runtime_invalid']}/{scores['n_tasks']}")

    # Per-circuit analysis
    pred_thr = np.array(cv_result["pred_thresholds"])
    true_thr = np.array(cv_result["true_thresholds"])
    pred_time = np.array(cv_result["pred_times"])
    true_time = np.array(cv_result["true_times"])
    indices = cv_result["indices"]

    thr_scores = np.array([compute_threshold_score(pt, tt) for pt, tt in zip(pred_thr, true_thr)])
    rt_scores = np.array([compute_runtime_log_score(ptr, ttr) for ptr, ttr in zip(pred_time, true_time)])

    worst_thr_idx = np.argsort(thr_scores)[:5]
    print("\n  Worst 5 threshold predictions:")
    for idx in worst_thr_idx:
        orig_idx = indices[idx]
        row = df.iloc[orig_idx]
        thr_score = compute_threshold_score(pred_thr[idx], true_thr[idx])
        print(f"    {row['file'][:30]:<30} {row['backend']:>3}/{row['precision']:<6} "
              f"thr: {int(true_thr[idx]):>3} -> {int(pred_thr[idx]):>3} (score={thr_score:.2f})")

    finite_rt = np.isfinite(rt_scores)
    if finite_rt.any():
        worst_rt_idx = np.argsort(rt_scores[finite_rt])[:5]
        print("\n  Worst 5 runtime predictions (log score):")
        finite_indices = np.flatnonzero(finite_rt)
        for local_idx in worst_rt_idx:
            idx = finite_indices[local_idx]
            orig_idx = indices[idx]
            row = df.iloc[orig_idx]
            rt_score = compute_runtime_log_score(pred_time[idx], true_time[idx])
            print(f"    {row['file'][:30]:<30} {row['backend']:>3}/{row['precision']:<6} "
                  f"time: {true_time[idx]:>7.1f} -> {pred_time[idx]:>7.1f} (score={rt_score:.3f})")

    # Threshold confusion matrix
    print("\n  Threshold prediction summary:")
    unique_thresholds = sorted(set(true_thr.astype(int)))
    for thr in unique_thresholds:
        mask = true_thr == thr
        preds = pred_thr[mask]
        under = (preds < thr).sum()
        exact = (preds == thr).sum()
        over = (preds > thr).sum()
        print(f"    True={thr:>3}: under={under:>2}, exact={exact:>2}, over={over:>2}")


def main():
    start_time = time.time()

    print("=" * 70)
    print("PHASE 2: Model Training")
    print("=" * 70)
    print("\nNote: Training uses CPU only. GPU/MPS is unnecessary for this dataset size.")
    print("Expected total time: 30-90 seconds\n")

    # ── Step 1: Load data ──────────────────────────────────────────────
    print("[1/5] Loading feature data...")
    df = load_data()
    print(f"  Loaded {len(df)} rows, {df['file'].nunique()} circuits")

    # Filter out rows with missing targets
    valid = df["selected_threshold"].notna() & df["forward_wall_s"].notna()
    df = df[valid].reset_index(drop=True)
    print(f"  After filtering: {len(df)} rows")

    # ── Step 2: Prepare features ───────────────────────────────────────
    print("\n[2/5] Preparing features...")
    X, y_threshold, y_total_time, y_setup_time, y_per_shot_time, feature_cols = prepare_features(df)
    print(f"  Feature matrix: {X.shape}")
    print(f"  Target ranges:")
    print(f"    Threshold: {int(y_threshold.min())} - {int(y_threshold.max())}")
    print(f"    Runtime: {y_total_time.min():.1f}s - {y_total_time.max():.1f}s")

    # ── Step 3: Hyperparameter tuning ──────────────────────────────────
    print("\n[3/5] Tuning hyperparameters with leave-one-circuit-out CV...")
    tune_start = time.time()

    if not CV_SPLITS_PATH.exists():
        raise FileNotFoundError(
            f"CV splits not found: {CV_SPLITS_PATH}\n"
            "Run 'python run_generate_cv_splits.py' first."
        )

    cv_splits, cv_meta = load_cv_splits(CV_SPLITS_PATH)
    if cv_meta.get("n_samples") is not None and cv_meta["n_samples"] != len(df):
        raise ValueError(
            "CV splits do not match current dataset size. "
            f"Splits have n_samples={cv_meta['n_samples']} but data has {len(df)} rows."
        )

    best_params = tune_hyperparameters(
        df, X, y_threshold, y_total_time, y_setup_time, y_per_shot_time,
        feature_names=feature_cols,
        splits=cv_splits,
        n_trials=OPTUNA_TRIALS,
        valid_size=VALID_SIZE,
    )

    tune_time = time.time() - tune_start
    print(f"\n  Tuning completed in {tune_time:.1f}s")

    # ── Step 4: Final CV evaluation ────────────────────────────────────
    print("\n[4/5] Final cross-validation evaluation...")

    final_splits = cv_splits

    final_cv = leave_one_circuit_out_cv(
        df, X, y_threshold, y_total_time, y_setup_time, y_per_shot_time,
        feature_cols,
        best_params["threshold_params"],
        best_params["runtime_params"],
        safety_margin=best_params.get("safety_margin", 0.0),
        use_family_floor=best_params.get("use_family_floor", True),
        n_features=best_params.get("n_features", 40),
        feature_method=best_params.get("feature_method", "mi"),
        decision_policy=best_params.get("decision_policy", "expected_score"),
        use_auxiliary_features=best_params.get("use_auxiliary_features", True),
        valid_size=VALID_SIZE,
        splits=final_splits,
    )

    analyze_cv_results(final_cv, df)

    print("\n  Naive bucketed baseline (same splits)...")
    naive_cv = evaluate_naive_baseline(df, final_splits)
    print(f"  Naive mean threshold score: {naive_cv['scores']['mean_threshold_score']:.4f}")
    print(f"  Naive threshold failures: {naive_cv['scores']['n_threshold_failures']}/{naive_cv['scores']['n_tasks']}")
    print(f"  Naive mean runtime log score: {naive_cv['scores']['mean_runtime_log_score']:.4f}")

    # ── Step 5: Train final models ─────────────────────────────────────
    print("\n[5/5] Training final models on full data...")

    # Step 5a: Optional auxiliary feature enrichment
    n_features = best_params.get("n_features", 40)
    feature_method = best_params.get("feature_method", "mi")
    decision_policy = best_params.get("decision_policy", "expected_score")
    X_enriched = X.copy()
    enriched_feature_cols = feature_cols.copy()
    final_aux_predictor = None

    if best_params.get("use_auxiliary_features", True):
        print("  Training auxiliary feature predictor...")
        auxiliary_targets = extract_auxiliary_targets(df)
        final_aux_predictor = AuxiliaryFeaturePredictor(n_estimators=150, max_depth=4, learning_rate=0.05)
        final_aux_predictor.fit(X, auxiliary_targets)
        X_aux = final_aux_predictor.predict_as_features(X)
        X_enriched = np.hstack([X, X_aux])
        enriched_feature_cols = feature_cols + [f"pred_{t}" for t in final_aux_predictor.models.keys()]
        print(f"  Added {X_aux.shape[1]} predicted auxiliary features")

    groups = df["file"].values
    train_idx_final, val_idx_final = split_train_valid_groups(
        np.arange(len(df)), groups, valid_size=VALID_SIZE, seed=123
    )

    X_train_enriched = X_enriched[train_idx_final]
    X_val_enriched = X_enriched[val_idx_final] if val_idx_final.size > 0 else None
    y_thr_train = y_threshold[train_idx_final]
    y_thr_val = y_threshold[val_idx_final] if val_idx_final.size > 0 else None
    y_time_train = y_total_time[train_idx_final]
    y_time_val = y_total_time[val_idx_final] if val_idx_final.size > 0 else None
    y_setup_train = y_setup_time[train_idx_final]
    y_setup_val = y_setup_time[val_idx_final] if val_idx_final.size > 0 else None
    y_per_shot_train = y_per_shot_time[train_idx_final]
    y_per_shot_val = y_per_shot_time[val_idx_final] if val_idx_final.size > 0 else None

    # Step 5b: Feature selection on (possibly enriched) data
    final_selector = FeatureSelector(k=n_features, method=feature_method)
    final_selector.fit(X_enriched, y_threshold, y_total_time, enriched_feature_cols)
    X_selected_thr = final_selector.transform_threshold(X_enriched)
    X_selected_rt = final_selector.transform_runtime(X_enriched)
    selected_feature_names_thr = final_selector.get_selected_names(target="threshold")
    selected_feature_names_rt = final_selector.get_selected_names(target="runtime")
    print(f"  Selected {n_features} threshold features from {len(enriched_feature_cols)}")
    print(f"  Selected {n_features} runtime features from {len(enriched_feature_cols)}")

    X_thr_train = X_selected_thr[train_idx_final]
    X_rt_train = X_selected_rt[train_idx_final]
    X_thr_val = X_selected_thr[val_idx_final] if val_idx_final.size > 0 else None
    X_rt_val = X_selected_rt[val_idx_final] if val_idx_final.size > 0 else None

    # Step 5c: Threshold model on selected features
    final_thr_model = ThresholdModel(
        lgb_params=best_params["threshold_params"],
        safety_margin=best_params.get("safety_margin", 0.0),
        decision_policy=decision_policy,
    )
    final_thr_eval = None
    if X_thr_val is not None and can_use_threshold_eval_set(y_thr_train, y_thr_val):
        final_thr_eval = (X_thr_val, y_thr_val)
    final_thr_model.fit(X_thr_train, y_thr_train, eval_set=final_thr_eval)

    # Step 5d: Runtime model on selected features + thresholds
    final_rt_model = RuntimeModel(lgb_params=best_params["runtime_params"])
    train_proba = final_thr_model.predict_proba(X_thr_train)
    idxs = np.arange(train_proba.shape[1])
    train_expected = train_proba @ idxs
    train_entropy = -(train_proba * np.log(train_proba + 1e-9)).sum(axis=1)
    extra_train = [train_expected, train_entropy]

    extra_val = []
    if X_thr_val is not None:
        val_proba = final_thr_model.predict_proba(X_thr_val)
        val_expected = val_proba @ idxs
        val_entropy = -(val_proba * np.log(val_proba + 1e-9)).sum(axis=1)
        extra_val = [val_expected, val_entropy]

    for flag_name in ["is_gpu", "is_double"]:
        if flag_name in enriched_feature_cols:
            flag_idx = enriched_feature_cols.index(flag_name)
            extra_train.append(X_enriched[train_idx_final][:, flag_idx])
            if X_val_enriched is not None:
                extra_val.append(X_val_enriched[:, flag_idx])

    X_rt_train = append_runtime_extras(X_rt_train, extra_train)
    if X_rt_val is not None:
        X_rt_val = append_runtime_extras(X_rt_val, extra_val)

    final_rt_model.fit(
        X_rt_train, y_time_train, y_thr_train,  # Use true thresholds for training
        y_setup_train, y_per_shot_train,
        eval_set=(
            (X_rt_val, y_time_val, y_thr_val, y_setup_val, y_per_shot_val)
            if X_rt_val is not None
            else None
        ),
    )

    # Combined predictor with selector
    combined = CombinedPredictor(final_thr_model, final_rt_model)
    combined.feature_columns = feature_cols  # Original feature columns
    combined.enriched_feature_columns = enriched_feature_cols  # Enriched columns (may include aux)
    combined.selected_feature_names_threshold = selected_feature_names_thr
    combined.selected_feature_names_runtime = selected_feature_names_rt
    combined.feature_selector = final_selector  # Store selector for prediction

    # Store auxiliary predictor if enabled
    if final_aux_predictor is not None:
        combined.aux_predictor = final_aux_predictor
        print("  Auxiliary predictor stored in model")


    # Save models
    OUTPUT_DIR.mkdir(exist_ok=True)
    combined.save(OUTPUT_DIR / "combined_model.pkl")
    print(f"  Saved model to: {OUTPUT_DIR / 'combined_model.pkl'}")

    # Save best parameters
    params_path = OUTPUT_DIR / "best_params.txt"
    with open(params_path, "w") as f:
        f.write("Best Hyperparameters\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Threshold Model:\n")
        for k, v in best_params["threshold_params"].items():
            f.write(f"  {k}: {v}\n")
        f.write(f"  safety_margin: {best_params['safety_margin']}\n")
        f.write(f"  use_family_floor: {best_params.get('use_family_floor', True)}\n\n")
        f.write(f"  decision_policy: {best_params.get('decision_policy', 'expected_score')}\n\n")
        f.write(f"Runtime Model:\n")
        for k, v in best_params["runtime_params"].items():
            f.write(f"  {k}: {v}\n")
        f.write(f"\nFeature Selection:\n")
        f.write(f"  n_features: {n_features}\n")
        f.write(f"  method: {best_params.get('feature_method', 'mi')}\n")
        f.write(f"  from original: {len(feature_cols)}\n")
        f.write(f"  enriched features: {len(enriched_feature_cols)}\n")
        f.write(f"\nEnhancements:\n")
        f.write(f"  use_auxiliary_features: {best_params.get('use_auxiliary_features', True)}\n")
        f.write(f"\nCV Threshold Score: {best_params['best_threshold_score']:.4f}\n")
    print(f"  Saved parameters to: {params_path}")

    # Save feature columns for prediction script (original + selected)
    feature_path = OUTPUT_DIR / "feature_columns.txt"
    with open(feature_path, "w") as f:
        for col in feature_cols:
            f.write(col + "\n")
    print(f"  Saved feature columns to: {feature_path}")

    # Save selected features
    selected_thr_path = OUTPUT_DIR / "selected_features_threshold.txt"
    with open(selected_thr_path, "w") as f:
        f.write(f"# Top {n_features} threshold features selected by {feature_method}\n")
        for col in selected_feature_names_thr:
            f.write(col + "\n")
    print(f"  Saved threshold features to: {selected_thr_path}")

    selected_rt_path = OUTPUT_DIR / "selected_features_runtime.txt"
    with open(selected_rt_path, "w") as f:
        f.write(f"# Top {n_features} runtime features selected by {feature_method}\n")
        for col in selected_feature_names_rt:
            f.write(col + "\n")
    print(f"  Saved runtime features to: {selected_rt_path}")

    # ── Summary ────────────────────────────────────────────────────────
    total_time = time.time() - start_time

    print("\n" + "=" * 70)
    print("Phase 2 COMPLETE")
    print(f"  Training time: {total_time:.1f}s")
    print(f"  CV mean threshold score: {final_cv['scores']['mean_threshold_score']:.4f}")
    print(f"  CV threshold failures: {final_cv['scores']['n_threshold_failures']}/{final_cv['scores']['n_tasks']}")
    print(f"  CV mean runtime log score: {final_cv['scores']['mean_runtime_log_score']:.4f}")
    print(f"  Model saved to: {OUTPUT_DIR / 'combined_model.pkl'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
