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
from itertools import product

import numpy as np
import pandas as pd

# Suppress sklearn warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models import (
    ThresholdModel,
    RuntimeModel,
    CombinedPredictor,
    get_feature_columns,
    apply_family_floor,
    FAMILY_THRESHOLD_FLOORS,
)
from src.scoring import (
    compute_overall_score,
    compute_threshold_score,
    compute_runtime_score,
    threshold_to_idx,
    idx_to_threshold,
    find_optimal_safety_margin,
)

# Paths
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FEATURES_PATH = OUTPUT_DIR / "training_features.pkl"


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

    # Fill NaN with column mean
    X = X.fillna(X.mean())

    # Target vectors
    y_threshold = df["selected_threshold"].values.astype(float)
    y_total_time = df["forward_wall_s"].values.astype(float)
    y_setup_time = df["estimated_setup_s"].values.astype(float)
    y_per_shot_time = df["estimated_per_shot_s"].values.astype(float)

    return X.values, y_threshold, y_total_time, y_setup_time, y_per_shot_time, feature_cols


def leave_one_circuit_out_cv(
    df: pd.DataFrame,
    X: np.ndarray,
    y_threshold: np.ndarray,
    y_total_time: np.ndarray,
    y_setup_time: np.ndarray,
    y_per_shot_time: np.ndarray,
    threshold_params: dict,
    runtime_params: dict,
    safety_margin: float = 0.3,
    use_family_floor: bool = True,
) -> dict:
    """
    Perform leave-one-circuit-out cross-validation.

    This is critical because rows from the same circuit are correlated
    (same circuit, different backend/precision configs).
    """
    circuits = df["file"].unique()
    n_circuits = len(circuits)

    all_pred_thresholds = []
    all_true_thresholds = []
    all_pred_times = []
    all_true_times = []
    all_indices = []
    all_families = []

    for i, circuit in enumerate(circuits):
        # Split: leave out all rows for this circuit
        test_mask = df["file"] == circuit
        train_mask = ~test_mask

        X_train, X_test = X[train_mask], X[test_mask]
        y_thr_train, y_thr_test = y_threshold[train_mask], y_threshold[test_mask]
        y_time_train, y_time_test = y_total_time[train_mask], y_total_time[test_mask]
        y_setup_train = y_setup_time[train_mask]
        y_per_shot_train = y_per_shot_time[train_mask]

        # Get predicted family for test circuit (use true_family for CV, predicted_family for holdout)
        test_families = df.loc[test_mask, "predicted_family"].tolist()

        # Train models
        thr_model = ThresholdModel(**threshold_params, safety_margin=safety_margin)
        thr_model.fit(X_train, y_thr_train, calibrate=False)  # Use fixed margin
        thr_model.safety_margin = safety_margin

        rt_model = RuntimeModel(**runtime_params)
        rt_model.fit(X_train, y_time_train, y_setup_train, y_per_shot_train)

        # Predict
        pred_thresholds = thr_model.predict(X_test)
        pred_times = rt_model.predict(X_test)

        # Apply family floor (Innovation #3)
        if use_family_floor:
            pred_thresholds = apply_family_floor(pred_thresholds, test_families)

        # Collect
        for j, (pt, tt, ptr, ttr, fam) in enumerate(zip(
            pred_thresholds, y_thr_test, pred_times, y_time_test, test_families
        )):
            all_pred_thresholds.append(int(pt))
            all_true_thresholds.append(int(tt))
            all_pred_times.append(float(ptr))
            all_true_times.append(float(ttr))
            all_indices.append(test_mask.values.nonzero()[0][j])
            all_families.append(fam)

    # Compute overall score
    scores = compute_overall_score(
        all_pred_thresholds, all_pred_times,
        all_true_thresholds, all_true_times
    )

    return {
        "scores": scores,
        "pred_thresholds": all_pred_thresholds,
        "true_thresholds": all_true_thresholds,
        "pred_times": all_pred_times,
        "true_times": all_true_times,
        "indices": all_indices,
        "families": all_families,
    }


def tune_hyperparameters(
    df: pd.DataFrame,
    X: np.ndarray,
    y_threshold: np.ndarray,
    y_total_time: np.ndarray,
    y_setup_time: np.ndarray,
    y_per_shot_time: np.ndarray,
) -> dict:
    """
    Grid search over hyperparameters using competition score.

    Returns best parameters for threshold and runtime models.
    """
    print("\n  Hyperparameter search (focused grid for speed):")

    # Focused grid - only tune most impactful params
    safety_margins = [0.0, 0.3, 0.5, 0.7, 1.0]

    # Fixed good baseline params (these work well for small datasets)
    base_thr_params = {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.1, "min_samples_leaf": 3}
    base_rt_params = {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.1, "min_samples_leaf": 3}

    best_score = -1
    best_margin = 0.3
    best_use_family = True
    total_evals = 0

    # First compare with vs without family floors
    print("\n  Comparing family floor strategies:")
    for use_family in [False, True]:
        result = leave_one_circuit_out_cv(
            df, X, y_threshold, y_total_time, y_setup_time, y_per_shot_time,
            base_thr_params, base_rt_params, safety_margin=0.0, use_family_floor=use_family
        )
        score = result["scores"]["overall_score"]
        n_fail = result["scores"]["n_threshold_failures"]
        label = "WITH family floors" if use_family else "WITHOUT family floors"
        print(f"    {label}: score={score:.4f}, failures={n_fail}")
        total_evals += 1

    # Tune safety margin WITH family floors (our innovation)
    print(f"\n  Tuning safety margin WITH family floors:")
    for margin in safety_margins:
        result = leave_one_circuit_out_cv(
            df, X, y_threshold, y_total_time, y_setup_time, y_per_shot_time,
            base_thr_params, base_rt_params, safety_margin=margin, use_family_floor=True
        )
        score = result["scores"]["overall_score"]
        thr_score = result["scores"]["mean_threshold_score"]
        rt_score = result["scores"]["mean_runtime_score"]
        n_fail = result["scores"]["n_threshold_failures"]
        print(f"    margin={margin:.1f}: score={score:.4f} (thr={thr_score:.3f}, rt={rt_score:.3f}, failures={n_fail})")
        if score > best_score:
            best_score = score
            best_margin = margin
            best_use_family = True
        total_evals += 1

    print(f"\n  Best configuration:")
    print(f"    Safety margin: {best_margin}")
    print(f"    Use family floors: {best_use_family}")
    print(f"    Best overall score: {best_score:.4f}")
    print(f"  Total evaluations: {total_evals}")

    return {
        "threshold_params": base_thr_params,
        "runtime_params": base_rt_params,
        "safety_margin": best_margin,
        "use_family_floor": best_use_family,
        "best_score": best_score,
    }


def analyze_cv_results(cv_result: dict, df: pd.DataFrame):
    """Print detailed analysis of CV results."""
    scores = cv_result["scores"]

    print(f"\n  Overall competition score: {scores['overall_score']:.4f}")
    print(f"  Mean threshold score: {scores['mean_threshold_score']:.4f}")
    print(f"  Mean runtime score: {scores['mean_runtime_score']:.4f}")
    print(f"  Threshold failures (score=0): {scores['n_threshold_failures']}/{scores['n_tasks']}")

    # Per-circuit analysis
    pred_thr = np.array(cv_result["pred_thresholds"])
    true_thr = np.array(cv_result["true_thresholds"])
    pred_time = np.array(cv_result["pred_times"])
    true_time = np.array(cv_result["true_times"])
    indices = cv_result["indices"]

    # Find worst predictions
    task_scores = []
    for pt, tt, ptr, ttr in zip(pred_thr, true_thr, pred_time, true_time):
        thr_score = compute_threshold_score(pt, tt)
        rt_score = compute_runtime_score(ptr, ttr)
        task_scores.append(thr_score * rt_score)

    task_scores = np.array(task_scores)
    worst_indices = np.argsort(task_scores)[:5]

    print("\n  Worst 5 predictions:")
    for idx in worst_indices:
        orig_idx = indices[idx]
        row = df.iloc[orig_idx]
        thr_score = compute_threshold_score(pred_thr[idx], true_thr[idx])
        rt_score = compute_runtime_score(pred_time[idx], true_time[idx])
        print(f"    {row['file'][:30]:<30} {row['backend']:>3}/{row['precision']:<6} "
              f"thr: {int(true_thr[idx]):>3} -> {int(pred_thr[idx]):>3} (score={thr_score:.2f}), "
              f"time: {true_time[idx]:>7.1f} -> {pred_time[idx]:>7.1f} (score={rt_score:.2f})")

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

    best_params = tune_hyperparameters(
        df, X, y_threshold, y_total_time, y_setup_time, y_per_shot_time
    )

    tune_time = time.time() - tune_start
    print(f"\n  Tuning completed in {tune_time:.1f}s")

    # ── Step 4: Final CV evaluation ────────────────────────────────────
    print("\n[4/5] Final cross-validation evaluation...")

    final_cv = leave_one_circuit_out_cv(
        df, X, y_threshold, y_total_time, y_setup_time, y_per_shot_time,
        best_params["threshold_params"],
        best_params["runtime_params"],
        best_params["safety_margin"],
        best_params.get("use_family_floor", True),
    )

    analyze_cv_results(final_cv, df)

    # ── Step 5: Train final models ─────────────────────────────────────
    print("\n[5/5] Training final models on full data...")

    # Threshold model
    final_thr_model = ThresholdModel(
        **best_params["threshold_params"],
        safety_margin=best_params["safety_margin"]
    )
    final_thr_model.fit(X, y_threshold, calibrate=False)

    # Runtime model
    final_rt_model = RuntimeModel(**best_params["runtime_params"])
    final_rt_model.fit(X, y_total_time, y_setup_time, y_per_shot_time)

    # Combined predictor
    combined = CombinedPredictor(final_thr_model, final_rt_model)
    combined.feature_columns = feature_cols

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
        f.write(f"Runtime Model:\n")
        for k, v in best_params["runtime_params"].items():
            f.write(f"  {k}: {v}\n")
        f.write(f"\nCV Score: {best_params['best_score']:.4f}\n")
    print(f"  Saved parameters to: {params_path}")

    # Save feature columns for prediction script
    feature_path = OUTPUT_DIR / "feature_columns.txt"
    with open(feature_path, "w") as f:
        for col in feature_cols:
            f.write(col + "\n")
    print(f"  Saved feature columns to: {feature_path}")

    # ── Summary ────────────────────────────────────────────────────────
    total_time = time.time() - start_time

    print("\n" + "=" * 70)
    print("Phase 2 COMPLETE")
    print(f"  Training time: {total_time:.1f}s")
    print(f"  CV Competition Score: {final_cv['scores']['overall_score']:.4f}")
    print(f"  Threshold failures: {final_cv['scores']['n_threshold_failures']}/{final_cv['scores']['n_tasks']}")
    print(f"  Model saved to: {OUTPUT_DIR / 'combined_model.pkl'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
