"""
Shared evaluation utilities for model comparison.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .scoring import compute_overall_score, compute_threshold_score, compute_runtime_score


def leave_one_circuit_out_cv(
    df: pd.DataFrame,
    X: np.ndarray,
    y_threshold: np.ndarray,
    y_total_time: np.ndarray,
    y_setup_time: np.ndarray,
    y_per_shot_time: np.ndarray,
    fit_predict_fn: Callable,
) -> dict:
    """
    Generic leave-one-circuit-out CV.

    fit_predict_fn signature:
        (X_train, X_test, y_thr_train, y_time_train, y_setup_train, y_per_shot_train, test_families)
        -> (pred_thresholds, pred_times)
    """
    circuits = df["file"].unique()

    all_pred_thresholds = []
    all_true_thresholds = []
    all_pred_times = []
    all_true_times = []
    all_indices = []
    all_families = []

    for circuit in circuits:
        test_mask = df["file"] == circuit
        train_mask = ~test_mask

        X_train, X_test = X[train_mask], X[test_mask]
        y_thr_train, y_thr_test = y_threshold[train_mask], y_threshold[test_mask]
        y_time_train, y_time_test = y_total_time[train_mask], y_total_time[test_mask]
        y_setup_train = y_setup_time[train_mask]
        y_per_shot_train = y_per_shot_time[train_mask]

        test_families = df.loc[test_mask, "predicted_family"].tolist()

        pred_thresholds, pred_times = fit_predict_fn(
            X_train,
            X_test,
            y_thr_train,
            y_time_train,
            y_setup_train,
            y_per_shot_train,
            test_families,
        )

        for j, (pt, tt, ptr, ttr, fam) in enumerate(
            zip(pred_thresholds, y_thr_test, pred_times, y_time_test, test_families)
        ):
            all_pred_thresholds.append(int(pt))
            all_true_thresholds.append(int(tt))
            all_pred_times.append(float(ptr))
            all_true_times.append(float(ttr))
            all_indices.append(test_mask.values.nonzero()[0][j])
            all_families.append(fam)

    scores = compute_overall_score(
        all_pred_thresholds, all_pred_times, all_true_thresholds, all_true_times
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


def analyze_cv_results(cv_result: dict, df: pd.DataFrame) -> None:
    """Print detailed analysis of CV results."""
    scores = cv_result["scores"]

    print(f"\n  Overall competition score: {scores['overall_score']:.4f}")
    print(f"  Mean threshold score: {scores['mean_threshold_score']:.4f}")
    print(f"  Mean runtime score: {scores['mean_runtime_score']:.4f}")
    print(
        f"  Threshold failures (score=0): {scores['n_threshold_failures']}/{scores['n_tasks']}"
    )

    pred_thr = np.array(cv_result["pred_thresholds"])
    true_thr = np.array(cv_result["true_thresholds"])
    pred_time = np.array(cv_result["pred_times"])
    true_time = np.array(cv_result["true_times"])
    indices = cv_result["indices"]

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
        print(
            f"    {row['file'][:30]:<30} {row['backend']:>3}/{row['precision']:<6} "
            f"thr: {int(true_thr[idx]):>3} -> {int(pred_thr[idx]):>3} (score={thr_score:.2f}), "
            f"time: {true_time[idx]:>7.1f} -> {pred_time[idx]:>7.1f} (score={rt_score:.2f})"
        )

    print("\n  Threshold prediction summary:")
    unique_thresholds = sorted(set(true_thr.astype(int)))
    for thr in unique_thresholds:
        mask = true_thr == thr
        preds = pred_thr[mask]
        under = (preds < thr).sum()
        exact = (preds == thr).sum()
        over = (preds > thr).sum()
        print(f"    True={thr:>3}: under={under:>2}, exact={exact:>2}, over={over:>2}")
