"""
Scoring utilities.

Threshold score:
- 0 if pred < true
- 2^(-steps_over) if pred >= true

Runtime score:
- -abs(log(pred/true))
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np


THRESHOLD_RUNGS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
THRESHOLD_TO_IDX = {t: i for i, t in enumerate(THRESHOLD_RUNGS)}
IDX_TO_THRESHOLD = {i: t for i, t in enumerate(THRESHOLD_RUNGS)}


def threshold_to_idx(threshold: int) -> int:
    """Convert threshold value to index (0-8)."""
    return THRESHOLD_TO_IDX.get(threshold, int(math.log2(threshold)))


def idx_to_threshold(idx: int) -> int:
    """Convert index (0-8) to threshold value."""
    idx = max(0, min(8, int(round(idx))))
    return IDX_TO_THRESHOLD.get(idx, 2 ** idx)


def compute_threshold_score(pred_threshold: int, true_threshold: int) -> float:
    """
    Compute threshold score.

    - If pred < true: return 0 (fidelity violated - catastrophic)
    - If pred >= true: return 2^(-steps_over)
    """
    if pred_threshold < true_threshold:
        return 0.0

    pred_idx = threshold_to_idx(pred_threshold)
    true_idx = threshold_to_idx(true_threshold)
    steps_over = pred_idx - true_idx

    return 2.0 ** (-steps_over)


def compute_runtime_log_score(pred_time: float, true_time: float) -> float:
    """Compute runtime score as -abs(log(pred/true))."""
    if true_time <= 0 or pred_time <= 0:
        return float("nan")
    if not np.isfinite(true_time) or not np.isfinite(pred_time):
        return float("nan")
    return -abs(float(np.log(pred_time / true_time)))


def compute_threshold_metrics(
    pred_thresholds: List[int],
    true_thresholds: List[int],
) -> Dict[str, float]:
    scores = []
    n_failures = 0
    for pred, true in zip(pred_thresholds, true_thresholds):
        score = compute_threshold_score(pred, true)
        scores.append(score)
        if score == 0:
            n_failures += 1

    return {
        "mean_threshold_score": float(np.mean(scores)) if scores else 0.0,
        "n_threshold_failures": n_failures,
        "n_tasks": len(pred_thresholds),
    }


def compute_runtime_metrics(
    pred_times: List[float],
    true_times: List[float],
) -> Dict[str, float]:
    scores = []
    invalid = 0
    for pred, true in zip(pred_times, true_times):
        score = compute_runtime_log_score(pred, true)
        if np.isnan(score):
            invalid += 1
            continue
        scores.append(score)

    mean_score = float(np.mean(scores)) if scores else float("nan")
    return {
        "mean_runtime_log_score": mean_score,
        "n_runtime_invalid": invalid,
        "n_tasks": len(pred_times),
    }


def asymmetric_threshold_loss(y_true_idx: np.ndarray, y_pred_idx: np.ndarray) -> float:
    """
    Asymmetric loss for threshold prediction that matches threshold scoring.

    Under-prediction is catastrophic (loss = 1.0).
    Over-prediction has exponential decay (loss = 1 - 2^(-steps_over)).
    """
    losses = []
    for true_idx, pred_idx in zip(y_true_idx, y_pred_idx):
        pred_idx_rounded = int(round(pred_idx))
        if pred_idx_rounded < true_idx:
            losses.append(1.0)  # catastrophic
        else:
            steps_over = pred_idx_rounded - true_idx
            losses.append(1.0 - 2.0 ** (-steps_over))
    return float(np.mean(losses)) if losses else 0.0


def find_optimal_safety_margin(
    y_true_idx: np.ndarray,
    y_pred_raw: np.ndarray,
    margins: List[float] = None,
) -> Tuple[float, float]:
    """
    Find the optimal safety margin that maximizes threshold score.

    The safety margin is added to raw predictions before rounding.
    Higher margin = more conservative (bias toward over-prediction).

    Returns: (best_margin, best_score)
    """
    if margins is None:
        margins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    best_margin = 0.0
    best_score = -1.0

    for margin in margins:
        y_pred_adjusted = y_pred_raw + margin
        y_pred_rounded = np.clip(np.round(y_pred_adjusted), 0, 8).astype(int)

        # Convert to thresholds and compute scores
        scores = []
        for true_idx, pred_idx in zip(y_true_idx, y_pred_rounded):
            true_thr = idx_to_threshold(true_idx)
            pred_thr = idx_to_threshold(pred_idx)
            scores.append(compute_threshold_score(pred_thr, true_thr))

        mean_score = np.mean(scores)
        if mean_score > best_score:
            best_score = mean_score
            best_margin = margin

    return best_margin, best_score
