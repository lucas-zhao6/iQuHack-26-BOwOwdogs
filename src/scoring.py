"""
Competition scoring utilities.

Implements the exact scoring formula from the challenge:
- Threshold score: 2^(-steps_over) if pred >= true, else 0
- Runtime score: min(r, 1/r) where r = pred/true
- Task score: threshold_score * runtime_score
- Overall score: mean(task_scores)
"""

from __future__ import annotations

import math
from typing import List, Tuple, Dict
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
    Compute threshold component of task score.

    - If pred < true: return 0 (fidelity violated - catastrophic)
    - If pred >= true: return 2^(-steps_over)
    """
    if pred_threshold < true_threshold:
        return 0.0

    pred_idx = threshold_to_idx(pred_threshold)
    true_idx = threshold_to_idx(true_threshold)
    steps_over = pred_idx - true_idx

    return 2.0 ** (-steps_over)


def compute_runtime_score(pred_time: float, true_time: float) -> float:
    """
    Compute runtime component of task score.

    runtime_score = min(r, 1/r) where r = pred/true
    Symmetric penalty: 2x over or under both give 0.5
    """
    if true_time <= 0 or pred_time <= 0:
        return 0.0

    r = pred_time / true_time
    return min(r, 1.0 / r)


def compute_task_score(
    pred_threshold: int,
    pred_time: float,
    true_threshold: int,
    true_time: float,
) -> Tuple[float, float, float]:
    """
    Compute full task score.

    Returns: (task_score, threshold_score, runtime_score)
    """
    thr_score = compute_threshold_score(pred_threshold, true_threshold)
    rt_score = compute_runtime_score(pred_time, true_time)
    task_score = thr_score * rt_score
    return task_score, thr_score, rt_score


def compute_overall_score(
    pred_thresholds: List[int],
    pred_times: List[float],
    true_thresholds: List[int],
    true_times: List[float],
) -> Dict[str, float]:
    """
    Compute overall competition score across all tasks.

    Returns dict with:
    - overall_score: mean of task scores
    - mean_threshold_score: mean of threshold scores
    - mean_runtime_score: mean of runtime scores
    - n_threshold_failures: count of pred < true (score = 0)
    """
    n = len(pred_thresholds)
    task_scores = []
    thr_scores = []
    rt_scores = []
    n_failures = 0

    for i in range(n):
        ts, ths, rts = compute_task_score(
            pred_thresholds[i], pred_times[i],
            true_thresholds[i], true_times[i]
        )
        task_scores.append(ts)
        thr_scores.append(ths)
        rt_scores.append(rts)
        if ths == 0:
            n_failures += 1

    return {
        "overall_score": np.mean(task_scores),
        "mean_threshold_score": np.mean(thr_scores),
        "mean_runtime_score": np.mean(rt_scores),
        "n_threshold_failures": n_failures,
        "n_tasks": n,
    }


def asymmetric_threshold_loss(y_true_idx: np.ndarray, y_pred_idx: np.ndarray) -> float:
    """
    Asymmetric loss for threshold prediction that matches competition scoring.

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
    return np.mean(losses)


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
