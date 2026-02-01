"""
Feature selection for reducing dimensionality.

With 137 samples and 99 features, we're at risk of overfitting.
This module selects the top K most informative features based on
correlation with the target variables.
"""

from __future__ import annotations

from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression


def select_features_by_correlation(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    k: int = 25,
    min_variance: float = 1e-6,
) -> Tuple[List[int], List[str], Dict[str, float]]:
    """
    Select top K features by absolute correlation with target.

    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target vector
        feature_names: Names of features
        k: Number of features to select
        min_variance: Minimum variance threshold (remove near-constant)

    Returns:
        selected_indices: Indices of selected features
        selected_names: Names of selected features
        correlations: Dict of feature_name -> correlation
    """
    n_features = X.shape[1]
    correlations = {}

    for i in range(n_features):
        col = X[:, i]

        # Skip near-constant features
        if np.std(col) < min_variance:
            correlations[feature_names[i]] = 0.0
            continue

        # Skip features with NaN
        valid = np.isfinite(col) & np.isfinite(y)
        if valid.sum() < 10:
            correlations[feature_names[i]] = 0.0
            continue

        # Compute Pearson correlation
        corr = np.corrcoef(col[valid], y[valid])[0, 1]
        if np.isfinite(corr):
            correlations[feature_names[i]] = abs(corr)
        else:
            correlations[feature_names[i]] = 0.0

    # Sort by correlation and select top K
    sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
    selected_names = [f[0] for f in sorted_features[:k]]
    selected_indices = [feature_names.index(name) for name in selected_names]

    return selected_indices, selected_names, correlations


def select_features_by_mutual_info(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    k: int = 25,
    min_variance: float = 1e-6,
    random_state: int = 42,
) -> Tuple[List[int], List[str], Dict[str, float]]:
    """
    Select top K features by mutual information with target.
    """
    n_features = X.shape[1]
    scores = {name: 0.0 for name in feature_names}

    valid_indices = []
    for i in range(n_features):
        col = X[:, i]
        if np.std(col) < min_variance:
            continue
        if not np.isfinite(col).any():
            continue
        valid_indices.append(i)

    if len(valid_indices) > 0:
        X_valid = X[:, valid_indices]
        y_valid = y.copy()
        mask = np.isfinite(y_valid)
        if mask.sum() > 5:
            mi = mutual_info_regression(
                X_valid[mask], y_valid[mask], random_state=random_state
            )
            for idx, val in zip(valid_indices, mi):
                scores[feature_names[idx]] = float(val)

    sorted_features = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    selected_names = [f[0] for f in sorted_features[:k]]
    selected_indices = [feature_names.index(name) for name in selected_names]
    return selected_indices, selected_names, scores


def select_features_combined(
    X: np.ndarray,
    y_threshold: np.ndarray,
    y_runtime: np.ndarray,
    feature_names: List[str],
    k: int = 30,
) -> Tuple[List[int], List[str]]:
    """
    Select features that are informative for BOTH threshold and runtime prediction.

    Uses a combined score: max(corr_threshold, corr_runtime) to ensure
    features useful for either task are included.
    """
    _, _, corr_thr = select_features_by_correlation(
        X, y_threshold, feature_names, k=len(feature_names)
    )
    _, _, corr_rt = select_features_by_correlation(
        X, y_runtime, feature_names, k=len(feature_names)
    )

    combined_scores = {
        name: max(corr_thr.get(name, 0.0), corr_rt.get(name, 0.0))
        for name in feature_names
    }
    sorted_features = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    selected_names = [f[0] for f in sorted_features[:k]]
    selected_indices = [feature_names.index(name) for name in selected_names]
    return selected_indices, selected_names


def select_features_split(
    X: np.ndarray,
    y_threshold: np.ndarray,
    y_runtime: np.ndarray,
    feature_names: List[str],
    k_threshold: int = 30,
    k_runtime: int = 30,
    method: str = "corr",
) -> Tuple[List[int], List[str], List[int], List[str]]:
    """
    Select separate feature sets for threshold and runtime prediction.
    """
    if method == "mi":
        _, _, corr_thr = select_features_by_mutual_info(
            X, y_threshold, feature_names, k=len(feature_names)
        )
        _, _, corr_rt = select_features_by_mutual_info(
            X, y_runtime, feature_names, k=len(feature_names)
        )
    else:
        _, _, corr_thr = select_features_by_correlation(
            X, y_threshold, feature_names, k=len(feature_names)
        )
        _, _, corr_rt = select_features_by_correlation(
            X, y_runtime, feature_names, k=len(feature_names)
        )

    thr_sorted = sorted(corr_thr.items(), key=lambda x: x[1], reverse=True)
    rt_sorted = sorted(corr_rt.items(), key=lambda x: x[1], reverse=True)

    thr_names = [f[0] for f in thr_sorted[:k_threshold]]
    rt_names = [f[0] for f in rt_sorted[:k_runtime]]
    thr_indices = [feature_names.index(name) for name in thr_names]
    rt_indices = [feature_names.index(name) for name in rt_names]
    return thr_indices, thr_names, rt_indices, rt_names


class FeatureSelector:
    """
    Feature selector that can be fitted and applied consistently.

    Supports split selection (different feature sets for threshold vs runtime).
    """

    def __init__(
        self,
        k: int = 30,
        k_threshold: int | None = None,
        k_runtime: int | None = None,
        method: str = "corr",
    ):
        self.k_threshold = k_threshold if k_threshold is not None else k
        self.k_runtime = k_runtime if k_runtime is not None else k
        self.method = method
        self.selected_indices_threshold: List[int] = []
        self.selected_names_threshold: List[str] = []
        self.selected_indices_runtime: List[int] = []
        self.selected_names_runtime: List[str] = []
        self.all_feature_names: List[str] = []
        self._fitted = False

    def fit(
        self,
        X: np.ndarray,
        y_threshold: np.ndarray,
        y_runtime: np.ndarray,
        feature_names: List[str],
    ):
        """Fit the selector to find best features."""
        self.all_feature_names = feature_names

        # Convert threshold to log2 scale for correlation
        y_thr_log = np.log2(np.clip(y_threshold, 1, 512))
        y_rt_log = np.log1p(y_runtime)

        (
            self.selected_indices_threshold,
            self.selected_names_threshold,
            self.selected_indices_runtime,
            self.selected_names_runtime,
        ) = select_features_split(
            X, y_thr_log, y_rt_log, feature_names,
            k_threshold=self.k_threshold, k_runtime=self.k_runtime,
            method=self.method,
        )
        self._fitted = True

    def transform_threshold(self, X: np.ndarray) -> np.ndarray:
        """Apply threshold feature selection."""
        if not self._fitted:
            raise RuntimeError("FeatureSelector not fitted")
        return X[:, self.selected_indices_threshold]

    def transform_runtime(self, X: np.ndarray) -> np.ndarray:
        """Apply runtime feature selection."""
        if not self._fitted:
            raise RuntimeError("FeatureSelector not fitted")
        return X[:, self.selected_indices_runtime]

    def transform(self, X: np.ndarray, target: str = "threshold") -> np.ndarray:
        """Apply feature selection for a specific target."""
        if target == "threshold":
            return self.transform_threshold(X)
        if target == "runtime":
            return self.transform_runtime(X)
        raise ValueError(f"Unknown target: {target}")

    def fit_transform(
        self,
        X: np.ndarray,
        y_threshold: np.ndarray,
        y_runtime: np.ndarray,
        feature_names: List[str],
    ) -> np.ndarray:
        """Fit and transform in one step (threshold selection)."""
        self.fit(X, y_threshold, y_runtime, feature_names)
        return self.transform_threshold(X)

    def get_selected_names(self, target: str = "threshold") -> List[str]:
        """Return names of selected features for a specific target."""
        if target == "threshold":
            return self.selected_names_threshold.copy()
        if target == "runtime":
            return self.selected_names_runtime.copy()
        raise ValueError(f"Unknown target: {target}")
