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

    Args:
        X: Feature matrix
        y_threshold: Threshold target (log2 scale recommended)
        y_runtime: Runtime target (log scale recommended)
        feature_names: Feature names
        k: Number of features to select

    Returns:
        selected_indices, selected_names
    """
    # Get correlations for both targets
    _, _, corr_thr = select_features_by_correlation(
        X, y_threshold, feature_names, k=len(feature_names)
    )
    _, _, corr_rt = select_features_by_correlation(
        X, y_runtime, feature_names, k=len(feature_names)
    )

    # Combined score: max of the two correlations
    combined_scores = {}
    for name in feature_names:
        combined_scores[name] = max(
            corr_thr.get(name, 0.0),
            corr_rt.get(name, 0.0)
        )

    # Sort and select
    sorted_features = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    selected_names = [f[0] for f in sorted_features[:k]]
    selected_indices = [feature_names.index(name) for name in selected_names]

    return selected_indices, selected_names


class FeatureSelector:
    """
    Feature selector that can be fitted and applied consistently.
    """

    def __init__(self, k: int = 30):
        self.k = k
        self.selected_indices: List[int] = []
        self.selected_names: List[str] = []
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

        self.selected_indices, self.selected_names = select_features_combined(
            X, y_thr_log, y_rt_log, feature_names, self.k
        )
        self._fitted = True

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply feature selection."""
        if not self._fitted:
            raise RuntimeError("FeatureSelector not fitted")
        return X[:, self.selected_indices]

    def fit_transform(
        self,
        X: np.ndarray,
        y_threshold: np.ndarray,
        y_runtime: np.ndarray,
        feature_names: List[str],
    ) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X, y_threshold, y_runtime, feature_names)
        return self.transform(X)

    def get_selected_names(self) -> List[str]:
        """Return names of selected features."""
        return self.selected_names.copy()
