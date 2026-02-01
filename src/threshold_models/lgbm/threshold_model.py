"""
Threshold model and related helpers.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import warnings

import numpy as np

try:
    import lightgbm as lgb
except ImportError as exc:
    raise ImportError(
        "LightGBM is required for training. Install with 'pip install lightgbm'."
    ) from exc

from ...evaluation.scoring import (
    THRESHOLD_RUNGS,
    idx_to_threshold,
    threshold_to_idx,
    compute_threshold_score,
)

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="X does not have valid feature names, but LGBMRegressor was fitted with feature names",
)


class ThresholdModel:
    """
    Predicts the minimum threshold rung needed to achieve >= 0.75 fidelity.

    Uses LightGBM multiclass classification over the 9 threshold rungs.
    Prediction uses a risk-aware decision rule that maximizes expected
    threshold score under the competition metric.
    """

    def __init__(
        self,
        lgb_params: Optional[Dict[str, Any]] = None,
        n_estimators: int = 1000,
        early_stopping_rounds: int = 50,
        decision_policy: str = "expected_score",
        safety_margin: float = 0.0,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.early_stopping_rounds = early_stopping_rounds
        self.decision_policy = decision_policy
        self.safety_margin = safety_margin
        self.random_state = random_state

        default_params = {
            "objective": "multiclass",
            "num_class": 9,
            "n_estimators": n_estimators,
            "learning_rate": 0.03,
            "max_depth": 4,
            "num_leaves": 15,
            "min_child_samples": 20,
            "min_gain_to_split": 0.01,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "lambda_l1": 1.0,
            "lambda_l2": 2.0,
            "extra_trees": True,
            "random_state": random_state,
            "verbosity": -1,
        }
        if lgb_params:
            default_params.update(lgb_params)

        self.params = default_params
        self.model = lgb.LGBMClassifier(**self.params)
        self._fitted = False
        self._score_matrix = self._build_score_matrix()

    def _build_score_matrix(self) -> np.ndarray:
        n_classes = len(THRESHOLD_RUNGS)
        matrix = np.zeros((n_classes, n_classes), dtype=float)
        for pred_idx in range(n_classes):
            pred_thr = idx_to_threshold(pred_idx)
            for true_idx in range(n_classes):
                true_thr = idx_to_threshold(true_idx)
                matrix[pred_idx, true_idx] = compute_threshold_score(pred_thr, true_thr)
        return matrix

    def _predict_proba_full(self, X: np.ndarray) -> np.ndarray:
        """Return predict_proba padded to all 9 threshold classes."""
        X = np.asarray(X)
        proba = self.model.predict_proba(X)
        n_classes = len(THRESHOLD_RUNGS)
        if proba.shape[1] == n_classes:
            return proba

        full = np.zeros((proba.shape[0], n_classes), dtype=float)
        classes = getattr(self.model, "classes_", None)
        if classes is None:
            raise RuntimeError("Model classes_ not available for probability padding.")
        for src_idx, class_label in enumerate(classes):
            class_idx = int(class_label)
            if 0 <= class_idx < n_classes:
                full[:, class_idx] = proba[:, src_idx]
        return full

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for all 9 threshold rungs."""
        return self._predict_proba_full(X)

    def fit(
        self,
        X: np.ndarray,
        y_threshold: np.ndarray,
        eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ):
        """
        Fit the threshold model.

        Args:
            X: Feature matrix (n_samples, n_features)
            y_threshold: Threshold values (1, 2, 4, ..., 256)
            eval_set: Optional (X_val, y_val) for early stopping
        """
        X = np.asarray(X)
        y_idx = np.array([threshold_to_idx(t) for t in y_threshold])

        callbacks = []
        fit_kwargs = {}
        if eval_set is not None:
            X_val, y_val = eval_set
            X_val = np.asarray(X_val)
            y_val_idx = np.array([threshold_to_idx(t) for t in y_val])
            fit_kwargs["eval_set"] = [(X_val, y_val_idx)]
            fit_kwargs["eval_metric"] = "multi_logloss"
            if self.early_stopping_rounds > 0:
                callbacks.append(
                    lgb.early_stopping(self.early_stopping_rounds, verbose=False)
                )
        if callbacks:
            fit_kwargs["callbacks"] = callbacks

        self.model.fit(X, y_idx, **fit_kwargs)
        self._fitted = True

    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        """Predict expected threshold index (continuous)."""
        proba = self._predict_proba_full(X)
        idxs = np.arange(proba.shape[1])
        return proba @ idxs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict threshold values with risk-aware decision rule."""
        proba = self._predict_proba_full(X)
        if self.decision_policy == "argmax":
            pred_idx = np.argmax(proba, axis=1)
        else:
            expected_scores = proba @ self._score_matrix.T
            pred_idx = np.argmax(expected_scores, axis=1)

        if self.safety_margin != 0.0:
            pred_idx = np.clip(np.round(pred_idx + self.safety_margin), 0, 8).astype(int)

        return np.array([idx_to_threshold(i) for i in pred_idx])

    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict thresholds with uncertainty estimate.

        Returns: (predictions, uncertainties)
        Uncertainty is estimated from the training residuals at similar feature values.
        """
        proba = self._predict_proba_full(X)
        predictions = self.predict(X)
        uncertainty = 1.0 - np.max(proba, axis=1)
        return predictions, uncertainty


# Families that need curve-aware prediction (high threshold families)
# For these families, we use max(direct, curve) to avoid under-prediction
HIGH_THRESHOLD_FAMILIES = {
    "TwoLocalRandom", "QNN", "Portfolio_QAOA", "Portfolio_VQE",
    "Pricing_Call", "Ground_State", "Amplitude_Estimation", "Shor", "GraphState"
}


# Family-to-minimum-threshold mapping from training data analysis
# This is our Innovation #3: use detected family to set floor thresholds
FAMILY_THRESHOLD_FLOORS = {
    # High threshold families (these NEVER work at low thresholds)
    "TwoLocalRandom": 64,       # Needs 256 but 64 is safe floor
    "QNN": 16,                  # Needs 32
    "Portfolio_QAOA": 8,        # Needs 16
    "Portfolio_VQE": 8,         # Needs 16
    "Pricing_Call": 4,          # Needs 8
    "Ground_State": 4,          # Needs 8
    "Amplitude_Estimation": 4,  # Needs 4-16
    "GraphState": 2,            # Needs 4
    "Shor": 2,                  # Needs 4

    # Medium threshold families
    "CutBell": 2,
    "QFT_Entangled": 2,
    "VQE": 2,
    "W_State": 2,
    "GHZ": 2,

    # Low threshold families (easy for MPS)
    "Grover_V_Chain": 1,
    "Grover_NoAncilla": 1,
    "QAOA": 1,
    "QFT": 1,
    "QPE_Exact": 1,
    "Deutsch_Jozsa": 1,

    # Unknown - be conservative
    "Unknown": 4,
}


def apply_family_floor(
    predicted_thresholds: np.ndarray,
    families: List[str],
) -> np.ndarray:
    """
    Apply family-based minimum threshold floors.

    This prevents under-prediction for families that are known to require
    high thresholds, regardless of what the ML model predicts.
    """
    result = predicted_thresholds.copy()
    for i, family in enumerate(families):
        floor = FAMILY_THRESHOLD_FLOORS.get(family, 1)
        if result[i] < floor:
            result[i] = floor
    return result
