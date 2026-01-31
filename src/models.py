"""
Model definitions for the Circuit Fingerprint Challenge.

Implements:
1. ThresholdModel: Ordinal regression with asymmetric safety margin calibration
2. RuntimeModel: Decomposed prediction (setup + per_shot * 10000)
3. CombinedPredictor: End-to-end prediction pipeline
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from .scoring import (
    idx_to_threshold,
    threshold_to_idx,
    find_optimal_safety_margin,
    compute_threshold_score,
)


class ThresholdModel:
    """
    Predicts the minimum threshold rung needed to achieve >= 0.99 fidelity.

    Uses ordinal regression (predict threshold index 0-8) with a calibrated
    safety margin that biases predictions upward to avoid catastrophic
    under-prediction (score = 0).

    Innovation: The safety margin is tuned on cross-validation to maximize
    the actual competition scoring metric, not surrogate accuracy.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 4,
        learning_rate: float = 0.1,
        min_samples_leaf: int = 3,
        safety_margin: float = 0.3,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_samples_leaf = min_samples_leaf
        self.safety_margin = safety_margin

        self.scaler = StandardScaler()
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
        )
        self._fitted = False

    def fit(self, X: np.ndarray, y_threshold: np.ndarray, calibrate: bool = True):
        """
        Fit the threshold model.

        Args:
            X: Feature matrix (n_samples, n_features)
            y_threshold: Threshold values (1, 2, 4, ..., 256)
            calibrate: If True, find optimal safety margin on training data
        """
        # Convert thresholds to indices (0-8)
        y_idx = np.array([threshold_to_idx(t) for t in y_threshold])

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Fit regression model
        self.model.fit(X_scaled, y_idx)
        self._fitted = True

        # Calibrate safety margin
        if calibrate:
            y_pred_raw = self.model.predict(X_scaled)
            best_margin, best_score = find_optimal_safety_margin(y_idx, y_pred_raw)
            self.safety_margin = best_margin

    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        """Predict raw threshold index (continuous, before rounding)."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict threshold values with safety margin applied."""
        raw = self.predict_raw(X)
        adjusted = raw + self.safety_margin
        idx_rounded = np.clip(np.round(adjusted), 0, 8).astype(int)
        return np.array([idx_to_threshold(i) for i in idx_rounded])

    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict thresholds with uncertainty estimate.

        Returns: (predictions, uncertainties)
        Uncertainty is estimated from the training residuals at similar feature values.
        """
        # Simple uncertainty: distance from round boundary
        raw = self.predict_raw(X)
        adjusted = raw + self.safety_margin
        uncertainty = 0.5 - np.abs(adjusted - np.round(adjusted))
        predictions = self.predict(X)
        return predictions, uncertainty


class RuntimeModel:
    """
    Predicts forward run wall time using decomposed modeling.

    Innovation: Instead of predicting total time directly, we model:
        total_time = setup_time + per_shot_time * 10000

    The model now takes predicted_threshold as an additional input feature,
    since higher threshold = more computation = longer runtime.

    Setup time is dominated by backend (GPU has 23x overhead).
    Per-shot time is dominated by qubit count, threshold, and circuit depth.

    Both are modeled in log-space and combined.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 4,
        learning_rate: float = 0.1,
        min_samples_leaf: int = 3,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_samples_leaf = min_samples_leaf

        self.scaler = StandardScaler()

        # Two separate models for setup and per-shot
        self.setup_model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
        )
        self.per_shot_model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            min_samples_leaf=min_samples_leaf,
            random_state=43,
        )

        # Fallback: direct total time model
        self.total_model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            min_samples_leaf=min_samples_leaf,
            random_state=44,
        )

        self._fitted = False
        self._has_decomposed = False

    def _add_threshold_feature(self, X: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
        """Add log2(threshold) as an additional feature column."""
        # Use log2 of threshold as feature (1->0, 2->1, 4->2, ..., 256->8)
        thr_feature = np.log2(np.clip(thresholds, 1, 512)).reshape(-1, 1)
        return np.hstack([X, thr_feature])

    def fit(
        self,
        X: np.ndarray,
        y_total_time: np.ndarray,
        thresholds: np.ndarray,
        y_setup_time: Optional[np.ndarray] = None,
        y_per_shot_time: Optional[np.ndarray] = None,
    ):
        """
        Fit the runtime model.

        Args:
            X: Feature matrix (n_samples, n_features)
            y_total_time: Total forward run time
            thresholds: Threshold values (used as input feature)
            y_setup_time: Optional setup time for decomposed modeling
            y_per_shot_time: Optional per-shot time for decomposed modeling
        """
        # Add threshold as input feature
        X_with_thr = self._add_threshold_feature(X, thresholds)
        X_scaled = self.scaler.fit_transform(X_with_thr)

        # Always fit total model as fallback
        y_log_total = np.log1p(y_total_time)
        self.total_model.fit(X_scaled, y_log_total)

        # Try decomposed if data available
        if y_setup_time is not None and y_per_shot_time is not None:
            # Filter out invalid values
            valid = (y_setup_time > 0) & (y_per_shot_time > 0) & np.isfinite(y_setup_time) & np.isfinite(y_per_shot_time)
            if valid.sum() > 10:
                y_log_setup = np.log1p(y_setup_time[valid])
                y_log_per_shot = np.log1p(y_per_shot_time[valid])
                X_valid = X_scaled[valid]

                self.setup_model.fit(X_valid, y_log_setup)
                self.per_shot_model.fit(X_valid, y_log_per_shot)
                self._has_decomposed = True

        self._fitted = True

    def predict(
        self,
        X: np.ndarray,
        thresholds: np.ndarray,
        use_decomposed: bool = True,
    ) -> np.ndarray:
        """
        Predict total forward run time.

        Args:
            X: Feature matrix
            thresholds: Predicted threshold values (used as input feature)
            use_decomposed: Whether to use decomposed prediction
        """
        X_with_thr = self._add_threshold_feature(X, thresholds)
        X_scaled = self.scaler.transform(X_with_thr)

        if use_decomposed and self._has_decomposed:
            log_setup = self.setup_model.predict(X_scaled)
            log_per_shot = self.per_shot_model.predict(X_scaled)

            setup = np.expm1(log_setup)
            per_shot = np.expm1(log_per_shot)
            total = setup + per_shot * 10000

            # Sanity bounds
            total = np.clip(total, 0.1, 10000)
            return total
        else:
            log_total = self.total_model.predict(X_scaled)
            return np.expm1(log_total)

    def predict_decomposed(
        self,
        X: np.ndarray,
        thresholds: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict setup time, per-shot time, and total time.

        Returns: (setup_time, per_shot_time, total_time)
        """
        X_with_thr = self._add_threshold_feature(X, thresholds)
        X_scaled = self.scaler.transform(X_with_thr)

        if self._has_decomposed:
            log_setup = self.setup_model.predict(X_scaled)
            log_per_shot = self.per_shot_model.predict(X_scaled)
            setup = np.expm1(log_setup)
            per_shot = np.expm1(log_per_shot)
            total = setup + per_shot * 10000
        else:
            total = self.predict(X, thresholds, use_decomposed=False)
            # Rough decomposition for non-decomposed model
            setup = total * 0.05  # Assume 5% is setup
            per_shot = (total - setup) / 10000

        return setup, per_shot, total


class CombinedPredictor:
    """
    End-to-end predictor combining threshold and runtime models.

    Provides a single interface for making predictions on new circuits.
    Supports feature selection and sequential prediction (threshold -> runtime).

    Innovation 5: Optional curve predictor for family-aware safety net.
    For high-threshold families, uses max(direct, curve) to avoid under-prediction.

    Additional enhancements:
    - Runtime curve predictor: predict runtime at each threshold rung
    - Auxiliary predictor: predict sweep-derived features for enrichment
    """

    def __init__(
        self,
        threshold_model: Optional[ThresholdModel] = None,
        runtime_model: Optional[RuntimeModel] = None,
    ):
        self.threshold_model = threshold_model or ThresholdModel()
        self.runtime_model = runtime_model or RuntimeModel()
        self.feature_columns: List[str] = []
        self.enriched_feature_columns: List[str] = []  # After auxiliary enrichment
        self.selected_feature_names: List[str] = []
        self.selected_feature_names_threshold: List[str] = []
        self.selected_feature_names_runtime: List[str] = []
        self.feature_selector = None  # Optional FeatureSelector
        self.curve_predictor = None  # Optional FidelityCurvePredictor (Innovation 5)
        self.rt_curve_predictor = None  # Optional RuntimeCurvePredictor
        self.aux_predictor = None  # Optional AuxiliaryFeaturePredictor

    def fit(
        self,
        X: np.ndarray,
        y_threshold: np.ndarray,
        y_total_time: np.ndarray,
        y_setup_time: Optional[np.ndarray] = None,
        y_per_shot_time: Optional[np.ndarray] = None,
        feature_columns: Optional[List[str]] = None,
    ):
        """Fit both models (legacy method, prefer training separately)."""
        self.threshold_model.fit(X, y_threshold)
        # For legacy fit, use true thresholds
        self.runtime_model.fit(X, y_total_time, y_threshold, y_setup_time, y_per_shot_time)

        if feature_columns is not None:
            self.feature_columns = feature_columns

    def predict(self, X: np.ndarray, families: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict threshold and runtime using sequential prediction.

        Steps:
        1. Enrich features with auxiliary predictions (if aux_predictor is set)
        2. Apply feature selection (if selector is set)
        3. Predict thresholds (direct model)
        4. Apply curve predictor safety net for high-threshold families (Innovation 5)
        5. Apply family floors (if families provided)
        6. Predict runtime using predicted thresholds
        7. Blend with runtime curve predictor (if available)

        Returns: (predicted_thresholds, predicted_times)
        """
        # Step 0: Enrich features with auxiliary predictions if available
        X_enriched = X
        if self.aux_predictor is not None:
            aux_features = self.aux_predictor.predict_as_features(X)
            if aux_features.shape[1] > 0:
                X_enriched = np.hstack([X, aux_features])

        # Apply feature selection if available
        if self.feature_selector is not None:
            if hasattr(self.feature_selector, "transform_threshold") and hasattr(self.feature_selector, "transform_runtime"):
                X_thr = self.feature_selector.transform_threshold(X_enriched)
                X_rt = self.feature_selector.transform_runtime(X_enriched)
            else:
                X_thr = self.feature_selector.transform(X_enriched)
                X_rt = X_thr
        else:
            X_thr = X_enriched
            X_rt = X_enriched

        # Step 1: Predict thresholds (direct model)
        thresholds = self.threshold_model.predict(X_thr)

        # Step 2: Apply curve predictor safety net (Innovation 5)
        # For high-threshold families, use max(direct, curve) to avoid under-prediction
        if self.curve_predictor is not None and families is not None:
            # Curve predictor uses enriched features
            curve_thresholds = self.curve_predictor.predict_threshold(X_enriched)
            for i, family in enumerate(families):
                if family in HIGH_THRESHOLD_FAMILIES:
                    # For hard families, take the more conservative prediction
                    thresholds[i] = max(thresholds[i], curve_thresholds[i])

        # Step 3: Apply family floors if families provided
        if families is not None:
            thresholds = apply_family_floor(thresholds, families)

        # Step 4: Predict runtime using predicted thresholds (sequential prediction)
        times = self.runtime_model.predict(X_rt, thresholds)

        # Step 5: Blend with runtime curve predictor if available
        if self.rt_curve_predictor is not None:
            curve_times = self.rt_curve_predictor.predict_runtime_at_threshold(X_enriched, thresholds)
            # Geometric mean blending for runtime
            times = np.sqrt(times * curve_times)

        return thresholds, times

    def save(self, path: str | Path):
        """Save the combined predictor to a file."""
        path = Path(path)
        with open(path, "wb") as f:
            pickle.dump({
                "threshold_model": self.threshold_model,
                "runtime_model": self.runtime_model,
                "feature_columns": self.feature_columns,
                "enriched_feature_columns": self.enriched_feature_columns,
                "selected_feature_names": self.selected_feature_names,
                "selected_feature_names_threshold": self.selected_feature_names_threshold,
                "selected_feature_names_runtime": self.selected_feature_names_runtime,
                "feature_selector": self.feature_selector,
                "curve_predictor": self.curve_predictor,
                "rt_curve_predictor": self.rt_curve_predictor,
                "aux_predictor": self.aux_predictor,
            }, f)

    @classmethod
    def load(cls, path: str | Path) -> "CombinedPredictor":
        """Load a combined predictor from a file."""
        path = Path(path)
        with open(path, "rb") as f:
            data = pickle.load(f)

        predictor = cls(
            threshold_model=data["threshold_model"],
            runtime_model=data["runtime_model"],
        )
        predictor.feature_columns = data.get("feature_columns", [])
        predictor.enriched_feature_columns = data.get("enriched_feature_columns", [])
        predictor.selected_feature_names = data.get("selected_feature_names", [])
        predictor.selected_feature_names_threshold = data.get(
            "selected_feature_names_threshold",
            predictor.selected_feature_names,
        )
        predictor.selected_feature_names_runtime = data.get(
            "selected_feature_names_runtime",
            predictor.selected_feature_names,
        )
        predictor.feature_selector = data.get("feature_selector", None)
        predictor.curve_predictor = data.get("curve_predictor", None)
        predictor.rt_curve_predictor = data.get("rt_curve_predictor", None)
        predictor.aux_predictor = data.get("aux_predictor", None)
        return predictor


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


def get_feature_columns(df_columns: List[str]) -> List[str]:
    """
    Select feature columns suitable for model training.

    Excludes:
    - Target columns (threshold, runtime, fidelity)
    - Metadata columns (file, backend, precision, family)
    - Sweep columns (per-rung data)
    """
    exclude_patterns = [
        "file", "backend", "precision", "family", "predicted_family", "true_family",
        "selected_threshold", "selected_fidelity", "selected_threshold_log2",
        "forward_wall_s", "forward_threshold", "forward_unique_outcomes", "forward_peak_rss_mb",
        "estimated_per_shot_s", "estimated_setup_s",
        "state_setup_wall_s", "state_setup_peak_rss_mb",
        "log_forward_wall_s", "log_setup_s", "log_per_shot_s",
        "is_gpu", "is_double",
        "verify_p_return_zero",
        "sweep_min_fidelity", "sweep_max_fidelity", "n_sweep_rungs",
        "fidelity_at_1", "rungs_to_099", "biggest_fid_jump", "biggest_jump_rung",
        "sweep_fid_", "sweep_wall_", "sweep_rss_",
    ]

    feature_cols = []
    for col in df_columns:
        exclude = False
        for pattern in exclude_patterns:
            if pattern in col:
                exclude = True
                break
        if not exclude:
            feature_cols.append(col)

    return feature_cols
