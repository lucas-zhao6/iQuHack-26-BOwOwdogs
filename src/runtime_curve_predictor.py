"""
Runtime Curve Prediction - Analogous to Innovation 5 for threshold.

Instead of predicting total runtime directly, we:
1. Predict runtime at each threshold rung (sweep_wall_1, sweep_wall_2, ..., sweep_wall_256)
2. Use the predicted threshold to look up the corresponding predicted runtime

This makes runtime prediction truly threshold-aware.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler


# Threshold rungs we predict runtime for
RUNTIME_RUNGS = [1, 2, 4, 8, 16, 32, 64, 128, 256]


class RuntimeCurvePredictor:
    """
    Predicts runtime at each threshold rung from circuit features.

    At inference time, uses the predicted threshold to look up
    the corresponding predicted runtime.
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
        self.models: Dict[int, GradientBoostingRegressor] = {}
        self._fitted = False

    def _create_model(self, random_state: int = 42) -> GradientBoostingRegressor:
        """Create a single GBR model."""
        return GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            min_samples_leaf=self.min_samples_leaf,
            random_state=random_state,
        )

    def fit(
        self,
        X: np.ndarray,
        runtime_curves: Dict[int, np.ndarray],
    ):
        """
        Fit the runtime curve predictor.

        Args:
            X: Feature matrix (n_samples, n_features)
            runtime_curves: Dict of rung -> runtime array (in seconds)
        """
        X_scaled = self.scaler.fit_transform(X)

        # Train separate model for each rung
        for i, rung in enumerate(RUNTIME_RUNGS):
            if rung not in runtime_curves:
                continue

            y = runtime_curves[rung]
            # Filter out NaN and invalid values
            valid = np.isfinite(y) & (y > 0)
            if valid.sum() < 10:
                continue

            # Work in log space for runtime (spans orders of magnitude)
            y_log = np.log1p(y[valid])

            model = self._create_model(random_state=42 + i)
            model.fit(X_scaled[valid], y_log)
            self.models[rung] = model

        self._fitted = True

    def predict_curve(self, X: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Predict runtime at each threshold rung.

        Returns:
            Dict of rung -> predicted runtime array (in seconds)
        """
        if not self._fitted:
            raise RuntimeError("RuntimeCurvePredictor not fitted")

        X_scaled = self.scaler.transform(X)
        predictions = {}

        for rung, model in self.models.items():
            pred_log = model.predict(X_scaled)
            pred = np.expm1(pred_log)  # Convert back from log space
            pred = np.clip(pred, 0.1, 10000)  # Sanity bounds
            predictions[rung] = pred

        return predictions

    def predict_runtime_at_threshold(
        self,
        X: np.ndarray,
        thresholds: np.ndarray,
    ) -> np.ndarray:
        """
        Predict runtime for given thresholds.

        For each sample, looks up the runtime prediction at the
        corresponding threshold rung.

        Args:
            X: Feature matrix
            thresholds: Predicted threshold values (1, 2, 4, ..., 256)

        Returns:
            Predicted runtimes
        """
        curves = self.predict_curve(X)
        n_samples = X.shape[0]
        runtimes = np.zeros(n_samples)

        for i in range(n_samples):
            thr = int(thresholds[i])
            # Find the closest rung
            if thr in curves:
                runtimes[i] = curves[thr][i]
            else:
                # Interpolate or use nearest
                available_rungs = sorted(curves.keys())
                if thr <= available_rungs[0]:
                    runtimes[i] = curves[available_rungs[0]][i]
                elif thr >= available_rungs[-1]:
                    runtimes[i] = curves[available_rungs[-1]][i]
                else:
                    # Find surrounding rungs and interpolate in log space
                    lower = max(r for r in available_rungs if r <= thr)
                    upper = min(r for r in available_rungs if r >= thr)
                    if lower == upper:
                        runtimes[i] = curves[lower][i]
                    else:
                        # Log-linear interpolation
                        t = (np.log2(thr) - np.log2(lower)) / (np.log2(upper) - np.log2(lower))
                        runtimes[i] = curves[lower][i] * (1 - t) + curves[upper][i] * t

        return runtimes


def extract_runtime_curves(df, rungs: List[int] = None) -> Dict[int, np.ndarray]:
    """
    Extract runtime curve targets from a dataframe.

    Args:
        df: DataFrame with sweep_wall_* columns
        rungs: Which rungs to extract (default: RUNTIME_RUNGS)

    Returns:
        Dict of rung -> runtime array
    """
    if rungs is None:
        rungs = RUNTIME_RUNGS

    curves = {}
    for rung in rungs:
        col = f"sweep_wall_{rung}"
        if col in df.columns:
            curves[rung] = df[col].values.astype(float)

    return curves


class EnhancedRuntimeModel:
    """
    Enhanced runtime model combining:
    1. Direct runtime prediction (existing approach)
    2. Runtime curve prediction (new approach)
    3. Weighted combination based on prediction confidence
    """

    def __init__(
        self,
        direct_weight: float = 0.5,
        curve_weight: float = 0.5,
        n_estimators: int = 100,
        max_depth: int = 4,
    ):
        self.direct_weight = direct_weight
        self.curve_weight = curve_weight

        self.curve_predictor = RuntimeCurvePredictor(
            n_estimators=n_estimators,
            max_depth=max_depth,
        )
        self.direct_scaler = StandardScaler()
        self.direct_model: Optional[GradientBoostingRegressor] = None
        self._fitted = False

    def fit(
        self,
        X: np.ndarray,
        y_runtime: np.ndarray,
        thresholds: np.ndarray,
        runtime_curves: Dict[int, np.ndarray],
    ):
        """
        Fit both direct and curve-based runtime predictors.

        Args:
            X: Feature matrix
            y_runtime: Total runtime targets
            thresholds: Threshold values (for adding as feature to direct model)
            runtime_curves: Dict of rung -> runtime array
        """
        # Fit curve predictor
        self.curve_predictor.fit(X, runtime_curves)

        # Fit direct predictor with threshold as feature
        thr_feature = np.log2(np.clip(thresholds, 1, 512)).reshape(-1, 1)
        X_with_thr = np.hstack([X, thr_feature])
        X_scaled = self.direct_scaler.fit_transform(X_with_thr)

        y_log = np.log1p(y_runtime)
        self.direct_model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=4,
            random_state=42,
        )
        self.direct_model.fit(X_scaled, y_log)

        self._fitted = True

    def predict(
        self,
        X: np.ndarray,
        thresholds: np.ndarray,
    ) -> np.ndarray:
        """
        Predict runtime using weighted combination of direct and curve predictions.
        """
        if not self._fitted:
            raise RuntimeError("EnhancedRuntimeModel not fitted")

        # Curve-based prediction
        curve_pred = self.curve_predictor.predict_runtime_at_threshold(X, thresholds)

        # Direct prediction
        thr_feature = np.log2(np.clip(thresholds, 1, 512)).reshape(-1, 1)
        X_with_thr = np.hstack([X, thr_feature])
        X_scaled = self.direct_scaler.transform(X_with_thr)
        direct_pred_log = self.direct_model.predict(X_scaled)
        direct_pred = np.expm1(direct_pred_log)

        # Weighted combination (in log space for stability)
        log_curve = np.log1p(np.clip(curve_pred, 0.1, 10000))
        log_direct = np.log1p(np.clip(direct_pred, 0.1, 10000))
        log_combined = self.curve_weight * log_curve + self.direct_weight * log_direct
        combined = np.expm1(log_combined)

        return np.clip(combined, 0.1, 10000)
