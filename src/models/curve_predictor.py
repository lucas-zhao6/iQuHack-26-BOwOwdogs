"""
Innovation 5: Fidelity Curve Prediction

Instead of directly predicting the minimum threshold, this model predicts
the entire fidelity-threshold curve, then derives the threshold from it.

Advantages:
1. Richer supervision (9 fidelity values vs 1 threshold label)
2. More interpretable (can visualize predicted vs actual curve)
3. Natural uncertainty: if predicted curve is "borderline", be conservative

Architecture:
- Multi-output GradientBoostingRegressor (one per threshold rung)
- Or a single model with threshold as input feature
"""

from __future__ import annotations

from typing import List, Dict, Tuple, Optional
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import warnings


# Threshold rungs we predict (excluding 512 as it's rarely needed)
PREDICTION_RUNGS = [1, 2, 4, 8, 16, 32, 64, 128, 256]


class FidelityCurvePredictor:
    """
    Predicts fidelity at each threshold rung from circuit features.

    Two modes:
    1. 'multi_output': Separate model for each rung (more accurate)
    2. 'single_model': One model with threshold as input (faster, shares learning)
    """

    def __init__(
        self,
        mode: str = "multi_output",
        n_estimators: int = 100,
        max_depth: int = 4,
        learning_rate: float = 0.1,
        min_samples_leaf: int = 3,
        safety_margin: float = 0.01,  # Predict threshold where fidelity >= 0.99 - margin
    ):
        self.mode = mode
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_samples_leaf = min_samples_leaf
        self.safety_margin = safety_margin

        self.scaler = StandardScaler()
        self.models: Dict[int, GradientBoostingRegressor] = {}
        self.single_model: Optional[GradientBoostingRegressor] = None
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
        y_curves: Dict[int, np.ndarray],
    ):
        """
        Fit the curve predictor.

        Args:
            X: Feature matrix (n_samples, n_features)
            y_curves: Dict of rung -> fidelity array, e.g. {1: [0.1, 0.2, ...], 2: [...], ...}
        """
        X_scaled = self.scaler.fit_transform(X)

        if self.mode == "multi_output":
            # Train separate model for each rung
            for i, rung in enumerate(PREDICTION_RUNGS):
                if rung not in y_curves:
                    continue

                y = y_curves[rung]
                # Filter out NaN values
                valid = np.isfinite(y)
                if valid.sum() < 10:
                    continue

                model = self._create_model(random_state=42 + i)
                model.fit(X_scaled[valid], y[valid])
                self.models[rung] = model

        elif self.mode == "single_model":
            # Create augmented dataset with threshold as feature
            X_aug_list = []
            y_list = []

            for rung in PREDICTION_RUNGS:
                if rung not in y_curves:
                    continue

                y = y_curves[rung]
                valid = np.isfinite(y)

                # Add log2(rung) as additional feature
                rung_feature = np.full((valid.sum(),), np.log2(rung))
                X_with_rung = np.column_stack([X_scaled[valid], rung_feature])

                X_aug_list.append(X_with_rung)
                y_list.append(y[valid])

            if len(X_aug_list) > 0:
                X_aug = np.vstack(X_aug_list)
                y_all = np.concatenate(y_list)

                self.single_model = self._create_model()
                self.single_model.fit(X_aug, y_all)

        self._fitted = True

    def predict_curve(self, X: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Predict fidelity at each threshold rung.

        Returns:
            Dict of rung -> predicted fidelity array
        """
        if not self._fitted:
            raise RuntimeError("FidelityCurvePredictor not fitted")

        X_scaled = self.scaler.transform(X)
        predictions = {}

        if self.mode == "multi_output":
            for rung, model in self.models.items():
                pred = model.predict(X_scaled)
                # Clip to valid fidelity range
                pred = np.clip(pred, 0.0, 1.0)
                predictions[rung] = pred

        elif self.mode == "single_model" and self.single_model is not None:
            for rung in PREDICTION_RUNGS:
                rung_feature = np.full((X_scaled.shape[0],), np.log2(rung))
                X_with_rung = np.column_stack([X_scaled, rung_feature])
                pred = self.single_model.predict(X_with_rung)
                pred = np.clip(pred, 0.0, 1.0)
                predictions[rung] = pred

        return predictions

    def predict_threshold(self, X: np.ndarray, target_fidelity: float = 0.99) -> np.ndarray:
        """
        Predict the minimum threshold needed to achieve target fidelity.

        Uses the predicted fidelity curve to find where fidelity crosses the target.
        """
        curves = self.predict_curve(X)
        n_samples = X.shape[0]

        # Adjust target with safety margin (be conservative)
        adjusted_target = target_fidelity - self.safety_margin

        thresholds = np.full(n_samples, 256)  # Default to max if never reaches target

        for i in range(n_samples):
            for rung in PREDICTION_RUNGS:
                if rung not in curves:
                    continue
                if curves[rung][i] >= adjusted_target:
                    thresholds[i] = rung
                    break

        return thresholds.astype(int)

    def predict_threshold_with_confidence(
        self, X: np.ndarray, target_fidelity: float = 0.99
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict threshold with confidence estimate.

        Confidence is based on how far above the target the predicted fidelity is.
        Low confidence suggests being more conservative (higher threshold).

        Returns:
            thresholds: Predicted threshold values
            confidences: Confidence scores (0-1)
        """
        curves = self.predict_curve(X)
        n_samples = X.shape[0]

        thresholds = np.full(n_samples, 256)
        confidences = np.zeros(n_samples)

        for i in range(n_samples):
            for rung in PREDICTION_RUNGS:
                if rung not in curves:
                    continue

                pred_fid = curves[rung][i]
                if pred_fid >= target_fidelity:
                    thresholds[i] = rung
                    # Confidence: how much above target?
                    confidences[i] = min(1.0, (pred_fid - target_fidelity) / 0.01)
                    break

        return thresholds.astype(int), confidences


def extract_curve_targets(df, rungs: List[int] = None) -> Dict[int, np.ndarray]:
    """
    Extract fidelity curve targets from a dataframe.

    Args:
        df: DataFrame with sweep_fid_* columns
        rungs: Which rungs to extract (default: PREDICTION_RUNGS)

    Returns:
        Dict of rung -> fidelity array
    """
    if rungs is None:
        rungs = PREDICTION_RUNGS

    curves = {}
    for rung in rungs:
        col = f"sweep_fid_{rung}"
        if col in df.columns:
            curves[rung] = df[col].values.astype(float)

    return curves


class CurveAwareThresholdModel:
    """
    Combines curve prediction with direct threshold prediction.

    Uses curve prediction as additional signal, with fallback to direct prediction
    when curve prediction is uncertain.
    """

    def __init__(
        self,
        curve_weight: float = 0.5,
        direct_weight: float = 0.5,
        **kwargs
    ):
        self.curve_weight = curve_weight
        self.direct_weight = direct_weight

        self.curve_predictor = FidelityCurvePredictor(**kwargs)
        self.direct_model: Optional[GradientBoostingRegressor] = None
        self.scaler = StandardScaler()
        self._fitted = False

    def fit(
        self,
        X: np.ndarray,
        y_threshold: np.ndarray,
        y_curves: Dict[int, np.ndarray],
    ):
        """Fit both curve and direct models."""
        # Fit curve predictor
        self.curve_predictor.fit(X, y_curves)

        # Fit direct threshold predictor
        X_scaled = self.scaler.fit_transform(X)
        y_idx = np.log2(np.clip(y_threshold, 1, 512))  # Convert to log2 scale

        self.direct_model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
        )
        self.direct_model.fit(X_scaled, y_idx)

        self._fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict threshold using weighted combination of curve and direct prediction.
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted")

        # Curve-based prediction
        curve_thresholds = self.curve_predictor.predict_threshold(X)
        curve_idx = np.log2(np.clip(curve_thresholds, 1, 512))

        # Direct prediction
        X_scaled = self.scaler.transform(X)
        direct_idx = self.direct_model.predict(X_scaled)

        # Weighted combination
        combined_idx = self.curve_weight * curve_idx + self.direct_weight * direct_idx

        # Round to nearest valid threshold
        combined_idx = np.clip(np.round(combined_idx), 0, 8).astype(int)

        # Convert back to threshold values
        thresholds = np.array([2**i for i in combined_idx])

        return thresholds
