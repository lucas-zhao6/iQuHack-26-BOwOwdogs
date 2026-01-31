"""
Auxiliary Feature Predictors

The training data contains derived sweep features that are only available
during training (computed from running the actual simulation):
- fidelity_at_1: Fidelity at threshold=1
- rungs_to_099: Number of rungs to reach 0.99 fidelity
- biggest_fid_jump: Largest fidelity jump in the sweep
- biggest_jump_rung: Rung where the biggest jump occurred
- verify_p_return_zero: Verification probability

These can't be used directly at test time, but we can:
1. Train models to PREDICT these from circuit features
2. Use predicted values as enriched features at test time

This provides additional signal about circuit difficulty.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler


# Auxiliary features we can predict
AUXILIARY_TARGETS = [
    "fidelity_at_1",      # Continuous [0, 1]
    "rungs_to_099",       # Discrete 0-9 (log2 scale)
    "biggest_fid_jump",   # Continuous [0, 1]
    "biggest_jump_rung",  # Discrete 1-256 (log2 scale)
]


class AuxiliaryFeaturePredictor:
    """
    Predicts auxiliary sweep-derived features from circuit features.

    These predicted features can be used to enrich the feature set
    at test time when actual sweep data is not available.
    """

    def __init__(
        self,
        n_estimators: int = 50,
        max_depth: int = 3,
        learning_rate: float = 0.1,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate

        self.scaler = StandardScaler()
        self.models: Dict[str, GradientBoostingRegressor] = {}
        self._fitted = False

    def fit(
        self,
        X: np.ndarray,
        auxiliary_targets: Dict[str, np.ndarray],
    ):
        """
        Fit predictors for each auxiliary target.

        Args:
            X: Feature matrix
            auxiliary_targets: Dict of target_name -> values
        """
        X_scaled = self.scaler.fit_transform(X)

        for target_name in AUXILIARY_TARGETS:
            if target_name not in auxiliary_targets:
                continue

            y = auxiliary_targets[target_name]
            valid = np.isfinite(y)
            if valid.sum() < 10:
                continue

            # Transform targets appropriately
            if target_name == "fidelity_at_1":
                # Logit transform for [0, 1] bounded
                y_transformed = np.clip(y[valid], 1e-6, 1 - 1e-6)
                y_transformed = np.log(y_transformed / (1 - y_transformed))
            elif target_name in ["rungs_to_099", "biggest_jump_rung"]:
                # Log2 transform for threshold-like values
                y_transformed = np.log2(np.clip(y[valid], 1, 512))
            else:
                # Direct for bounded continuous
                y_transformed = y[valid]

            model = GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=42,
            )
            model.fit(X_scaled[valid], y_transformed)
            self.models[target_name] = model

        self._fitted = True

    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Predict all auxiliary features.

        Returns:
            Dict of target_name -> predicted values
        """
        if not self._fitted:
            raise RuntimeError("AuxiliaryFeaturePredictor not fitted")

        X_scaled = self.scaler.transform(X)
        predictions = {}

        for target_name, model in self.models.items():
            pred = model.predict(X_scaled)

            # Inverse transform
            if target_name == "fidelity_at_1":
                # Inverse logit
                pred = 1 / (1 + np.exp(-pred))
            elif target_name in ["rungs_to_099", "biggest_jump_rung"]:
                # Inverse log2, round to valid thresholds
                pred = np.power(2, np.clip(pred, 0, 9))
                pred = np.round(pred)
            else:
                pred = np.clip(pred, 0, 1)

            predictions[target_name] = pred

        return predictions

    def predict_as_features(self, X: np.ndarray) -> np.ndarray:
        """
        Predict auxiliary features and return as feature matrix.

        Returns array of shape (n_samples, n_auxiliary_features).
        """
        preds = self.predict(X)

        # Stack in consistent order
        feature_list = []
        for target_name in AUXILIARY_TARGETS:
            if target_name in preds:
                feature_list.append(preds[target_name].reshape(-1, 1))

        if len(feature_list) == 0:
            return np.zeros((X.shape[0], 0))

        return np.hstack(feature_list)


def extract_auxiliary_targets(df) -> Dict[str, np.ndarray]:
    """
    Extract auxiliary targets from a dataframe.

    Args:
        df: DataFrame with auxiliary columns

    Returns:
        Dict of target_name -> values
    """
    targets = {}

    for col in AUXILIARY_TARGETS:
        if col in df.columns:
            targets[col] = df[col].values.astype(float)

    return targets


class EnrichedFeatureBuilder:
    """
    Builds enriched features by combining:
    1. Original circuit features
    2. Predicted auxiliary features (from sweep data patterns)
    3. Backend/precision features

    This maximizes information extraction from training data.
    """

    def __init__(self):
        self.aux_predictor = AuxiliaryFeaturePredictor()
        self.feature_names: List[str] = []
        self._fitted = False

    def fit(
        self,
        X: np.ndarray,
        auxiliary_targets: Dict[str, np.ndarray],
        feature_names: List[str],
    ):
        """
        Fit the auxiliary predictor.

        Args:
            X: Original feature matrix
            auxiliary_targets: Sweep-derived targets
            feature_names: Names of original features
        """
        self.aux_predictor.fit(X, auxiliary_targets)
        self.feature_names = feature_names.copy()

        # Add auxiliary feature names
        for target_name in AUXILIARY_TARGETS:
            if target_name in self.aux_predictor.models:
                self.feature_names.append(f"pred_{target_name}")

        self._fitted = True

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform features by adding predicted auxiliary features.

        Args:
            X: Original feature matrix

        Returns:
            Enriched feature matrix
        """
        if not self._fitted:
            raise RuntimeError("EnrichedFeatureBuilder not fitted")

        aux_features = self.aux_predictor.predict_as_features(X)

        if aux_features.shape[1] > 0:
            return np.hstack([X, aux_features])
        return X

    def fit_transform(
        self,
        X: np.ndarray,
        auxiliary_targets: Dict[str, np.ndarray],
        feature_names: List[str],
    ) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X, auxiliary_targets, feature_names)
        return self.transform(X)

    def get_feature_names(self) -> List[str]:
        """Return all feature names including auxiliary."""
        return self.feature_names.copy()
