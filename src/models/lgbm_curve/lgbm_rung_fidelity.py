"""
LightGBM curve-fit model that predicts fidelity given circuit features + threshold.

This reuses the base LightGBM regression architecture and trains on all available
(rung, fidelity) pairs by stacking rows across rungs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np

try:
    import lightgbm as lgb
except ImportError as exc:
    raise ImportError(
        "LightGBM is required for training. Install with 'pip install lightgbm'."
    ) from exc

from lightgbm import LGBMRegressor


DEFAULT_RUNGS = [1, 2, 4, 8, 16, 32, 64, 128, 256]


@dataclass
class LGBMCurveFidelityModel:
    """Predict fidelity given circuit features and threshold as input."""

    lgb_params: Optional[dict] = None
    model: Optional[LGBMRegressor] = None
    fitted: bool = False

    def _build_training_matrix(
        self,
        X: np.ndarray,
        y_curves: Dict[int, np.ndarray],
        rungs: Iterable[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Stack rows with an extra threshold feature for supervised training."""
        X_list = []
        y_list = []

        for rung in rungs:
            if rung not in y_curves:
                continue

            y = y_curves[rung]
            valid = np.isfinite(y)
            if valid.sum() == 0:
                continue

            thr_feature = np.full((valid.sum(), 1), float(rung), dtype=float)
            X_aug = np.hstack([X[valid], thr_feature])

            X_list.append(X_aug)
            y_list.append(y[valid].astype(float))

        if not X_list:
            raise ValueError("No valid fidelity targets found for the requested rungs")

        X_all = np.vstack(X_list)
        y_all = np.concatenate(y_list)
        return X_all, y_all

    def fit(
        self,
        X: np.ndarray,
        y_curves: Dict[int, np.ndarray],
        rungs: Iterable[int] = DEFAULT_RUNGS,
    ) -> None:
        """Train a LightGBM regressor with threshold as an input feature."""
        X_all, y_all = self._build_training_matrix(X, y_curves, rungs)

        params = {
            "objective": "regression_l1",
            "n_estimators": 1000,
            "learning_rate": 0.03,
            "max_depth": 5,
            "num_leaves": 31,
            "min_child_samples": 15,
            "min_gain_to_split": 0.01,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "lambda_l1": 1.0,
            "lambda_l2": 2.0,
            "extra_trees": True,
            "random_state": 42,
            "verbosity": -1,
        }
        if self.lgb_params:
            params.update(self.lgb_params)

        self.model = LGBMRegressor(**params)
        self.model.fit(X_all, y_all)
        self.fitted = True

    def predict(
        self,
        X: np.ndarray,
        thresholds: Iterable[int] = DEFAULT_RUNGS,
    ) -> Dict[int, np.ndarray]:
        """Predict fidelity for each threshold in thresholds.

        Returns a dict of threshold -> fidelity predictions.
        """
        if not self.fitted or self.model is None:
            raise RuntimeError("Model not fitted")

        preds: Dict[int, np.ndarray] = {}
        for thr in thresholds:
            thr_feature = np.full((X.shape[0], 1), float(thr), dtype=float)
            X_aug = np.hstack([X, thr_feature])
            pred = self.model.predict(X_aug)
            preds[thr] = np.clip(pred, 0.0, 1.0)

        return preds
