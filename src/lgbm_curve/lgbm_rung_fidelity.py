"""
LightGBM model that takes rung as an input feature and predicts fidelity.

This follows the single-model approach: stack (X, log2(rung)) rows for all
rungs with available fidelity targets, then train a single regressor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np
from lightgbm import LGBMRegressor


DEFAULT_RUNGS = [1, 2, 4, 8, 16, 32, 64, 128, 256]


@dataclass
class LGBMRungFidelityModel:
    """Predict fidelity given circuit features and rung as input."""

    lgb_params: Optional[dict] = None
    model: Optional[LGBMRegressor] = None
    fitted: bool = False

    def _build_training_matrix(
        self,
        X: np.ndarray,
        y_curves: Dict[int, np.ndarray],
        rungs: Iterable[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Stack rows with an extra log2(rung) feature for supervised training."""
        X_list = []
        y_list = []

        for rung in rungs:
            if rung not in y_curves:
                continue

            y = y_curves[rung]
            valid = np.isfinite(y)
            if valid.sum() == 0:
                continue

            rung_feature = np.full((valid.sum(), 1), np.log2(rung), dtype=float)
            X_aug = np.hstack([X[valid], rung_feature])

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
        """Train a single LightGBM regressor with rung as an input feature."""
        X_all, y_all = self._build_training_matrix(X, y_curves, rungs)

        params = {
            "n_estimators": 400,
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
        }
        if self.lgb_params:
            params.update(self.lgb_params)

        self.model = LGBMRegressor(**params)
        self.model.fit(X_all, y_all)
        self.fitted = True

    def predict(
        self,
        X: np.ndarray,
        rungs: Iterable[int] = DEFAULT_RUNGS,
    ) -> Dict[int, np.ndarray]:
        """Predict fidelity for each rung in rungs.

        Returns a dict of rung -> fidelity predictions.
        """
        if not self.fitted or self.model is None:
            raise RuntimeError("Model not fitted")

        preds: Dict[int, np.ndarray] = {}
        for rung in rungs:
            rung_feature = np.full((X.shape[0], 1), np.log2(rung), dtype=float)
            X_aug = np.hstack([X, rung_feature])
            pred = self.model.predict(X_aug)
            preds[rung] = np.clip(pred, 0.0, 1.0)

        return preds
