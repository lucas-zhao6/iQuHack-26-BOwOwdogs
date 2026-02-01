"""
Curve-based threshold model using Explainable Boosting Machines (EBM).

Predicts fidelity for each threshold rung and derives the minimum threshold
that achieves the target fidelity.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np

try:
    from interpret.glassbox import ExplainableBoostingRegressor
except ImportError as exc:
    raise ImportError(
        "EBM requires 'interpret'. Install with 'pip install interpret'."
    ) from exc

from src.evaluation.scoring import THRESHOLD_RUNGS


TARGET_FIDELITY = 0.75


class EBMCurveThresholdModel:
    """Predict fidelity at each rung using independent EBM regressors."""

    def __init__(
        self,
        ebm_params: Optional[dict] = None,
        random_state: int = 42,
    ) -> None:
        params = {
            "random_state": random_state,
        }
        if ebm_params:
            params.update(ebm_params)

        self.params = params
        self.models: dict[int, ExplainableBoostingRegressor | None] = {}
        self._fitted = False

    def fit(
        self,
        X: np.ndarray,
        y_curves: Dict[int, np.ndarray],
        rungs: Iterable[int] = THRESHOLD_RUNGS,
    ) -> None:
        X = np.asarray(X)
        rungs_list = list(rungs)

        for rung in rungs_list:
            y = np.asarray(y_curves.get(rung, []), dtype=float)
            if y.size == 0:
                self.models[rung] = None
                continue

            valid = np.isfinite(y)
            if valid.sum() == 0:
                self.models[rung] = None
                continue

            model = ExplainableBoostingRegressor(**self.params)
            model.fit(X[valid], y[valid])
            self.models[rung] = model

        self._fitted = True

    @staticmethod
    def _enforce_monotone(preds: np.ndarray) -> np.ndarray:
        """Force non-decreasing fidelity across rungs for each sample."""
        return np.maximum.accumulate(preds, axis=1)

    def predict_fidelity_vector(
        self,
        X: np.ndarray,
        thresholds: Iterable[int] = THRESHOLD_RUNGS,
    ) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Model not fitted.")

        X = np.asarray(X)
        rungs = list(thresholds)
        n_samples = X.shape[0]
        preds = np.zeros((n_samples, len(rungs)), dtype=float)

        for idx, rung in enumerate(rungs):
            model = self.models.get(rung)
            if model is None:
                if idx > 0:
                    preds[:, idx] = preds[:, idx - 1]
                continue
            preds[:, idx] = model.predict(X)

        preds = np.clip(preds, 0.0, 1.0)
        preds = self._enforce_monotone(preds)
        return preds

    def predict_fidelity(
        self,
        X: np.ndarray,
        thresholds: Iterable[int] = THRESHOLD_RUNGS,
    ) -> Dict[int, np.ndarray]:
        preds = self.predict_fidelity_vector(X, thresholds=thresholds)
        rungs = list(thresholds)
        return {rung: preds[:, i] for i, rung in enumerate(rungs)}

    def predict_threshold(
        self,
        X: np.ndarray,
        target_fidelity: float = TARGET_FIDELITY,
        thresholds: Iterable[int] = THRESHOLD_RUNGS,
    ) -> np.ndarray:
        preds = self.predict_fidelity_vector(X, thresholds=thresholds)
        rungs = list(thresholds)
        selected = np.full(preds.shape[0], rungs[-1], dtype=int)

        for i in range(preds.shape[0]):
            for r_idx, rung in enumerate(rungs):
                if preds[i, r_idx] >= target_fidelity:
                    selected[i] = rung
                    break
        return selected
