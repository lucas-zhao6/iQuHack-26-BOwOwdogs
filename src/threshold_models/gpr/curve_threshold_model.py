"""
Curve-based threshold model using Gaussian Process Regression (GPR).

Predicts fidelity for each threshold rung and derives the minimum threshold
that achieves the target fidelity. Enforces monotonicity across rungs
post-hoc via cumulative max.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler

from src.evaluation.scoring import THRESHOLD_RUNGS


TARGET_FIDELITY = 0.75


class GPRCurveThresholdModel:
    """Predict fidelity at each rung using independent GPR models."""

    def __init__(
        self,
        kernel: Optional[object] = None,
        alpha: float = 1e-6,
        normalize_y: bool = True,
        n_restarts_optimizer: int = 1,
        max_train_size: Optional[int] = 2000,
        random_state: int = 42,
    ) -> None:
        if kernel is None:
            kernel = ConstantKernel(1.0, (1e-2, 1e3)) * RBF(1.0, (1e-2, 1e3)) + WhiteKernel(
                1e-3, (1e-10, 1e-1)
            )
        self.kernel = kernel
        self.alpha = alpha
        self.normalize_y = normalize_y
        self.n_restarts_optimizer = n_restarts_optimizer
        self.max_train_size = max_train_size
        self.random_state = random_state

        self.models: dict[int, GaussianProcessRegressor | None] = {}
        self.scalers: dict[int, StandardScaler | None] = {}
        self._fitted = False

    def _subsample_idx(self, n_samples: int) -> np.ndarray:
        if self.max_train_size is None or n_samples <= self.max_train_size:
            return np.arange(n_samples)
        rng = np.random.default_rng(self.random_state)
        return rng.choice(n_samples, size=self.max_train_size, replace=False)

    def fit(
        self,
        X: np.ndarray,
        y_curves: Dict[int, np.ndarray],
        rungs: Iterable[int] = THRESHOLD_RUNGS,
    ) -> None:
        X = np.asarray(X)
        rungs_list = list(rungs)
        base_idx = self._subsample_idx(X.shape[0])

        for rung in rungs_list:
            y = np.asarray(y_curves.get(rung, []), dtype=float)
            if y.size == 0:
                self.models[rung] = None
                self.scalers[rung] = None
                continue

            valid = np.isfinite(y)
            if valid.sum() == 0:
                self.models[rung] = None
                self.scalers[rung] = None
                continue

            use_idx = base_idx[valid[base_idx]]
            if use_idx.size == 0:
                self.models[rung] = None
                self.scalers[rung] = None
                continue

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X[use_idx])

            model = GaussianProcessRegressor(
                kernel=self.kernel,
                alpha=self.alpha,
                normalize_y=self.normalize_y,
                n_restarts_optimizer=self.n_restarts_optimizer,
                random_state=self.random_state,
            )
            model.fit(X_scaled, y[use_idx])
            self.models[rung] = model
            self.scalers[rung] = scaler

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
            scaler = self.scalers.get(rung)
            if model is None or scaler is None:
                if idx > 0:
                    preds[:, idx] = preds[:, idx - 1]
                continue
            X_scaled = scaler.transform(X)
            preds[:, idx] = model.predict(X_scaled)

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
