"""
GPR runtime model for forward run wall time.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler, RobustScaler


class GPRRuntimeModel:
    """Predicts forward run wall time using GPR on log1p(runtime)."""

    def __init__(
        self,
        kernel: Optional[object] = None,
        alpha: float = 1e-6,
        normalize_y: bool = True,
        n_restarts_optimizer: int = 1,
        max_train_size: Optional[int] = 2000,
        scaler_type: str = "standard",
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
        self.scaler_type = scaler_type
        self.random_state = random_state

        self.model: Optional[GaussianProcessRegressor] = None
        self.scaler: Optional[StandardScaler] = None
        self._fitted = False

    def _subsample_idx(self, n_samples: int) -> np.ndarray:
        if self.max_train_size is None or n_samples <= self.max_train_size:
            return np.arange(n_samples)
        rng = np.random.default_rng(self.random_state)
        return rng.choice(n_samples, size=self.max_train_size, replace=False)

    @staticmethod
    def _add_threshold_feature(X: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        thresholds = np.asarray(thresholds)
        if thresholds.ndim == 1:
            thr_feature = thresholds.reshape(-1, 1)
        else:
            thr_feature = thresholds
        return np.hstack([X, thr_feature])

    def fit(
        self,
        X: np.ndarray,
        thresholds: np.ndarray,
        y_total_time: np.ndarray,
    ) -> None:
        X = np.asarray(X)
        thresholds = np.asarray(thresholds)
        y_total_time = np.asarray(y_total_time)

        valid = np.isfinite(y_total_time) & (y_total_time > 0)
        X = X[valid]
        thresholds = thresholds[valid]
        y_total_time = y_total_time[valid]

        if X.shape[0] == 0:
            raise ValueError("No valid runtime targets for training.")

        X_with_thr = self._add_threshold_feature(X, thresholds)

        idx = self._subsample_idx(X_with_thr.shape[0])
        X_use = X_with_thr[idx]
        y_use = y_total_time[idx]

        if self.scaler_type == "standard":
            self.scaler = StandardScaler()
        elif self.scaler_type == "robust":
            self.scaler = RobustScaler()
        else:
            self.scaler = None

        X_scaled = self.scaler.fit_transform(X_use) if self.scaler is not None else X_use

        y_log = np.log1p(np.clip(y_use, 1e-9, None))
        self.model = GaussianProcessRegressor(
            kernel=self.kernel,
            alpha=self.alpha,
            normalize_y=self.normalize_y,
            n_restarts_optimizer=self.n_restarts_optimizer,
            random_state=self.random_state,
        )
        self.model.fit(X_scaled, y_log)
        self._fitted = True

    def predict(self, X: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
        if not self._fitted or self.model is None or self.scaler is None:
            raise RuntimeError("Model not fitted.")

        X = np.asarray(X)
        thresholds = np.asarray(thresholds)
        X_with_thr = self._add_threshold_feature(X, thresholds)
        X_scaled = self.scaler.transform(X_with_thr) if self.scaler is not None else X_with_thr
        pred_log = self.model.predict(X_scaled)
        pred = np.expm1(pred_log)
        return np.clip(pred, 1e-9, np.inf)
