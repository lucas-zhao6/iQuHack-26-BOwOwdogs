"""
Curve-based threshold model using LightGBM.

Predicts fidelity given circuit features + threshold, then derives the
minimum threshold that achieves the target fidelity.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import numpy as np

try:
    import lightgbm as lgb
except ImportError as exc:
    raise ImportError(
        "LightGBM is required for training. Install with 'pip install lightgbm'."
    ) from exc

from lightgbm import LGBMRegressor

from src.evaluation.scoring import THRESHOLD_RUNGS


TARGET_FIDELITY = 0.75


class LGBMCurveThresholdModel:
    """Predict fidelity given circuit features and threshold as input."""

    def __init__(
        self,
        lgb_params: Optional[dict] = None,
        n_estimators: int = 1000,
        early_stopping_rounds: int = 50,
    ) -> None:
        self.n_estimators = n_estimators
        self.early_stopping_rounds = early_stopping_rounds

        params = {
            "objective": "regression_l1",
            "n_estimators": n_estimators,
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
        if lgb_params:
            params.update(lgb_params)

        self.params = params
        self.model = LGBMRegressor(**params)
        self._fitted = False

    def _build_training_matrix(
        self,
        X: np.ndarray,
        y_curves: Dict[int, np.ndarray],
        rungs: Iterable[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X)
        X_list = []
        y_list = []

        for rung in rungs:
            if rung not in y_curves:
                continue
            y = np.asarray(y_curves[rung])
            valid = np.isfinite(y)
            if valid.sum() == 0:
                continue

            thr_feature = np.full((valid.sum(), 1), float(rung), dtype=float)
            X_aug = np.hstack([X[valid], thr_feature])
            X_list.append(X_aug)
            y_list.append(y[valid].astype(float))

        if not X_list:
            raise ValueError("No valid fidelity targets found for the requested rungs.")

        X_all = np.vstack(X_list)
        y_all = np.concatenate(y_list)
        return X_all, y_all

    def fit(
        self,
        X: np.ndarray,
        y_curves: Dict[int, np.ndarray],
        rungs: Iterable[int] = THRESHOLD_RUNGS,
        eval_set: Optional[Tuple[np.ndarray, Dict[int, np.ndarray]]] = None,
    ) -> None:
        X_all, y_all = self._build_training_matrix(X, y_curves, rungs)
        X_all = np.asarray(X_all)

        eps = 1e-6
        y_train = np.log(np.clip(y_all, eps, 1.0))

        fit_kwargs = {}
        callbacks = []
        if eval_set is not None:
            X_val, y_val_curves = eval_set
            X_val_all, y_val_all = self._build_training_matrix(X_val, y_val_curves, rungs)
            X_val_all = np.asarray(X_val_all)
            y_val = np.log(np.clip(y_val_all, eps, 1.0))
            fit_kwargs["eval_set"] = [(X_val_all, y_val)]
            fit_kwargs["eval_metric"] = "l1"
            if self.early_stopping_rounds > 0:
                callbacks.append(
                    lgb.early_stopping(self.early_stopping_rounds, verbose=False)
                )
        if callbacks:
            fit_kwargs["callbacks"] = callbacks

        self.model.fit(np.asarray(X_all), np.asarray(y_train), **fit_kwargs)
        if hasattr(self.model, "feature_names_in_"):
            try:
                delattr(self.model, "feature_names_in_")
            except AttributeError:
                self.model.feature_names_in_ = None
        self._fitted = True

    def predict_fidelity(
        self,
        X: np.ndarray,
        thresholds: Iterable[int] = THRESHOLD_RUNGS,
    ) -> Dict[int, np.ndarray]:
        if not self._fitted:
            raise RuntimeError("Model not fitted.")

        X = np.asarray(X)
        preds: Dict[int, np.ndarray] = {}
        for thr in thresholds:
            thr_feature = np.full((X.shape[0], 1), float(thr), dtype=float)
            X_aug = np.hstack([X, thr_feature])
            pred_log = self.model.predict(X_aug)
            pred = np.exp(pred_log)
            preds[thr] = np.clip(pred, 0.0, 1.0)
        return preds

    def predict_threshold(
        self,
        X: np.ndarray,
        target_fidelity: float = TARGET_FIDELITY,
        thresholds: Iterable[int] = THRESHOLD_RUNGS,
    ) -> np.ndarray:
        preds = self.predict_fidelity(X, thresholds=thresholds)
        rungs = list(thresholds)
        n = X.shape[0]
        selected = np.full(n, rungs[-1], dtype=int)

        for i in range(n):
            for rung in rungs:
                if preds[rung][i] >= target_fidelity:
                    selected[i] = rung
                    break
        return selected
