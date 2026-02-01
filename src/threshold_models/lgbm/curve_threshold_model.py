"""
Curve-based threshold model using LightGBM.

Predicts fidelity given circuit features + threshold, then derives the
minimum threshold that achieves the target fidelity.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple
import warnings

import numpy as np

try:
    import lightgbm as lgb
except ImportError as exc:
    raise ImportError(
        "LightGBM is required for training. Install with 'pip install lightgbm'."
    ) from exc

from src.evaluation.scoring import THRESHOLD_RUNGS


TARGET_FIDELITY = 0.75

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="X does not have valid feature names, but LGBMRegressor was fitted with feature names",
)


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
            "objective": "quantile",
            "alpha": 1.0 / 6.0,
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
        self.model = None
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
            thr_feature = np.log2(np.clip(thr_feature, 1, 512))
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
        self._apply_monotone_constraints(X_all)

        eps = 1e-6
        y_train = np.clip(y_all, eps, 1.0)
        weights = self._build_sample_weights(X_all, y_all, rungs)

        fit_kwargs = {"sample_weight": weights}
        callbacks = []
        if eval_set is not None:
            X_val, y_val_curves = eval_set
            X_val_all, y_val_all = self._build_training_matrix(X_val, y_val_curves, rungs)
            X_val_all = np.asarray(X_val_all)
            y_val = np.clip(y_val_all, eps, 1.0)
            fit_kwargs["eval_set"] = [(X_val_all, y_val)]
            if self.early_stopping_rounds > 0:
                callbacks.append(
                    lgb.early_stopping(self.early_stopping_rounds, verbose=False)
                )
        if callbacks:
            fit_kwargs["callbacks"] = callbacks

        self.params["metric"] = "l1"
        train_data = lgb.Dataset(
            np.asarray(X_all),
            label=np.asarray(y_train),
            weight=fit_kwargs.pop("sample_weight", None),
            free_raw_data=False,
        )
        valid_sets = []
        if "eval_set" in fit_kwargs:
            X_val_all, y_val = fit_kwargs["eval_set"][0]
            valid_sets = [
                lgb.Dataset(
                    np.asarray(X_val_all),
                    label=np.asarray(y_val),
                    free_raw_data=False,
                )
            ]
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=self.n_estimators,
            valid_sets=valid_sets if valid_sets else None,
            callbacks=callbacks if callbacks else None,
        )
        self._fitted = True

    def _apply_monotone_constraints(self, X_with_thr: np.ndarray) -> None:
        """Enforce non-decreasing fidelity with threshold feature."""
        if self.params.get("objective") == "quantile":
            return
        n_features = X_with_thr.shape[1]
        constraints = [0] * n_features
        constraints[-1] = 1
        self.params["monotone_constraints"] = constraints

    def _build_sample_weights(
        self,
        X_with_thr: np.ndarray,
        y_all: np.ndarray,
        rungs: Iterable[int],
    ) -> np.ndarray:
        """Emphasize samples near the target fidelity and crossing rung."""
        rungs_list = list(rungs)
        n_samples = X_with_thr.shape[0]
        n_rungs = len(rungs_list)
        if n_rungs == 0:
            return np.ones(n_samples, dtype=float)

        total = n_samples
        if total % n_rungs != 0:
            return np.ones(n_samples, dtype=float)

        n_base = total // n_rungs
        y_matrix = y_all.reshape(n_rungs, n_base)

        idx_cross = np.full(n_base, n_rungs - 1, dtype=int)
        for i in range(n_base):
            for r_idx in range(n_rungs):
                if np.isfinite(y_matrix[r_idx, i]) and y_matrix[r_idx, i] >= TARGET_FIDELITY:
                    idx_cross[i] = r_idx
                    break

        weights = np.zeros_like(y_matrix, dtype=float)
        for r_idx, rung in enumerate(rungs_list):
            thr_weight = 1.0 / (1.0 + abs(np.log2(rung / rungs_list[0])))
            fidelity_weight = 1.0 + 2.0 * np.exp(
                -((y_matrix[r_idx] - TARGET_FIDELITY) ** 2) / (2 * 0.08**2)
            )
            near_cross = np.exp(-((r_idx - idx_cross) ** 2) / (2 * 1.0**2))
            weights[r_idx] = thr_weight * fidelity_weight * (1.0 + 2.0 * near_cross)

        return weights.reshape(-1)

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
            thr_feature = np.log2(np.clip(thr_feature, 1, 512))
            X_aug = np.hstack([X, thr_feature])
            pred = self.model.predict(X_aug)
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
