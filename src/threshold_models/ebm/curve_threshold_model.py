"""
Curve-based threshold model using Explainable Boosting Machines (EBM).

Predicts fidelity given circuit features + threshold rung, then derives the
minimum threshold that achieves the target fidelity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import inspect
import pickle

import numpy as np

try:
    from interpret.glassbox import ExplainableBoostingRegressor
except ImportError as exc:
    raise ImportError(
        "InterpretML is required for EBM training. Install with 'pip install interpret'."
    ) from exc

from src.evaluation.scoring import THRESHOLD_RUNGS


TARGET_FIDELITY = 0.75


@dataclass
class EBMConfig:
    ebm_params: Optional[dict] = None
    use_monotone_constraints: bool = False
    enforce_monotone_projection: bool = True
    threshold_feature_name: str = "threshold_rung"
    binary_feature_names: Optional[Sequence[str]] = None
    random_state: int = 42


class EBMCurveThresholdModel:
    """Predict fidelity given circuit features and threshold rung as input."""

    def __init__(self, config: Optional[EBMConfig] = None) -> None:
        self.config = config or EBMConfig()
        self.model: Optional[ExplainableBoostingRegressor] = None
        self.feature_names_: List[str] = []
        self.binary_feature_names_: List[str] = list(self.config.binary_feature_names or [])
        self.threshold_feature_name_ = self.config.threshold_feature_name
        self.threshold_feature_index_: Optional[int] = None
        self.binary_feature_indices_: Dict[str, int] = {}
        self._fitted = False

    def _filter_supported_params(self, params: dict) -> dict:
        try:
            sig = inspect.signature(ExplainableBoostingRegressor)
        except (TypeError, ValueError):
            return params
        supported = set(sig.parameters.keys())
        return {k: v for k, v in params.items() if k in supported}

    def _build_base_params(self) -> dict:
        params = {
            "random_state": self.config.random_state,
            "interactions": 0,
            "learning_rate": 0.05,
            "max_bins": 64,
            "max_rounds": 200,
            "min_samples_leaf": 2,
            "outer_bags": 8,
            "inner_bags": 0,
            "n_jobs": 1,
        }
        if self.config.ebm_params:
            params.update(self.config.ebm_params)
        return self._filter_supported_params(params)

    def _resolve_feature_names(self, feature_names: Optional[Sequence[str]]) -> None:
        if feature_names is None:
            self.feature_names_ = []
            return
        self.feature_names_ = list(feature_names)
        self.threshold_feature_index_ = None
        if self.threshold_feature_name_ in self.feature_names_:
            self.threshold_feature_index_ = self.feature_names_.index(self.threshold_feature_name_)
        self.binary_feature_indices_ = {
            name: self.feature_names_.index(name)
            for name in self.binary_feature_names_
            if name in self.feature_names_
        }

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

            rung_idx = int(round(np.log2(float(rung))))
            thr_feature = np.full((valid.sum(), 1), float(rung_idx), dtype=float)
            X_aug = np.hstack([X[valid], thr_feature])
            X_list.append(X_aug)
            y_list.append(y[valid].astype(float))

        if not X_list:
            raise ValueError("No valid fidelity targets found for the requested rungs.")

        X_all = np.vstack(X_list)
        y_all = np.concatenate(y_list)
        return X_all, y_all

    def _apply_monotone_constraints(self, n_features: int) -> dict:
        if not self.config.use_monotone_constraints:
            return {}
        constraints = [0] * n_features
        constraints[-1] = 1
        return self._filter_supported_params({"monotone_constraints": constraints})

    def _apply_monotonize(self) -> None:
        if self.model is None or not hasattr(self.model, "monotonize"):
            return
        term_names = getattr(self.model, "term_names_", None)
        if not term_names:
            return

        term_index = None
        for idx, name in enumerate(term_names):
            if isinstance(name, str) and name == self.threshold_feature_name_:
                term_index = idx
                break
            if isinstance(name, (list, tuple)) and self.threshold_feature_name_ in name:
                term_index = idx
                break

        if term_index is None:
            return

        try:
            self.model.monotonize(term_index, increasing=True)
        except Exception:
            try:
                self.model.monotonize(self.threshold_feature_name_, increasing=True)
            except Exception:
                return

    @staticmethod
    def _is_non_decreasing(values: np.ndarray, tol: float = 1e-12) -> bool:
        return np.all(np.diff(values) >= -tol)

    @staticmethod
    def _pava_isotonic(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if values.size == 0:
            return values

        v = values.copy().tolist()
        w = [1] * len(v)
        i = 0
        while i < len(v) - 1:
            if v[i] <= v[i + 1] + 1e-12:
                i += 1
                continue
            total = v[i] * w[i] + v[i + 1] * w[i + 1]
            w_sum = w[i] + w[i + 1]
            v[i] = total / w_sum
            w[i] = w_sum
            del v[i + 1]
            del w[i + 1]
            if i > 0:
                i -= 1
        expanded = []
        for value, count in zip(v, w):
            expanded.extend([value] * count)
        return np.array(expanded, dtype=float)

    def _apply_monotone_projection(self, values: np.ndarray) -> np.ndarray:
        if self._is_non_decreasing(values):
            return values
        return self._pava_isotonic(values)

    def fit(
        self,
        X: np.ndarray,
        y_curves: Dict[int, np.ndarray],
        rungs: Iterable[int] = THRESHOLD_RUNGS,
        feature_names: Optional[Sequence[str]] = None,
    ) -> "EBMCurveThresholdModel":
        X_all, y_all = self._build_training_matrix(X, y_curves, rungs)
        X_all = np.asarray(X_all)

        params = self._build_base_params()
        if feature_names is not None:
            name_params = self._filter_supported_params(
                {"feature_names": list(feature_names) + [self.threshold_feature_name_]}
            )
            params.update(name_params)
        params.update(self._apply_monotone_constraints(X_all.shape[1]))
        self.model = ExplainableBoostingRegressor(**params)
        self.model.fit(X_all, y_all)

        if feature_names is not None:
            self._resolve_feature_names(list(feature_names) + [self.threshold_feature_name_])
        else:
            self.feature_names_ = []

        self._apply_monotonize()
        self._fitted = True
        return self

    def predict(self, X_with_rung: np.ndarray) -> np.ndarray:
        if not self._fitted or self.model is None:
            raise RuntimeError("Model not fitted.")
        return np.asarray(self.model.predict(np.asarray(X_with_rung)))

    def predict_fidelity(
        self,
        X: np.ndarray,
        thresholds: Iterable[int] = THRESHOLD_RUNGS,
    ) -> Dict[int, np.ndarray]:
        if not self._fitted or self.model is None:
            raise RuntimeError("Model not fitted.")

        X = np.asarray(X)
        preds: Dict[int, np.ndarray] = {}
        rungs = list(thresholds)
        rung_indices = [int(round(np.log2(float(r)))) for r in rungs]

        for rung, rung_idx in zip(rungs, rung_indices):
            thr_feature = np.full((X.shape[0], 1), float(rung_idx), dtype=float)
            X_aug = np.hstack([X, thr_feature])
            pred = self.model.predict(X_aug)
            preds[rung] = np.clip(pred, 0.0, 1.0)

        if self.config.enforce_monotone_projection and preds:
            stacked = np.vstack([preds[r] for r in rungs]).T
            for i in range(stacked.shape[0]):
                stacked[i] = self._apply_monotone_projection(stacked[i])
            for idx, rung in enumerate(rungs):
                preds[rung] = stacked[:, idx]

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

    def predict_curve(
        self,
        circuit_features: np.ndarray,
        b1_value: float,
        b2_value: float,
        thresholds: Iterable[int] = THRESHOLD_RUNGS,
    ) -> np.ndarray:
        if not self._fitted or self.model is None:
            raise RuntimeError("Model not fitted.")

        base = np.asarray(circuit_features, dtype=float).reshape(1, -1)
        if self.feature_names_:
            expected = len(self.feature_names_) - 1
            if base.shape[1] != expected:
                raise ValueError(
                    f"Expected {expected} base features, got {base.shape[1]}."
                )
        if self.feature_names_:
            if self.binary_feature_names_:
                for name, value in zip(self.binary_feature_names_, [b1_value, b2_value]):
                    if name in self.binary_feature_indices_:
                        base[0, self.binary_feature_indices_[name]] = float(value)

        rungs = list(thresholds)
        rung_indices = np.array([int(round(np.log2(float(r)))) for r in rungs], dtype=float)
        X_rep = np.repeat(base, len(rungs), axis=0)
        X_aug = np.hstack([X_rep, rung_indices.reshape(-1, 1)])

        preds = np.clip(self.model.predict(X_aug), 0.0, 1.0)
        if self.config.enforce_monotone_projection:
            preds = self._apply_monotone_projection(preds)
        return preds

    def save(self, path: str) -> None:
        payload = {
            "model": self.model,
            "feature_names": self.feature_names_,
            "binary_feature_names": self.binary_feature_names_,
            "threshold_feature_name": self.threshold_feature_name_,
            "config": self.config,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: str) -> "EBMCurveThresholdModel":
        with open(path, "rb") as f:
            payload = pickle.load(f)

        model = cls(config=payload.get("config", EBMConfig()))
        model.model = payload["model"]
        model.feature_names_ = payload.get("feature_names", [])
        model.binary_feature_names_ = payload.get("binary_feature_names", [])
        model.threshold_feature_name_ = payload.get("threshold_feature_name", "threshold_rung")
        model._resolve_feature_names(model.feature_names_)
        model._fitted = True
        return model
