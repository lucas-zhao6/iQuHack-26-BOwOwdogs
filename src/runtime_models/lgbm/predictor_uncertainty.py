"""
Runtime predictor that adds threshold-uncertainty features.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .runtime_model import RuntimeModel


class RuntimePredictorWithThresholdUncertainty:
    """Runtime predictor that augments features with threshold uncertainty."""

    def __init__(
        self,
        runtime_model: Optional[RuntimeModel] = None,
        threshold_predictor=None,
    ) -> None:
        self.runtime_model = runtime_model or RuntimeModel()
        self.threshold_predictor = threshold_predictor
        self.feature_columns: List[str] = []
        self.enriched_feature_columns: List[str] = []
        self.selected_feature_names: List[str] = []
        self.feature_selector = None
        self.aux_predictor = None

    def _compute_threshold_uncertainty(self, X_base: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.threshold_predictor is None:
            raise RuntimeError("Threshold predictor is required for uncertainty features.")

        X_thr = X_base
        if getattr(self.threshold_predictor, "aux_predictor", None) is not None:
            aux_features = self.threshold_predictor.aux_predictor.predict_as_features(X_base)
            if aux_features.shape[1] > 0:
                X_thr = np.hstack([X_base, aux_features])

        if getattr(self.threshold_predictor, "feature_selector", None) is not None:
            X_thr = self.threshold_predictor.feature_selector.transform_threshold(X_thr)

        proba = self.threshold_predictor.threshold_model.predict_proba(X_thr)
        idxs = np.arange(proba.shape[1])
        expected = proba @ idxs
        entropy = -(proba * np.log(proba + 1e-9)).sum(axis=1)
        return expected, entropy

    def _append_runtime_extras(
        self,
        X_rt: np.ndarray,
        X_enriched: np.ndarray,
        X_base: np.ndarray,
    ) -> np.ndarray:
        extras = []

        expected, entropy = self._compute_threshold_uncertainty(X_base)
        extras.extend([expected, entropy])

        for flag_name in ["is_gpu", "is_double"]:
            if flag_name in self.enriched_feature_columns:
                flag_idx = self.enriched_feature_columns.index(flag_name)
                extras.append(X_enriched[:, flag_idx])

        extra_cols = [e.reshape(-1, 1) if e.ndim == 1 else e for e in extras]
        return np.hstack([X_rt] + extra_cols)

    def predict(self, X: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
        X_base = X
        X_enriched = X
        if self.aux_predictor is not None:
            aux_features = self.aux_predictor.predict_as_features(X)
            if aux_features.shape[1] > 0:
                X_enriched = np.hstack([X, aux_features])

        if self.feature_selector is not None:
            X_rt = self.feature_selector.transform_runtime(X_enriched)
        else:
            X_rt = X_enriched

        X_rt = self._append_runtime_extras(X_rt, X_enriched, X_base)
        return self.runtime_model.predict(X_rt, thresholds)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        with path.open("wb") as f:
            pickle.dump(
                {
                    "runtime_model": self.runtime_model,
                    "threshold_predictor": self.threshold_predictor,
                    "feature_columns": self.feature_columns,
                    "enriched_feature_columns": self.enriched_feature_columns,
                    "selected_feature_names": self.selected_feature_names,
                    "feature_selector": self.feature_selector,
                    "aux_predictor": self.aux_predictor,
                },
                f,
            )

    @classmethod
    def load(cls, path: str | Path) -> "RuntimePredictorWithThresholdUncertainty":
        path = Path(path)
        with path.open("rb") as f:
            data = pickle.load(f)
        predictor = cls(
            runtime_model=data["runtime_model"],
            threshold_predictor=data.get("threshold_predictor", None),
        )
        predictor.feature_columns = data.get("feature_columns", [])
        predictor.enriched_feature_columns = data.get("enriched_feature_columns", [])
        predictor.selected_feature_names = data.get("selected_feature_names", [])
        predictor.feature_selector = data.get("feature_selector", None)
        predictor.aux_predictor = data.get("aux_predictor", None)
        return predictor
