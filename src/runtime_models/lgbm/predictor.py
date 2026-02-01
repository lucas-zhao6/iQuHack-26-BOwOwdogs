"""
Runtime predictor wrapper for LGBM-based runtime modeling.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np

from .runtime_model import RuntimeModel


class RuntimePredictor:
    """End-to-end runtime predictor with optional feature enrichment/selection."""

    def __init__(
        self,
        runtime_model: Optional[RuntimeModel] = None,
    ) -> None:
        self.runtime_model = runtime_model or RuntimeModel()
        self.feature_columns: List[str] = []
        self.enriched_feature_columns: List[str] = []
        self.selected_feature_names: List[str] = []
        self.feature_selector = None
        self.aux_predictor = None

    def _append_runtime_extras(self, X_rt: np.ndarray, X_enriched: np.ndarray) -> np.ndarray:
        extras = []
        for flag_name in ["is_gpu", "is_double"]:
            if flag_name in self.enriched_feature_columns:
                flag_idx = self.enriched_feature_columns.index(flag_name)
                extras.append(X_enriched[:, flag_idx])
        if not extras:
            return X_rt
        extra_cols = [e.reshape(-1, 1) if e.ndim == 1 else e for e in extras]
        return np.hstack([X_rt] + extra_cols)

    def predict(self, X: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
        X_enriched = X
        if self.aux_predictor is not None:
            aux_features = self.aux_predictor.predict_as_features(X)
            if aux_features.shape[1] > 0:
                X_enriched = np.hstack([X, aux_features])

        if self.feature_selector is not None:
            X_rt = self.feature_selector.transform_runtime(X_enriched)
        else:
            X_rt = X_enriched

        X_rt = self._append_runtime_extras(X_rt, X_enriched)
        return self.runtime_model.predict(X_rt, thresholds)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        with path.open("wb") as f:
            pickle.dump(
                {
                    "runtime_model": self.runtime_model,
                    "feature_columns": self.feature_columns,
                    "enriched_feature_columns": self.enriched_feature_columns,
                    "selected_feature_names": self.selected_feature_names,
                    "feature_selector": self.feature_selector,
                    "aux_predictor": self.aux_predictor,
                },
                f,
            )

    @classmethod
    def load(cls, path: str | Path) -> "RuntimePredictor":
        path = Path(path)
        with path.open("rb") as f:
            data = pickle.load(f)
        predictor = cls(runtime_model=data["runtime_model"])
        predictor.feature_columns = data.get("feature_columns", [])
        predictor.enriched_feature_columns = data.get("enriched_feature_columns", [])
        predictor.selected_feature_names = data.get("selected_feature_names", [])
        predictor.feature_selector = data.get("feature_selector", None)
        predictor.aux_predictor = data.get("aux_predictor", None)
        return predictor
