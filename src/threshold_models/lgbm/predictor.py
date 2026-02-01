"""
Threshold predictor wrapper for LGBM-based threshold modeling.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np

from .threshold_model import ThresholdModel, apply_family_floor


class ThresholdPredictor:
    """End-to-end threshold predictor with optional feature enrichment/selection."""

    def __init__(
        self,
        threshold_model: Optional[ThresholdModel] = None,
    ) -> None:
        self.threshold_model = threshold_model or ThresholdModel()
        self.feature_columns: List[str] = []
        self.enriched_feature_columns: List[str] = []
        self.selected_feature_names: List[str] = []
        self.feature_selector = None
        self.aux_predictor = None
        self.use_family_floor = True

    def predict(self, X: np.ndarray, families: Optional[List[str]] = None) -> np.ndarray:
        X_enriched = X
        if self.aux_predictor is not None:
            aux_features = self.aux_predictor.predict_as_features(X)
            if aux_features.shape[1] > 0:
                X_enriched = np.hstack([X, aux_features])

        if self.feature_selector is not None:
            X_thr = self.feature_selector.transform_threshold(X_enriched)
        else:
            X_thr = X_enriched

        thresholds = self.threshold_model.predict(X_thr)
        if families is not None and self.use_family_floor:
            thresholds = apply_family_floor(thresholds, families)
        return thresholds

    def save(self, path: str | Path) -> None:
        path = Path(path)
        with path.open("wb") as f:
            pickle.dump(
                {
                    "threshold_model": self.threshold_model,
                    "feature_columns": self.feature_columns,
                    "enriched_feature_columns": self.enriched_feature_columns,
                    "selected_feature_names": self.selected_feature_names,
                    "feature_selector": self.feature_selector,
                    "aux_predictor": self.aux_predictor,
                    "use_family_floor": self.use_family_floor,
                },
                f,
            )

    @classmethod
    def load(cls, path: str | Path) -> "ThresholdPredictor":
        path = Path(path)
        with path.open("rb") as f:
            data = pickle.load(f)
        predictor = cls(threshold_model=data["threshold_model"])
        predictor.feature_columns = data.get("feature_columns", [])
        predictor.enriched_feature_columns = data.get("enriched_feature_columns", [])
        predictor.selected_feature_names = data.get("selected_feature_names", [])
        predictor.feature_selector = data.get("feature_selector", None)
        predictor.aux_predictor = data.get("aux_predictor", None)
        predictor.use_family_floor = data.get("use_family_floor", True)
        return predictor
