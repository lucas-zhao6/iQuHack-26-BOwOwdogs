"""
Runtime-only predictor based on the old combined model architecture.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Optional
import warnings

import numpy as np

from src.threshold_models.lgbm.threshold_model import ThresholdModel, apply_family_floor, HIGH_THRESHOLD_FAMILIES
from src.runtime_models.lgbm.runtime_model import RuntimeModel


warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="X does not have valid feature names, but LGBMRegressor was fitted with feature names",
)


class CombinedRuntimePredictor:
    """
    End-to-end runtime predictor that internally uses a threshold model
    for sequential prediction, but exposes runtime outputs only.
    """

    def __init__(
        self,
        threshold_model: Optional[ThresholdModel] = None,
        runtime_model: Optional[RuntimeModel] = None,
    ) -> None:
        self.threshold_model = threshold_model or ThresholdModel()
        self.runtime_model = runtime_model or RuntimeModel()
        self.feature_columns: List[str] = []
        self.enriched_feature_columns: List[str] = []
        self.selected_feature_names_threshold: List[str] = []
        self.selected_feature_names_runtime: List[str] = []
        self.feature_selector = None
        self.curve_predictor = None
        self.rt_curve_predictor = None
        self.aux_predictor = None
        self.use_family_floor = True

    def predict(self, X: np.ndarray, families: Optional[List[str]] = None) -> np.ndarray:
        X_enriched = X
        if self.aux_predictor is not None:
            aux_features = self.aux_predictor.predict_as_features(X)
            if aux_features.shape[1] > 0:
                X_enriched = np.hstack([X, aux_features])

        if self.feature_selector is not None:
            if hasattr(self.feature_selector, "transform_threshold") and hasattr(
                self.feature_selector, "transform_runtime"
            ):
                X_thr = self.feature_selector.transform_threshold(X_enriched)
                X_rt = self.feature_selector.transform_runtime(X_enriched)
            else:
                X_thr = self.feature_selector.transform(X_enriched)
                X_rt = X_thr
        else:
            X_thr = X_enriched
            X_rt = X_enriched

        thresholds = self.threshold_model.predict(X_thr)

        if self.curve_predictor is not None and families is not None:
            curve_thresholds = self.curve_predictor.predict_threshold(X_enriched)
            for i, family in enumerate(families):
                if family in HIGH_THRESHOLD_FAMILIES:
                    thresholds[i] = max(thresholds[i], curve_thresholds[i])

        if families is not None and self.use_family_floor:
            thresholds = apply_family_floor(thresholds, families)

        if hasattr(self.threshold_model, "predict_proba"):
            proba = self.threshold_model.predict_proba(X_thr)
            idxs = np.arange(proba.shape[1])
            expected = proba @ idxs
            entropy = -(proba * np.log(proba + 1e-9)).sum(axis=1)
            extras = [expected, entropy]
        else:
            extras = []

        for flag_name in ["is_gpu", "is_double"]:
            if flag_name in self.enriched_feature_columns:
                flag_idx = self.enriched_feature_columns.index(flag_name)
                extras.append(X_enriched[:, flag_idx])

        if extras:
            extra_cols = [e.reshape(-1, 1) if e.ndim == 1 else e for e in extras]
            X_rt = np.hstack([X_rt] + extra_cols)

        times = self.runtime_model.predict(X_rt, thresholds)

        if self.rt_curve_predictor is not None:
            curve_times = self.rt_curve_predictor.predict_runtime_at_threshold(X_enriched, thresholds)
            times = np.sqrt(times * curve_times)

        return times

    def save(self, path: str | Path) -> None:
        path = Path(path)
        with path.open("wb") as f:
            pickle.dump(
                {
                    "threshold_model": self.threshold_model,
                    "runtime_model": self.runtime_model,
                    "feature_columns": self.feature_columns,
                    "enriched_feature_columns": self.enriched_feature_columns,
                    "selected_feature_names_threshold": self.selected_feature_names_threshold,
                    "selected_feature_names_runtime": self.selected_feature_names_runtime,
                    "feature_selector": self.feature_selector,
                    "curve_predictor": self.curve_predictor,
                    "rt_curve_predictor": self.rt_curve_predictor,
                    "aux_predictor": self.aux_predictor,
                    "use_family_floor": self.use_family_floor,
                },
                f,
            )

    @classmethod
    def load(cls, path: str | Path) -> "CombinedRuntimePredictor":
        path = Path(path)
        with path.open("rb") as f:
            data = pickle.load(f)

        predictor = cls(
            threshold_model=data["threshold_model"],
            runtime_model=data["runtime_model"],
        )
        predictor.feature_columns = data.get("feature_columns", [])
        predictor.enriched_feature_columns = data.get("enriched_feature_columns", [])
        predictor.selected_feature_names_threshold = data.get(
            "selected_feature_names_threshold",
            [],
        )
        predictor.selected_feature_names_runtime = data.get(
            "selected_feature_names_runtime",
            [],
        )
        predictor.feature_selector = data.get("feature_selector", None)
        predictor.curve_predictor = data.get("curve_predictor", None)
        predictor.rt_curve_predictor = data.get("rt_curve_predictor", None)
        predictor.aux_predictor = data.get("aux_predictor", None)
        predictor.use_family_floor = data.get("use_family_floor", True)
        return predictor
