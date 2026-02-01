"""
Model definitions for the Circuit Fingerprint Challenge.

Implements:
1. CombinedPredictor: End-to-end prediction pipeline
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .threshold_model import ThresholdModel, apply_family_floor, HIGH_THRESHOLD_FAMILIES
from .runtime_model import RuntimeModel


class CombinedPredictor:
    """
    End-to-end predictor combining threshold and runtime models.

    Provides a single interface for making predictions on new circuits.
    Supports feature selection and sequential prediction (threshold -> runtime).

    Innovation 5: Optional curve predictor for family-aware safety net.
    For high-threshold families, uses max(direct, curve) to avoid under-prediction.

    Additional enhancements:
    - Runtime curve predictor: predict runtime at each threshold rung
    - Auxiliary predictor: predict sweep-derived features for enrichment
    """

    def __init__(
        self,
        threshold_model: Optional[ThresholdModel] = None,
        runtime_model: Optional[RuntimeModel] = None,
    ):
        self.threshold_model = threshold_model or ThresholdModel()
        self.runtime_model = runtime_model or RuntimeModel()
        self.feature_columns: List[str] = []
        self.enriched_feature_columns: List[str] = []  # After auxiliary enrichment
        self.selected_feature_names: List[str] = []
        self.selected_feature_names_threshold: List[str] = []
        self.selected_feature_names_runtime: List[str] = []
        self.feature_selector = None  # Optional FeatureSelector
        self.curve_predictor = None  # Optional FidelityCurvePredictor (Innovation 5)
        self.rt_curve_predictor = None  # Optional RuntimeCurvePredictor
        self.aux_predictor = None  # Optional AuxiliaryFeaturePredictor

    def fit(
        self,
        X: np.ndarray,
        y_threshold: np.ndarray,
        y_total_time: np.ndarray,
        y_setup_time: Optional[np.ndarray] = None,
        y_per_shot_time: Optional[np.ndarray] = None,
        feature_columns: Optional[List[str]] = None,
    ):
        """Fit both models (legacy method, prefer training separately)."""
        self.threshold_model.fit(X, y_threshold)
        # For legacy fit, use true thresholds
        self.runtime_model.fit(X, y_total_time, y_threshold, y_setup_time, y_per_shot_time)

        if feature_columns is not None:
            self.feature_columns = feature_columns

    def predict(self, X: np.ndarray, families: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict threshold and runtime using sequential prediction.

        Steps:
        1. Enrich features with auxiliary predictions (if aux_predictor is set)
        2. Apply feature selection (if selector is set)
        3. Predict thresholds (direct model)
        4. Apply curve predictor safety net for high-threshold families (Innovation 5)
        5. Apply family floors (if families provided)
        6. Predict runtime using predicted thresholds
        7. Blend with runtime curve predictor (if available)

        Returns: (predicted_thresholds, predicted_times)
        """
        # Step 0: Enrich features with auxiliary predictions if available
        X_enriched = X
        if self.aux_predictor is not None:
            aux_features = self.aux_predictor.predict_as_features(X)
            if aux_features.shape[1] > 0:
                X_enriched = np.hstack([X, aux_features])

        # Apply feature selection if available
        if self.feature_selector is not None:
            if hasattr(self.feature_selector, "transform_threshold") and hasattr(self.feature_selector, "transform_runtime"):
                X_thr = self.feature_selector.transform_threshold(X_enriched)
                X_rt = self.feature_selector.transform_runtime(X_enriched)
            else:
                X_thr = self.feature_selector.transform(X_enriched)
                X_rt = X_thr
        else:
            X_thr = X_enriched
            X_rt = X_enriched

        # Step 1: Predict thresholds (direct model)
        thresholds = self.threshold_model.predict(X_thr)

        # Step 2: Apply curve predictor safety net (Innovation 5)
        # For high-threshold families, use max(direct, curve) to avoid under-prediction
        if self.curve_predictor is not None and families is not None:
            # Curve predictor uses enriched features
            curve_thresholds = self.curve_predictor.predict_threshold(X_enriched)
            for i, family in enumerate(families):
                if family in HIGH_THRESHOLD_FAMILIES:
                    # For hard families, take the more conservative prediction
                    thresholds[i] = max(thresholds[i], curve_thresholds[i])

        # Step 3: Apply family floors if families provided
        if families is not None:
            thresholds = apply_family_floor(thresholds, families)

        # Step 4: Predict runtime using predicted thresholds (sequential prediction)
        times = self.runtime_model.predict(X_rt, thresholds)

        # Step 5: Blend with runtime curve predictor if available
        if self.rt_curve_predictor is not None:
            curve_times = self.rt_curve_predictor.predict_runtime_at_threshold(X_enriched, thresholds)
            # Geometric mean blending for runtime
            times = np.sqrt(times * curve_times)

        return thresholds, times

    def save(self, path: str | Path):
        """Save the combined predictor to a file."""
        path = Path(path)
        with open(path, "wb") as f:
            pickle.dump({
                "threshold_model": self.threshold_model,
                "runtime_model": self.runtime_model,
                "feature_columns": self.feature_columns,
                "enriched_feature_columns": self.enriched_feature_columns,
                "selected_feature_names": self.selected_feature_names,
                "selected_feature_names_threshold": self.selected_feature_names_threshold,
                "selected_feature_names_runtime": self.selected_feature_names_runtime,
                "feature_selector": self.feature_selector,
                "curve_predictor": self.curve_predictor,
                "rt_curve_predictor": self.rt_curve_predictor,
                "aux_predictor": self.aux_predictor,
            }, f)

    @classmethod
    def load(cls, path: str | Path) -> "CombinedPredictor":
        """Load a combined predictor from a file."""
        path = Path(path)
        with open(path, "rb") as f:
            data = pickle.load(f)

        predictor = cls(
            threshold_model=data["threshold_model"],
            runtime_model=data["runtime_model"],
        )
        predictor.feature_columns = data.get("feature_columns", [])
        predictor.enriched_feature_columns = data.get("enriched_feature_columns", [])
        predictor.selected_feature_names = data.get("selected_feature_names", [])
        predictor.selected_feature_names_threshold = data.get(
            "selected_feature_names_threshold",
            predictor.selected_feature_names,
        )
        predictor.selected_feature_names_runtime = data.get(
            "selected_feature_names_runtime",
            predictor.selected_feature_names,
        )
        predictor.feature_selector = data.get("feature_selector", None)
        predictor.curve_predictor = data.get("curve_predictor", None)
        predictor.rt_curve_predictor = data.get("rt_curve_predictor", None)
        predictor.aux_predictor = data.get("aux_predictor", None)
        return predictor


def get_feature_columns(df_columns: List[str]) -> List[str]:
    """
    Select feature columns suitable for model training.

    Excludes:
    - Target columns (threshold, runtime, fidelity)
    - Metadata columns (file, backend, precision, family)
    - Sweep columns (per-rung data)
    """
    exclude_patterns = [
        "file", "backend", "precision", "family", "predicted_family", "true_family",
        "selected_threshold", "selected_fidelity", "selected_threshold_log2",
        "forward_wall_s", "forward_threshold", "forward_unique_outcomes", "forward_peak_rss_mb",
        "estimated_per_shot_s", "estimated_setup_s",
        "state_setup_wall_s", "state_setup_peak_rss_mb",
        "log_forward_wall_s", "log_setup_s", "log_per_shot_s",
        "is_gpu", "is_double",
        "verify_p_return_zero",
        "sweep_min_fidelity", "sweep_max_fidelity", "n_sweep_rungs",
        "fidelity_at_1", "rungs_to_099", "biggest_fid_jump", "biggest_jump_rung",
        "sweep_fid_", "sweep_wall_", "sweep_rss_",
    ]

    feature_cols = []
    for col in df_columns:
        exclude = False
        for pattern in exclude_patterns:
            if pattern in col:
                exclude = True
                break
        if not exclude:
            feature_cols.append(col)

    return feature_cols
