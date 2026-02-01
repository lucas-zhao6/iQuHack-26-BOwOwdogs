"""
Naive bucketed baseline for runtime prediction.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from src.naive_bucket_base import NaiveBucketModel, NaiveBucketConfig


class NaiveBucketRuntimeModel:
    """Wrapper that returns only runtime predictions."""

    def __init__(
        self,
        config: Optional[NaiveBucketConfig] = None,
    ) -> None:
        self._base = NaiveBucketModel(config=config)

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y_threshold: Optional[Sequence[float]] = None,
        y_runtime: Optional[Sequence[float]] = None,
        feature_names: Optional[Sequence[str]] = None,
        threshold_col: str = "selected_threshold",
        runtime_col: str = "forward_wall_s",
    ) -> "NaiveBucketRuntimeModel":
        self._base.fit(
            X,
            y_threshold=y_threshold,
            y_runtime=y_runtime,
            feature_names=feature_names,
            threshold_col=threshold_col,
            runtime_col=runtime_col,
        )
        return self

    def predict(
        self,
        X: pd.DataFrame | np.ndarray,
        feature_names: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        _thresholds, runtimes = self._base.predict(X, feature_names=feature_names)
        return runtimes

    @property
    def base_model(self) -> NaiveBucketModel:
        return self._base
