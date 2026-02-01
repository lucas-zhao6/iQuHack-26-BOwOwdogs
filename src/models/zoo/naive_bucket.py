"""Naive bucketed baseline runner."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from ..naive_bucket.model import NaiveBucketModel


class NaiveBucketRunner:
    name = "naive_bucket"

    def __init__(self):
        self.model: Optional[NaiveBucketModel] = None

    def fit(
        self,
        train_df,
        X_train: np.ndarray,
        y_thr_train: np.ndarray,
        y_time_train: np.ndarray,
        feature_names: List[str],
        val_df=None,
        X_val: Optional[np.ndarray] = None,
        y_thr_val: Optional[np.ndarray] = None,
        y_time_val: Optional[np.ndarray] = None,
    ) -> None:
        self.model = NaiveBucketModel()
        self.model.fit(train_df, threshold_col="selected_threshold", runtime_col="forward_wall_s")

    def predict(
        self,
        test_df,
        X_test: np.ndarray,
        feature_names: List[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.model is None:
            raise RuntimeError("NaiveBucketRunner not fitted.")
        return self.model.predict(test_df)
