"""Shared data preparation helpers for training/evaluation."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from ..models.lgbm.combined import get_feature_columns


def build_feature_matrix(
    df: pd.DataFrame,
    feature_cols: List[str] | None = None,
) -> Tuple[np.ndarray, List[str]]:
    """Build a numeric feature matrix with stable column ordering."""
    add_backend_flags = False
    if feature_cols is None:
        feature_cols = get_feature_columns(df.columns.tolist())
        valid_cols: list[str] = []
        for col in feature_cols:
            if col in df.columns:
                series = df[col]
                if series.dtype in [np.float64, np.int64, float, int]:
                    if series.notna().sum() > 0 and series.std() > 1e-10:
                        valid_cols.append(col)
        feature_cols = valid_cols
        X = df[feature_cols].copy()
        add_backend_flags = True
    else:
        data = {}
        for col in feature_cols:
            if col == "is_gpu":
                data[col] = (df["backend"] == "GPU").astype(float)
            elif col == "is_double":
                data[col] = (df["precision"] == "double").astype(float)
            elif col in df.columns:
                data[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                data[col] = 0.0
        X = pd.DataFrame(data, index=df.index)
        if "is_gpu" not in feature_cols or "is_double" not in feature_cols:
            add_backend_flags = True

    if add_backend_flags:
        if "is_gpu" not in feature_cols:
            X = X.assign(is_gpu=(df["backend"] == "GPU").astype(float))
            feature_cols = feature_cols + ["is_gpu"]
        if "is_double" not in feature_cols:
            X = X.assign(is_double=(df["precision"] == "double").astype(float))
            feature_cols = feature_cols + ["is_double"]

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))

    return X, feature_cols


def build_targets(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract model targets from the training dataframe."""
    y_threshold = df["selected_threshold"].values.astype(float)
    y_total_time = df["forward_wall_s"].values.astype(float)
    y_setup_time = df["estimated_setup_s"].values.astype(float)
    y_per_shot_time = df["estimated_per_shot_s"].values.astype(float)
    return y_threshold, y_total_time, y_setup_time, y_per_shot_time
