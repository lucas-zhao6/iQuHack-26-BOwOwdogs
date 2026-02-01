"""
Naive bucketed baseline model for comparison.

Discretizes a small set of circuit features into coarse bins, conditioned on
(mode = precision, device). Predicts:
- threshold_pred: majority threshold in the bucket (tie -> smallest threshold)
- runtime_pred: exp(median(log(runtime))) in the bucket
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import warnings

import numpy as np
import pandas as pd

import sys
sys.modules.setdefault("src.naive_model", sys.modules[__name__])
sys.modules.setdefault("src.models.naive_model", sys.modules[__name__])


@dataclass
class NaiveBucketConfig:
    n_bins: int = 5
    bucket_features: Tuple[str, ...] = ("n_qubits", "twoq_gate_count", "twoq_depth")
    precision_col: str = "precision"
    device_col: str = "backend"
    is_double_col: str = "is_double"
    is_gpu_col: str = "is_gpu"
    runtime_epsilon: float = 1e-9


DEFAULT_FEATURE_ALIASES: Dict[str, Tuple[str, ...]] = {
    "n_qubits": ("n_qubits", "num_qubits", "nq"),
    "twoq_gate_count": ("twoq_gate_count", "n_2q_gates", "n_entangling_gates", "cx_count", "num_2q_gates"),
    "twoq_depth": ("twoq_depth", "depth_2q", "entangling_depth", "n_entangling_layers", "entangling_layers", "circuit_depth"),
}


class NaiveBucketModel:
    """
    Naive bucketed baseline model.

    Fit expects a feature table that includes mode fields (precision, device)
    and target columns (threshold, runtime). Use fit() to learn bin edges
    and per-bucket statistics, then predict() for inference.
    """

    def __init__(
        self,
        config: Optional[NaiveBucketConfig] = None,
        feature_aliases: Optional[Dict[str, Sequence[str]]] = None,
    ) -> None:
        self.config = config or NaiveBucketConfig()
        self.feature_aliases = {
            **DEFAULT_FEATURE_ALIASES,
            **(feature_aliases or {}),
        }

        self.bucket_feature_names: List[str] = []
        self.bin_edges: List[np.ndarray] = []
        self.feature_medians: Dict[str, float] = {}

        self.bucket_majority_threshold: Dict[Tuple[str, str, Tuple[int, ...]], int] = {}
        self.bucket_median_log_runtime: Dict[Tuple[str, str, Tuple[int, ...]], float] = {}
        self.mode_majority_threshold: Dict[Tuple[str, str], int] = {}
        self.mode_median_log_runtime: Dict[Tuple[str, str], float] = {}
        self.global_majority_threshold: int = 1
        self.global_median_log_runtime: float = 0.0
        self.runtime_invalid_count: int = 0
        self._fitted = False

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y_threshold: Optional[Sequence[float]] = None,
        y_runtime: Optional[Sequence[float]] = None,
        feature_names: Optional[Sequence[str]] = None,
        threshold_col: str = "selected_threshold",
        runtime_col: str = "forward_wall_s",
    ) -> "NaiveBucketModel":
        df = self._ensure_dataframe(X, feature_names)

        if y_threshold is None:
            if threshold_col not in df.columns:
                raise ValueError(f"Missing threshold column: {threshold_col}")
            y_threshold_arr = df[threshold_col].values
        else:
            y_threshold_arr = np.asarray(y_threshold)

        if y_runtime is None:
            if runtime_col not in df.columns:
                raise ValueError(f"Missing runtime column: {runtime_col}")
            y_runtime_arr = df[runtime_col].values
        else:
            y_runtime_arr = np.asarray(y_runtime)

        self.bucket_feature_names = self._resolve_bucket_features(df.columns)
        bucket_values = self._prepare_bucket_values(df, self.bucket_feature_names)
        self.bin_edges = self._fit_bin_edges(bucket_values, self.config.n_bins)

        modes = self._extract_modes(df)
        thresholds = np.asarray(y_threshold_arr, dtype=float)
        runtimes = np.asarray(y_runtime_arr, dtype=float)

        bucket_thresholds: Dict[Tuple[str, str, Tuple[int, ...]], List[int]] = {}
        bucket_log_runtimes: Dict[Tuple[str, str, Tuple[int, ...]], List[float]] = {}
        mode_thresholds: Dict[Tuple[str, str], List[int]] = {}
        mode_log_runtimes: Dict[Tuple[str, str], List[float]] = {}
        global_thresholds: List[int] = []
        global_log_runtimes: List[float] = []

        bucket_indices = self._assign_bins(bucket_values)

        invalid_runtime = 0
        for i in range(len(df)):
            mode_key = modes[i]
            bucket_key = (mode_key[0], mode_key[1], tuple(bucket_indices[i]))

            thr = int(thresholds[i])
            bucket_thresholds.setdefault(bucket_key, []).append(thr)
            mode_thresholds.setdefault(mode_key, []).append(thr)
            global_thresholds.append(thr)

            rt = runtimes[i]
            if rt <= 0 or not np.isfinite(rt):
                invalid_runtime += 1
                continue
            log_rt = float(np.log(rt))
            bucket_log_runtimes.setdefault(bucket_key, []).append(log_rt)
            mode_log_runtimes.setdefault(mode_key, []).append(log_rt)
            global_log_runtimes.append(log_rt)

        self.runtime_invalid_count = invalid_runtime
        if invalid_runtime > 0:
            warnings.warn(
                f"NaiveBucketModel: filtered {invalid_runtime} non-positive/invalid runtimes",
                RuntimeWarning,
            )

        self.bucket_majority_threshold = {
            k: self._majority_threshold(v) for k, v in bucket_thresholds.items()
        }
        self.bucket_median_log_runtime = {
            k: float(np.median(v)) for k, v in bucket_log_runtimes.items() if v
        }
        self.mode_majority_threshold = {
            k: self._majority_threshold(v) for k, v in mode_thresholds.items()
        }
        self.mode_median_log_runtime = {
            k: float(np.median(v)) for k, v in mode_log_runtimes.items() if v
        }

        if global_thresholds:
            self.global_majority_threshold = self._majority_threshold(global_thresholds)
        if global_log_runtimes:
            self.global_median_log_runtime = float(np.median(global_log_runtimes))
        else:
            self.global_median_log_runtime = float(np.log(self.config.runtime_epsilon))

        self._fitted = True
        return self

    def predict(
        self,
        X: pd.DataFrame | np.ndarray,
        feature_names: Optional[Sequence[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not self._fitted:
            raise RuntimeError("NaiveBucketModel must be fitted before predict().")

        df = self._ensure_dataframe(X, feature_names)
        bucket_values = self._prepare_bucket_values(df, self.bucket_feature_names)
        bucket_indices = self._assign_bins(bucket_values)
        modes = self._extract_modes(df)

        n = len(df)
        pred_thresholds = np.zeros(n, dtype=int)
        pred_runtimes = np.zeros(n, dtype=float)

        for i in range(n):
            mode_key = modes[i]
            bucket_key = (mode_key[0], mode_key[1], tuple(bucket_indices[i]))

            thr = self.bucket_majority_threshold.get(
                bucket_key,
                self.mode_majority_threshold.get(
                    mode_key,
                    self.global_majority_threshold,
                ),
            )

            log_rt = self.bucket_median_log_runtime.get(
                bucket_key,
                self.mode_median_log_runtime.get(
                    mode_key,
                    self.global_median_log_runtime,
                ),
            )

            pred_thresholds[i] = int(thr)
            pred_runtimes[i] = float(np.exp(log_rt))

        return pred_thresholds, pred_runtimes

    def _ensure_dataframe(
        self,
        X: pd.DataFrame | np.ndarray,
        feature_names: Optional[Sequence[str]],
    ) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        if feature_names is None:
            raise ValueError("feature_names is required when X is a numpy array.")
        return pd.DataFrame(X, columns=list(feature_names))

    def _resolve_bucket_features(self, columns: Iterable[str]) -> List[str]:
        columns_set = set(columns)
        selected: List[str] = []
        missing: List[str] = []

        for canonical in self.config.bucket_features:
            aliases = self.feature_aliases.get(canonical, (canonical,))
            found = None
            for name in aliases:
                if name in columns_set:
                    found = name
                    break
            if found is None:
                missing.append(canonical)
            else:
                selected.append(found)

        if missing:
            raise ValueError(f"Missing bucket feature(s): {', '.join(missing)}")
        return selected

    def _prepare_bucket_values(self, df: pd.DataFrame, cols: Sequence[str]) -> np.ndarray:
        bucket_values = df[list(cols)].copy()
        bucket_values = bucket_values.replace([np.inf, -np.inf], np.nan)

        for col in cols:
            series = pd.to_numeric(bucket_values[col], errors="coerce")
            if self._fitted and col in self.feature_medians:
                median = self.feature_medians[col]
            else:
                median = float(series.median()) if series.notna().any() else 0.0
                self.feature_medians[col] = median
            bucket_values[col] = series.fillna(median)

        return bucket_values.values.astype(float)

    def _fit_bin_edges(self, values: np.ndarray, n_bins: int) -> List[np.ndarray]:
        edges_list: List[np.ndarray] = []
        for col in range(values.shape[1]):
            col_values = values[:, col]
            edges_list.append(self._quantile_edges(col_values, n_bins))
        return edges_list

    def _assign_bins(self, values: np.ndarray) -> np.ndarray:
        bins = np.zeros_like(values, dtype=int)
        for col, edges in enumerate(self.bin_edges):
            if edges.size <= 2:
                bins[:, col] = 0
                continue
            bins[:, col] = np.searchsorted(edges[1:-1], values[:, col], side="right")
        return bins

    def _extract_modes(self, df: pd.DataFrame) -> List[Tuple[str, str]]:
        precision_values = None
        device_values = None

        if self.config.precision_col in df.columns:
            precision_values = df[self.config.precision_col].values
        elif self.config.is_double_col in df.columns:
            precision_values = df[self.config.is_double_col].values

        if self.config.device_col in df.columns:
            device_values = df[self.config.device_col].values
        elif self.config.is_gpu_col in df.columns:
            device_values = df[self.config.is_gpu_col].values

        n = len(df)
        modes: List[Tuple[str, str]] = []
        for i in range(n):
            precision_fallback = None
            if self.config.is_double_col in df.columns:
                precision_fallback = df[self.config.is_double_col].values[i]

            device_fallback = None
            if self.config.is_gpu_col in df.columns:
                device_fallback = df[self.config.is_gpu_col].values[i]

            precision = self._normalize_precision(
                precision_values[i] if precision_values is not None else None,
                fallback_is_double=precision_fallback if precision_values is None else None,
            )
            if precision == "unknown" and precision_values is not None and precision_fallback is not None:
                precision = self._normalize_precision(None, fallback_is_double=precision_fallback)

            device = self._normalize_device(
                device_values[i] if device_values is not None else None,
                fallback_is_gpu=device_fallback if device_values is None else None,
            )
            if device == "unknown" and device_values is not None and device_fallback is not None:
                device = self._normalize_device(None, fallback_is_gpu=device_fallback)
            modes.append((precision, device))
        return modes

    def _normalize_precision(self, value, fallback_is_double=None) -> str:
        if fallback_is_double is not None:
            return "double" if bool(fallback_is_double) else "single"
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "unknown"
        if isinstance(value, (bool, np.bool_)):
            return "double" if value else "single"
        if isinstance(value, (int, np.integer)):
            return "double" if value != 0 else "single"
        if isinstance(value, str):
            v = value.strip().lower()
            if "double" in v or v in {"fp64", "float64"}:
                return "double"
            if "single" in v or v in {"fp32", "float32"}:
                return "single"
        return "unknown"

    def _normalize_device(self, value, fallback_is_gpu=None) -> str:
        if fallback_is_gpu is not None:
            return "GPU" if bool(fallback_is_gpu) else "CPU"
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "unknown"
        if isinstance(value, (bool, np.bool_)):
            return "GPU" if value else "CPU"
        if isinstance(value, (int, np.integer)):
            return "GPU" if value != 0 else "CPU"
        if isinstance(value, str):
            v = value.strip().lower()
            if "gpu" in v:
                return "GPU"
            if "cpu" in v:
                return "CPU"
        return "unknown"

    def _majority_threshold(self, values: Sequence[int]) -> int:
        unique, counts = np.unique(values, return_counts=True)
        max_count = counts.max()
        candidates = unique[counts == max_count]
        return int(candidates.min())

    def _quantile_edges(self, values: np.ndarray, n_bins: int) -> np.ndarray:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return np.array([-1.0, 1.0])
        unique = np.unique(finite)
        if unique.size == 1:
            v = unique[0]
            return np.array([v - 1e-9, v + 1e-9])

        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.quantile(finite, quantiles)
        edges = np.unique(edges)
        if edges.size < 2:
            vmin = float(finite.min())
            vmax = float(finite.max())
            if vmin == vmax:
                return np.array([vmin - 1e-9, vmax + 1e-9])
            return np.array([vmin, vmax])
        return edges
