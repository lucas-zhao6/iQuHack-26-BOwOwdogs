"""Simple LightGBM threshold + runtime runner."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from ..lgbm.threshold_model import ThresholdModel, apply_family_floor
from ..lgbm.runtime_model import RuntimeModel
from ...evaluation.scoring import threshold_to_idx
from ..data_loader import THRESHOLD_RUNGS


class LGBMSimpleRunner:
    name = "lgbm_simple"

    def __init__(
        self,
        threshold_params: Optional[dict] = None,
        runtime_params: Optional[dict] = None,
        safety_margin: float = 0.0,
        decision_policy: str = "expected_score",
        use_family_floor: bool = True,
    ):
        self.threshold_params = threshold_params or {}
        self.runtime_params = runtime_params or {}
        self.safety_margin = safety_margin
        self.decision_policy = decision_policy
        self.use_family_floor = use_family_floor

        self.threshold_model: Optional[ThresholdModel] = None
        self.runtime_model: Optional[RuntimeModel] = None

    def _can_use_threshold_eval_set(self, y_train: np.ndarray, y_val: Optional[np.ndarray]) -> bool:
        if y_val is None or y_val.size == 0:
            return False
        train_labels = {threshold_to_idx(v) for v in y_train}
        val_labels = {threshold_to_idx(v) for v in y_val}
        return val_labels.issubset(train_labels)

    def _expand_runtime_sweep(
        self,
        X: np.ndarray,
        df_subset,
        threshold_col: str = "selected_threshold",
        forward_threshold_col: str = "forward_threshold",
        forward_runtime_col: str = "forward_wall_s",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        df_local = df_subset.reset_index(drop=True)
        if len(df_local) != X.shape[0]:
            raise ValueError("expand_runtime_sweep: X and df_subset must align row-wise.")

        X_rows: list[np.ndarray] = []
        y_time: list[float] = []
        y_thr: list[float] = []

        for i, row in df_local.iterrows():
            has_sweep = False
            for rung in THRESHOLD_RUNGS:
                col = f"sweep_wall_{rung}"
                if col not in df_local.columns:
                    continue
                val = row[col]
                if val is None or not np.isfinite(val) or val <= 0:
                    continue
                X_rows.append(X[i])
                y_time.append(float(val))
                y_thr.append(float(rung))
                has_sweep = True

            if not has_sweep:
                val = row.get(forward_runtime_col, None)
                thr = row.get(threshold_col, None)
                if thr is None or not np.isfinite(thr):
                    thr = row.get(forward_threshold_col, None)
                if val is None or thr is None:
                    continue
                if np.isfinite(val) and val > 0 and np.isfinite(thr):
                    X_rows.append(X[i])
                    y_time.append(float(val))
                    y_thr.append(float(thr))

        if not X_rows:
            return np.empty((0, X.shape[1])), np.array([], dtype=float), np.array([], dtype=float)

        return np.vstack(X_rows), np.array(y_time, dtype=float), np.array(y_thr, dtype=float)

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
        self.threshold_model = ThresholdModel(
            lgb_params=self.threshold_params,
            safety_margin=self.safety_margin,
            decision_policy=self.decision_policy,
        )
        thr_eval_set = None
        if X_val is not None and y_thr_val is not None and self._can_use_threshold_eval_set(y_thr_train, y_thr_val):
            thr_eval_set = (X_val, y_thr_val)
        self.threshold_model.fit(X_train, y_thr_train, eval_set=thr_eval_set)

        self.runtime_model = RuntimeModel(lgb_params=self.runtime_params)
        X_train_exp, y_time_exp, y_thr_exp = self._expand_runtime_sweep(X_train, train_df)
        if X_train_exp.shape[0] == 0:
            X_train_exp = X_train
            y_time_exp = y_time_train
            y_thr_exp = y_thr_train

        X_val_exp = None
        y_time_val_exp = None
        y_thr_val_exp = None
        if X_val is not None and val_df is not None:
            X_val_exp, y_time_val_exp, y_thr_val_exp = self._expand_runtime_sweep(X_val, val_df)
            if X_val_exp.shape[0] == 0:
                X_val_exp = None
                y_time_val_exp = None
                y_thr_val_exp = None

        eval_set = None
        if X_val_exp is not None:
            eval_set = (X_val_exp, y_time_val_exp, y_thr_val_exp, None, None)

        self.runtime_model.fit(
            X_train_exp,
            y_time_exp,
            y_thr_exp,
            None,
            None,
            eval_set=eval_set,
        )

    def predict(
        self,
        test_df,
        X_test: np.ndarray,
        feature_names: List[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.threshold_model is None or self.runtime_model is None:
            raise RuntimeError("LGBMSimpleRunner not fitted.")

        thresholds = self.threshold_model.predict(X_test)
        if self.use_family_floor:
            families = test_df["predicted_family"].tolist() if "predicted_family" in test_df.columns else None
            if families is not None:
                thresholds = apply_family_floor(thresholds, families)

        times = self.runtime_model.predict(X_test, thresholds)
        return thresholds, times
