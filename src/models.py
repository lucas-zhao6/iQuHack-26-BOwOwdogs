"""
Model definitions for the Circuit Fingerprint Challenge.

Implements:
1. ThresholdModel: Ordinal regression with asymmetric safety margin calibration
2. RuntimeModel: Decomposed prediction (setup + per_shot * 10000)
3. CombinedPredictor: End-to-end prediction pipeline
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

try:
    import lightgbm as lgb
except ImportError as exc:
    raise ImportError(
        "LightGBM is required for training. Install with 'pip install lightgbm'."
    ) from exc

try:
    from catboost import CatBoostRegressor
except ImportError as exc:
    raise ImportError(
        "CatBoost is required for runtime ensemble. Install with 'pip install catboost'."
    ) from exc

from sklearn.linear_model import ElasticNet

from .scoring import (
    THRESHOLD_RUNGS,
    idx_to_threshold,
    threshold_to_idx,
    compute_threshold_score,
)


class ThresholdModel:
    """
    Predicts the minimum threshold rung needed to achieve >= 0.99 fidelity.

    Uses LightGBM multiclass classification over the 9 threshold rungs.
    Prediction uses a risk-aware decision rule that maximizes expected
    threshold score under the competition metric.
    """

    def __init__(
        self,
        lgb_params: Optional[Dict[str, Any]] = None,
        n_estimators: int = 1000,
        early_stopping_rounds: int = 50,
        decision_policy: str = "expected_score",
        safety_margin: float = 0.0,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.early_stopping_rounds = early_stopping_rounds
        self.decision_policy = decision_policy
        self.safety_margin = safety_margin
        self.random_state = random_state

        default_params = {
            "objective": "multiclass",
            "num_class": 9,
            "n_estimators": n_estimators,
            "learning_rate": 0.03,
            "max_depth": 4,
            "num_leaves": 15,
            "min_child_samples": 20,
            "min_gain_to_split": 0.01,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "lambda_l1": 1.0,
            "lambda_l2": 2.0,
            "extra_trees": True,
            "random_state": random_state,
            "verbosity": -1,
        }
        if lgb_params:
            default_params.update(lgb_params)

        self.params = default_params
        self.model = lgb.LGBMClassifier(**self.params)
        self._fitted = False
        self._score_matrix = self._build_score_matrix()

    def _build_score_matrix(self) -> np.ndarray:
        n_classes = len(THRESHOLD_RUNGS)
        matrix = np.zeros((n_classes, n_classes), dtype=float)
        for pred_idx in range(n_classes):
            pred_thr = idx_to_threshold(pred_idx)
            for true_idx in range(n_classes):
                true_thr = idx_to_threshold(true_idx)
                matrix[pred_idx, true_idx] = compute_threshold_score(pred_thr, true_thr)
        return matrix

    def _predict_proba_full(self, X: np.ndarray) -> np.ndarray:
        """Return predict_proba padded to all 9 threshold classes."""
        proba = self.model.predict_proba(X)
        n_classes = len(THRESHOLD_RUNGS)
        if proba.shape[1] == n_classes:
            return proba

        full = np.zeros((proba.shape[0], n_classes), dtype=float)
        classes = getattr(self.model, "classes_", None)
        if classes is None:
            raise RuntimeError("Model classes_ not available for probability padding.")
        for src_idx, class_label in enumerate(classes):
            class_idx = int(class_label)
            if 0 <= class_idx < n_classes:
                full[:, class_idx] = proba[:, src_idx]
        return full

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for all 9 threshold rungs."""
        return self._predict_proba_full(X)

    def fit(
        self,
        X: np.ndarray,
        y_threshold: np.ndarray,
        eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ):
        """
        Fit the threshold model.

        Args:
            X: Feature matrix (n_samples, n_features)
            y_threshold: Threshold values (1, 2, 4, ..., 256)
            eval_set: Optional (X_val, y_val) for early stopping
        """
        y_idx = np.array([threshold_to_idx(t) for t in y_threshold])

        callbacks = []
        fit_kwargs = {}
        if eval_set is not None:
            X_val, y_val = eval_set
            y_val_idx = np.array([threshold_to_idx(t) for t in y_val])
            fit_kwargs["eval_set"] = [(X_val, y_val_idx)]
            fit_kwargs["eval_metric"] = "multi_logloss"
            if self.early_stopping_rounds > 0:
                callbacks.append(
                    lgb.early_stopping(self.early_stopping_rounds, verbose=False)
                )
        if callbacks:
            fit_kwargs["callbacks"] = callbacks

        self.model.fit(X, y_idx, **fit_kwargs)
        self._fitted = True

    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        """Predict expected threshold index (continuous)."""
        proba = self._predict_proba_full(X)
        idxs = np.arange(proba.shape[1])
        return proba @ idxs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict threshold values with risk-aware decision rule."""
        proba = self._predict_proba_full(X)
        if self.decision_policy == "argmax":
            pred_idx = np.argmax(proba, axis=1)
        else:
            expected_scores = proba @ self._score_matrix.T
            pred_idx = np.argmax(expected_scores, axis=1)

        if self.safety_margin != 0.0:
            pred_idx = np.clip(np.round(pred_idx + self.safety_margin), 0, 8).astype(int)

        return np.array([idx_to_threshold(i) for i in pred_idx])

    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict thresholds with uncertainty estimate.

        Returns: (predictions, uncertainties)
        Uncertainty is estimated from the training residuals at similar feature values.
        """
        proba = self._predict_proba_full(X)
        predictions = self.predict(X)
        uncertainty = 1.0 - np.max(proba, axis=1)
        return predictions, uncertainty


class RuntimeModel:
    """
    Predicts forward run wall time using decomposed modeling.

    total_time = setup_time + per_shot_time * 10000
    Both setup and per-shot time are modeled in log-space with LightGBM.
    """

    def __init__(
        self,
        lgb_params: Optional[Dict[str, Any]] = None,
        n_estimators: int = 1200,
        early_stopping_rounds: int = 50,
        use_ensemble: bool = True,
        use_calibration: bool = True,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.early_stopping_rounds = early_stopping_rounds
        self.use_ensemble = use_ensemble
        self.use_calibration = use_calibration
        self.random_state = random_state

        default_params = {
            "objective": "regression_l1",
            "n_estimators": n_estimators,
            "learning_rate": 0.03,
            "max_depth": 5,
            "num_leaves": 31,
            "min_child_samples": 15,
            "min_gain_to_split": 0.01,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "lambda_l1": 1.0,
            "lambda_l2": 2.0,
            "extra_trees": True,
            "random_state": random_state,
            "verbosity": -1,
        }
        if lgb_params:
            default_params.update(lgb_params)

        self.params = default_params

        # Two separate models for setup and per-shot
        self.setup_model = lgb.LGBMRegressor(**{**self.params, "random_state": random_state})
        self.per_shot_model = lgb.LGBMRegressor(**{**self.params, "random_state": random_state + 1})

        # Fallback: direct total time model
        self.total_model = lgb.LGBMRegressor(**{**self.params, "random_state": random_state + 2})

        self._fitted = False
        self._has_decomposed = False
        self._calibration = None
        self._blend_weights = None
        self._ensemble_models = {}

    def _add_threshold_feature(self, X: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
        """Add log2(threshold) as an additional feature column."""
        # Use log2 of threshold as feature (1->0, 2->1, 4->2, ..., 256->8)
        thr_feature = np.log2(np.clip(thresholds, 1, 512)).reshape(-1, 1)
        return np.hstack([X, thr_feature])

    def _apply_monotone_constraints(self, X_with_thr: np.ndarray):
        """Set monotone constraints to enforce increasing runtime with threshold."""
        if self.params.get("objective") == "regression_l1":
            return
        n_features = X_with_thr.shape[1]
        constraints = [0] * n_features
        constraints[-1] = 1
        for model in (self.total_model, self.setup_model, self.per_shot_model):
            model.set_params(monotone_constraints=constraints)

    def _fit_calibrator(self, pred: np.ndarray, true: np.ndarray):
        """Fit linear calibration in log-space: log(true) ≈ a*log(pred)+b."""
        x = np.log1p(np.clip(pred, 1e-6, None))
        y = np.log1p(np.clip(true, 1e-6, None))
        if len(x) < 5:
            self._calibration = None
            return
        a, b = np.polyfit(x, y, 1)
        self._calibration = (float(a), float(b))

    def _apply_calibration(self, pred: np.ndarray) -> np.ndarray:
        if not self._calibration:
            return pred
        a, b = self._calibration
        x = np.log1p(np.clip(pred, 1e-6, None))
        y = a * x + b
        return np.expm1(y)

    def _fit_runtime_ensemble(
        self,
        X_with_thr: np.ndarray,
        y_total_time: np.ndarray,
        eval_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ):
        """Fit auxiliary ensemble models on log1p(time)."""
        y_log = np.log1p(y_total_time)

        cat = CatBoostRegressor(
            depth=6,
            learning_rate=0.05,
            loss_function="MAE",
            iterations=800,
            l2_leaf_reg=6.0,
            random_seed=self.random_state,
            verbose=False,
        )
        cat.fit(X_with_thr, y_log)
        self._ensemble_models["catboost"] = cat

        enet = ElasticNet(alpha=0.002, l1_ratio=0.2, max_iter=5000, random_state=self.random_state)
        enet.fit(X_with_thr, y_log)
        self._ensemble_models["elasticnet"] = enet

        if eval_data is not None:
            X_val, y_val = eval_data
            base_pred = self._predict_base_from_features(X_val)
            preds = {
                "base": base_pred,
                "catboost": np.expm1(cat.predict(X_val)),
                "elasticnet": np.expm1(enet.predict(X_val)),
            }
            best_score = -1.0
            best_weights = (0.6, 0.2, 0.2)
            grid = np.linspace(0.0, 1.0, 11)
            for w_base in grid:
                for w_cat in grid:
                    w_enet = 1.0 - w_base - w_cat
                    if w_enet < 0:
                        continue
                    blended = (
                        w_base * preds["base"]
                        + w_cat * preds["catboost"]
                        + w_enet * preds["elasticnet"]
                    )
                    scores = []
                    for p, t in zip(blended, y_val):
                        scores.append(min(p / t, t / p) if p > 0 and t > 0 else 0.0)
                    score = float(np.mean(scores))
                    if score > best_score:
                        best_score = score
                        best_weights = (w_base, w_cat, w_enet)
            self._blend_weights = best_weights

    def _predict_base_from_features(self, X_with_thr: np.ndarray) -> np.ndarray:
        """Predict runtime without ensemble/calibration."""
        if self._has_decomposed:
            log_setup = self.setup_model.predict(X_with_thr)
            log_per_shot = self.per_shot_model.predict(X_with_thr)
            setup = np.expm1(log_setup)
            per_shot = np.expm1(log_per_shot)
            total = setup + per_shot * 10000
        else:
            total = np.expm1(self.total_model.predict(X_with_thr))
        return np.clip(total, 0.1, 10000)

    def fit(
        self,
        X: np.ndarray,
        y_total_time: np.ndarray,
        thresholds: np.ndarray,
        y_setup_time: Optional[np.ndarray] = None,
        y_per_shot_time: Optional[np.ndarray] = None,
        eval_set: Optional[
            Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]
        ] = None,
    ):
        """
        Fit the runtime model.

        Args:
            X: Feature matrix (n_samples, n_features)
            y_total_time: Total forward run time
            thresholds: Threshold values (used as input feature)
            y_setup_time: Optional setup time for decomposed modeling
            y_per_shot_time: Optional per-shot time for decomposed modeling
            eval_set: Optional validation tuple for early stopping
        """
        # Add threshold as input feature
        X_with_thr = self._add_threshold_feature(X, thresholds)
        self._apply_monotone_constraints(X_with_thr)

        # Always fit total model as fallback
        y_log_total = np.log1p(y_total_time)

        callbacks = []
        fit_kwargs = {}
        eval_total = None
        if eval_set is not None:
            X_val, y_val, thr_val, y_setup_val, y_per_shot_val = eval_set
            X_val_with_thr = self._add_threshold_feature(X_val, thr_val)
            eval_total = (X_val_with_thr, np.log1p(y_val))
            fit_kwargs["eval_set"] = [eval_total]
            fit_kwargs["eval_metric"] = "l1"
            if self.early_stopping_rounds > 0:
                callbacks.append(
                    lgb.early_stopping(self.early_stopping_rounds, verbose=False)
                )
        if callbacks:
            fit_kwargs["callbacks"] = callbacks

        self.total_model.fit(X_with_thr, y_log_total, **fit_kwargs)

        # Try decomposed if data available
        if y_setup_time is not None and y_per_shot_time is not None:
            # Filter out invalid values
            valid = (y_setup_time > 0) & (y_per_shot_time > 0) & np.isfinite(y_setup_time) & np.isfinite(y_per_shot_time)
            if valid.sum() > 10:
                y_log_setup = np.log1p(y_setup_time[valid])
                y_log_per_shot = np.log1p(y_per_shot_time[valid])
                X_valid = X_with_thr[valid]

                setup_kwargs = {}
                per_shot_kwargs = {}
                if eval_set is not None and eval_total is not None:
                    if y_setup_val is not None and y_per_shot_val is not None:
                        valid_val = (
                            (y_setup_val > 0)
                            & (y_per_shot_val > 0)
                            & np.isfinite(y_setup_val)
                            & np.isfinite(y_per_shot_val)
                        )
                        if valid_val.sum() > 5:
                            setup_kwargs["eval_set"] = [
                                (eval_total[0][valid_val], np.log1p(y_setup_val[valid_val]))
                            ]
                            setup_kwargs["eval_metric"] = "l1"
                            per_shot_kwargs["eval_set"] = [
                                (eval_total[0][valid_val], np.log1p(y_per_shot_val[valid_val]))
                            ]
                            per_shot_kwargs["eval_metric"] = "l1"
                            if self.early_stopping_rounds > 0:
                                setup_kwargs["callbacks"] = [
                                    lgb.early_stopping(self.early_stopping_rounds, verbose=False)
                                ]
                                per_shot_kwargs["callbacks"] = [
                                    lgb.early_stopping(self.early_stopping_rounds, verbose=False)
                                ]

                self.setup_model.fit(X_valid, y_log_setup, **setup_kwargs)
                self.per_shot_model.fit(X_valid, y_log_per_shot, **per_shot_kwargs)
                self._has_decomposed = True

        if self.use_ensemble:
            eval_data = None
            if eval_set is not None:
                X_val, y_val, thr_val, _y_setup_val, _y_per_shot_val = eval_set
                X_val_with_thr = self._add_threshold_feature(X_val, thr_val)
                eval_data = (X_val_with_thr, y_val)
            self._fit_runtime_ensemble(X_with_thr, y_total_time, eval_data=eval_data)

        if self.use_calibration and eval_set is not None:
            X_val, y_val, thr_val, _y_setup_val, _y_per_shot_val = eval_set
            X_val_with_thr = self._add_threshold_feature(X_val, thr_val)
            base_pred = self._predict_base_from_features(X_val_with_thr)
            if self.use_ensemble and self._blend_weights is not None:
                w_base, w_cat, w_enet = self._blend_weights
                pred = (
                    w_base * base_pred
                    + w_cat * np.expm1(self._ensemble_models["catboost"].predict(X_val_with_thr))
                    + w_enet * np.expm1(self._ensemble_models["elasticnet"].predict(X_val_with_thr))
                )
            else:
                pred = base_pred
            self._fit_calibrator(pred, y_val)

        self._fitted = True

    def predict(
        self,
        X: np.ndarray,
        thresholds: np.ndarray,
        use_decomposed: bool = True,
    ) -> np.ndarray:
        """
        Predict total forward run time.

        Args:
            X: Feature matrix
            thresholds: Predicted threshold values (used as input feature)
            use_decomposed: Whether to use decomposed prediction
        """
        X_with_thr = self._add_threshold_feature(X, thresholds)

        if use_decomposed and self._has_decomposed:
            log_setup = self.setup_model.predict(X_with_thr)
            log_per_shot = self.per_shot_model.predict(X_with_thr)

            setup = np.expm1(log_setup)
            per_shot = np.expm1(log_per_shot)
            total = setup + per_shot * 10000
        else:
            log_total = self.total_model.predict(X_with_thr)
            total = np.expm1(log_total)

        if self.use_ensemble and self._blend_weights is not None:
            w_base, w_cat, w_enet = self._blend_weights
            total = (
                w_base * total
                + w_cat * np.expm1(self._ensemble_models["catboost"].predict(X_with_thr))
                + w_enet * np.expm1(self._ensemble_models["elasticnet"].predict(X_with_thr))
            )

        total = self._apply_calibration(total)
        return np.clip(total, 0.1, 10000)

    def predict_decomposed(
        self,
        X: np.ndarray,
        thresholds: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict setup time, per-shot time, and total time.

        Returns: (setup_time, per_shot_time, total_time)
        """
        X_with_thr = self._add_threshold_feature(X, thresholds)

        if self._has_decomposed:
            log_setup = self.setup_model.predict(X_with_thr)
            log_per_shot = self.per_shot_model.predict(X_with_thr)
            setup = np.expm1(log_setup)
            per_shot = np.expm1(log_per_shot)
            total = setup + per_shot * 10000
        else:
            total = np.expm1(self.total_model.predict(X_with_thr))
            # Rough decomposition for non-decomposed model
            setup = total * 0.05  # Assume 5% is setup
            per_shot = (total - setup) / 10000

        total = self._apply_calibration(total)
        return setup, per_shot, total


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


# Families that need curve-aware prediction (high threshold families)
# For these families, we use max(direct, curve) to avoid under-prediction
HIGH_THRESHOLD_FAMILIES = {
    "TwoLocalRandom", "QNN", "Portfolio_QAOA", "Portfolio_VQE",
    "Pricing_Call", "Ground_State", "Amplitude_Estimation", "Shor", "GraphState"
}


# Family-to-minimum-threshold mapping from training data analysis
# This is our Innovation #3: use detected family to set floor thresholds
FAMILY_THRESHOLD_FLOORS = {
    # High threshold families (these NEVER work at low thresholds)
    "TwoLocalRandom": 64,       # Needs 256 but 64 is safe floor
    "QNN": 16,                  # Needs 32
    "Portfolio_QAOA": 8,        # Needs 16
    "Portfolio_VQE": 8,         # Needs 16
    "Pricing_Call": 4,          # Needs 8
    "Ground_State": 4,          # Needs 8
    "Amplitude_Estimation": 4,  # Needs 4-16
    "GraphState": 2,            # Needs 4
    "Shor": 2,                  # Needs 4

    # Medium threshold families
    "CutBell": 2,
    "QFT_Entangled": 2,
    "VQE": 2,
    "W_State": 2,
    "GHZ": 2,

    # Low threshold families (easy for MPS)
    "Grover_V_Chain": 1,
    "Grover_NoAncilla": 1,
    "QAOA": 1,
    "QFT": 1,
    "QPE_Exact": 1,
    "Deutsch_Jozsa": 1,

    # Unknown - be conservative
    "Unknown": 4,
}


def apply_family_floor(
    predicted_thresholds: np.ndarray,
    families: List[str],
) -> np.ndarray:
    """
    Apply family-based minimum threshold floors.

    This prevents under-prediction for families that are known to require
    high thresholds, regardless of what the ML model predicts.
    """
    result = predicted_thresholds.copy()
    for i, family in enumerate(families):
        floor = FAMILY_THRESHOLD_FLOORS.get(family, 1)
        if result[i] < floor:
            result[i] = floor
    return result


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
