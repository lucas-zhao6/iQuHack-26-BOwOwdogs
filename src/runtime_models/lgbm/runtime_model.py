"""
Runtime model for forward run wall time.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import warnings

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
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="X does not have valid feature names, but LGBMRegressor was fitted with feature names",
)


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
        self._elastic_scaler = None

    def _safe_expm1(self, x: np.ndarray, max_log: float = 12.0) -> np.ndarray:
        """Exponentiate safely from log1p space."""
        x = np.nan_to_num(x, nan=0.0, posinf=max_log, neginf=-20.0)
        x = np.clip(x, -20.0, max_log)
        return np.expm1(x)

    def _add_threshold_feature(self, X: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
        """Add log2(threshold) as an additional feature column."""
        # Use log2 of threshold as feature (1->0, 2->1, 4->2, ..., 256->8)
        X = np.asarray(X)
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
        mask = (
            np.isfinite(pred)
            & np.isfinite(true)
            & (pred > 0)
            & (true > 0)
        )
        pred = pred[mask]
        true = true[mask]
        if pred.size < 5:
            self._calibration = None
            return
        x = np.log1p(np.clip(pred, 1e-6, None))
        y = np.log1p(np.clip(true, 1e-6, None))
        try:
            a, b = np.polyfit(x, y, 1)
            self._calibration = (float(a), float(b))
        except np.linalg.LinAlgError:
            self._calibration = None

    def _apply_calibration(self, pred: np.ndarray) -> np.ndarray:
        if not self._calibration:
            return pred
        a, b = self._calibration
        x = np.log1p(np.clip(pred, 1e-6, None))
        y = a * x + b
        return self._safe_expm1(y)

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

        self._elastic_scaler = StandardScaler()
        X_scaled = self._elastic_scaler.fit_transform(X_with_thr)
        enet = ElasticNet(
            alpha=0.002,
            l1_ratio=0.2,
            max_iter=15000,
            tol=1e-4,
            random_state=self.random_state,
        )
        enet.fit(X_scaled, y_log)
        self._ensemble_models["elasticnet"] = enet

        if eval_data is not None:
            X_val, y_val = eval_data
            base_pred = self._predict_base_from_features(X_val)
            preds = {
                "base": base_pred,
                "catboost": self._safe_expm1(cat.predict(X_val)),
                "elasticnet": self._safe_expm1(
                    enet.predict(
                        getattr(self, "_elastic_scaler", None).transform(X_val)
                    )
                    if getattr(self, "_elastic_scaler", None) is not None
                    else enet.predict(X_val)
                ),
            }

            best_score = -np.inf
            best_weights = (1.0, 0.0, 0.0)
            for w_base in np.linspace(0.0, 1.0, 6):
                for w_cat in np.linspace(0.0, 1.0 - w_base, 6):
                    w_enet = 1.0 - w_base - w_cat
                    if w_enet < 0:
                        continue
                    pred = (
                        w_base * preds["base"]
                        + w_cat * preds["catboost"]
                        + w_enet * preds["elasticnet"]
                    )
                    score = -np.mean(np.abs(np.log(np.clip(pred, 1e-6, None) / np.clip(y_val, 1e-6, None))))
                    if score > best_score:
                        best_score = score
                        best_weights = (w_base, w_cat, w_enet)
            self._blend_weights = best_weights

    def _predict_base_from_features(self, X_with_thr: np.ndarray) -> np.ndarray:
        """Predict runtime without ensemble/calibration."""
        if self._has_decomposed:
            log_setup = self.setup_model.predict(X_with_thr)
            log_per_shot = self.per_shot_model.predict(X_with_thr)
            setup = self._safe_expm1(log_setup)
            per_shot = self._safe_expm1(log_per_shot)
            total = setup + per_shot * 10000
        else:
            total = self._safe_expm1(self.total_model.predict(X_with_thr))
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
        X_with_thr = self._add_threshold_feature(np.asarray(X), thresholds)
        self._apply_monotone_constraints(X_with_thr)

        # Always fit total model as fallback
        y_log_total = np.log1p(y_total_time)

        callbacks = []
        fit_kwargs = {}
        eval_total = None
        if eval_set is not None:
            X_val, y_val, thr_val, y_setup_val, y_per_shot_val = eval_set
            X_val = np.asarray(X_val)
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
            valid = (
                (y_setup_time > 0)
                & (y_per_shot_time > 0)
                & np.isfinite(y_setup_time)
                & np.isfinite(y_per_shot_time)
            )
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
                    + w_cat * self._safe_expm1(self._ensemble_models["catboost"].predict(X_val_with_thr))
                    + w_enet * self._safe_expm1(
                        self._ensemble_models["elasticnet"].predict(
                            getattr(self, "_elastic_scaler", None).transform(X_val_with_thr)
                        )
                        if getattr(self, "_elastic_scaler", None) is not None
                        else self._ensemble_models["elasticnet"].predict(X_val_with_thr)
                    )
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
        Predict runtime from features and thresholds.

        Args:
            X: Feature matrix
            thresholds: Predicted threshold values (used as input feature)
            use_decomposed: Whether to use decomposed prediction
        """
        X_with_thr = self._add_threshold_feature(np.asarray(X), thresholds)

        if use_decomposed and self._has_decomposed:
            log_setup = self.setup_model.predict(X_with_thr)
            log_per_shot = self.per_shot_model.predict(X_with_thr)
            setup = self._safe_expm1(log_setup)
            per_shot = self._safe_expm1(log_per_shot)
            total = setup + per_shot * 10000
        else:
            log_total = self.total_model.predict(X_with_thr)
            total = self._safe_expm1(log_total)

        if self.use_ensemble and self._blend_weights is not None:
            w_base, w_cat, w_enet = self._blend_weights
            total = (
                w_base * total
                + w_cat * self._safe_expm1(self._ensemble_models["catboost"].predict(X_with_thr))
                + w_enet * self._safe_expm1(
                    self._ensemble_models["elasticnet"].predict(
                        getattr(self, "_elastic_scaler", None).transform(X_with_thr)
                    )
                    if getattr(self, "_elastic_scaler", None) is not None
                    else self._ensemble_models["elasticnet"].predict(X_with_thr)
                )
            )

        total = self._apply_calibration(total)
        return np.clip(total, 0.1, 10000)

    def predict_decomposed(
        self,
        X: np.ndarray,
        thresholds: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict decomposed runtime components."""
        X_with_thr = self._add_threshold_feature(np.asarray(X), thresholds)

        if self._has_decomposed:
            log_setup = self.setup_model.predict(X_with_thr)
            log_per_shot = self.per_shot_model.predict(X_with_thr)
            setup = self._safe_expm1(log_setup)
            per_shot = self._safe_expm1(log_per_shot)
            total = setup + per_shot * 10000
        else:
            total = self._safe_expm1(self.total_model.predict(X_with_thr))
            setup = total * 0.1
            per_shot = (total - setup) / 10000

        total = self._apply_calibration(total)
        return setup, per_shot, total
