#!/usr/bin/env python3
"""Evaluate and compare pretrained models on fixed CV splits."""

from __future__ import annotations

import sys
import pickle
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names*")
warnings.filterwarnings("ignore", category=UserWarning, message="X has feature names, but StandardScaler was fitted without feature names*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*joblib will operate in serial mode.*")

import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.cv_splits import load_cv_splits
from src.data.data_prep import build_feature_matrix
from src.models.lgbm.combined import CombinedPredictor
from src.models.lgbm_curve import LGBMCurveFidelityModel, DEFAULT_RUNGS
from src.evaluation.scoring import compute_threshold_metrics, compute_runtime_metrics

OUTPUT_DIR = PROJECT_ROOT / "outputs"
FEATURES_PATH = OUTPUT_DIR / "training_features.pkl"
CV_SPLITS_PATH = OUTPUT_DIR / "cv_splits.json"
COMBINED_DIR = OUTPUT_DIR / "combined_model"
NAIVE_DIR = OUTPUT_DIR / "naive_bucket"
CURVE_DIR = OUTPUT_DIR / "lgbm_curve"


def load_data() -> pd.DataFrame:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Feature file not found: {FEATURES_PATH}\n"
            "Run 'python run_phase1.py' first."
        )
    return pd.read_pickle(FEATURES_PATH)


def evaluate_model(
    name: str,
    df: pd.DataFrame,
    splits: list[dict],
    predict_fold,
) -> dict:
    all_pred_thresholds: list[int] = []
    all_true_thresholds: list[int] = []
    all_pred_times: list[float] = []
    all_true_times: list[float] = []

    for fold, split in enumerate(splits):
        test_df = df.iloc[split["test_idx"]]
        pred_thr, pred_time = predict_fold(fold, test_df)

        all_pred_thresholds.extend([int(v) for v in pred_thr])
        all_true_thresholds.extend([int(v) for v in test_df["selected_threshold"].values])
        all_pred_times.extend([float(v) for v in pred_time])
        all_true_times.extend([float(v) for v in test_df["forward_wall_s"].values])

    scores = {}
    scores.update(compute_threshold_metrics(all_pred_thresholds, all_true_thresholds))
    scores.update(compute_runtime_metrics(all_pred_times, all_true_times))

    return {"name": name, "scores": scores}


def main() -> None:
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    if not CV_SPLITS_PATH.exists():
        raise FileNotFoundError(
            f"CV splits not found: {CV_SPLITS_PATH}\n"
            "Run 'python run_generate_cv_splits.py' first."
        )

    df = load_data()
    valid = df["selected_threshold"].notna() & df["forward_wall_s"].notna()
    df = df[valid].reset_index(drop=True)

    splits, meta = load_cv_splits(CV_SPLITS_PATH)
    if meta.get("n_samples") is not None and meta["n_samples"] != len(df):
        raise ValueError(
            "CV splits do not match current dataset size. "
            f"Splits have n_samples={meta['n_samples']} but data has {len(df)} rows."
        )

    def predict_combined(fold: int, test_df: pd.DataFrame):
        model_path = COMBINED_DIR / f"fold_{fold}.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Combined model not found: {model_path}")
        model = CombinedPredictor.load(model_path)
        X_test, _ = build_feature_matrix(test_df, model.feature_columns)
        families = test_df["predicted_family"].tolist() if "predicted_family" in test_df.columns else None
        return model.predict(X_test, families=families)

    def predict_naive(fold: int, test_df: pd.DataFrame):
        model_path = NAIVE_DIR / f"fold_{fold}.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Naive model not found: {model_path}")
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return model.predict(test_df)

    def predict_curve(fold: int, test_df: pd.DataFrame):
        model_path = CURVE_DIR / f"fold_{fold}.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Curve model not found: {model_path}")
        with open(model_path, "rb") as f:
            payload = pickle.load(f)
        model: LGBMCurveFidelityModel = payload["model"]
        feature_cols = payload["feature_cols"]
        rungs = payload.get("rungs", DEFAULT_RUNGS)
        X_test, _ = build_feature_matrix(test_df, feature_cols)
        X_test = X_test.to_numpy()
        preds = model.predict(X_test, thresholds=rungs)

        # Convert curve predictions to threshold/runtime via score-optimal rule.
        # Use the smallest rung with predicted fidelity >= 0.99.
        thresholds = []
        for i in range(len(test_df)):
            thr = rungs[-1]
            for rung in rungs:
                if preds[rung][i] >= 0.99:
                    thr = rung
                    break
            thresholds.append(thr)
        thresholds = np.array(thresholds, dtype=int)

        # Use naive runtime lookup for compatibility (same as naive bucket model)
        # (Curve model predicts fidelity only.)
        naive_model_path = NAIVE_DIR / f"fold_{fold}.pkl"
        if not naive_model_path.exists():
            raise FileNotFoundError(f"Naive model not found for curve runtime: {naive_model_path}")
        with open(naive_model_path, "rb") as f:
            naive = pickle.load(f)
        _, runtime = naive.predict(test_df)
        return thresholds, runtime

    results = [
        evaluate_model("combined_lgbm", df, splits, predict_combined),
        evaluate_model("naive_bucket", df, splits, predict_naive),
        evaluate_model("lgbm_curve", df, splits, predict_curve),
    ]

    print("\nModel comparison (fixed CV splits):")
    for r in results:
        s = r["scores"]
        print(
            f"  {r['name']:<16} "
            f"thr_score={s['mean_threshold_score']:.4f} "
            f"thr_fail={s['n_threshold_failures']}/{s['n_tasks']} "
            f"rt_log={s['mean_runtime_log_score']:.4f}"
        )


if __name__ == "__main__":
    main()
