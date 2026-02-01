#!/usr/bin/env python3
"""Evaluate and compare pretrained threshold models on fixed CV splits."""

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

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.cv_splits import load_cv_splits
from src.data.data_prep import build_feature_matrix
from src.threshold_models.lgbm.predictor import ThresholdPredictor
from src.evaluation.scoring import compute_threshold_metrics

OUTPUT_DIR = PROJECT_ROOT / "outputs"
FEATURES_PATH = OUTPUT_DIR / "training_features.pkl"
CV_SPLITS_PATH = OUTPUT_DIR / "cv_splits.json"
LGBM_DIR = OUTPUT_DIR / "threshold_models" / "threshold_lgbm"
LGBM_CURVE_DIR = OUTPUT_DIR / "threshold_models" / "threshold_lgbm_curve"
NAIVE_DIR = OUTPUT_DIR / "threshold_models" / "threshold_naive"


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

    for fold, split in enumerate(splits):
        test_df = df.iloc[split["test_idx"]]
        pred_thr = predict_fold(fold, test_df)

        all_pred_thresholds.extend([int(v) for v in pred_thr])
        all_true_thresholds.extend([int(v) for v in test_df["selected_threshold"].values])

    scores = compute_threshold_metrics(all_pred_thresholds, all_true_thresholds)
    return {"name": name, "scores": scores}


def main() -> None:
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    if not CV_SPLITS_PATH.exists():
        raise FileNotFoundError(
            f"CV splits not found: {CV_SPLITS_PATH}\n"
            "Run 'python src/evaluation/scripts/prep_cv_splits.py' first."
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

    def predict_lgbm(fold: int, test_df: pd.DataFrame):
        model_path = LGBM_DIR / f"fold_{fold}.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"LGBM threshold model not found: {model_path}")
        model = ThresholdPredictor.load(model_path)
        X_test, _ = build_feature_matrix(test_df, model.feature_columns)
        X_test = X_test.to_numpy()
        families = test_df["predicted_family"].tolist() if "predicted_family" in test_df.columns else None
        return model.predict(X_test, families=families)

    def predict_naive(fold: int, test_df: pd.DataFrame):
        model_path = NAIVE_DIR / f"fold_{fold}.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Naive threshold model not found: {model_path}")
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return model.predict(test_df)

    def predict_lgbm_curve(fold: int, test_df: pd.DataFrame):
        model_path = LGBM_CURVE_DIR / f"fold_{fold}.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"LGBM curve threshold model not found: {model_path}")
        with open(model_path, "rb") as f:
            payload = pickle.load(f)
        model = payload["model"]
        feature_cols = payload["feature_cols"]
        rungs = payload.get("rungs", [])
        target_fidelity = payload.get("target_fidelity", 0.75)
        X_test, _ = build_feature_matrix(test_df, feature_cols)
        X_test = X_test.to_numpy()
        return model.predict_threshold(X_test, target_fidelity=target_fidelity, thresholds=rungs)

    results = [
        evaluate_model("threshold_lgbm", df, splits, predict_lgbm),
        evaluate_model("threshold_lgbm_curve", df, splits, predict_lgbm_curve),
        evaluate_model("threshold_naive", df, splits, predict_naive),
    ]

    print("\nThreshold model comparison (fixed CV splits):")
    for r in results:
        s = r["scores"]
        print(
            f"  {r['name']:<18} "
            f"thr_score={s['mean_threshold_score']:.4f} "
            f"thr_fail={s['n_threshold_failures']}/{s['n_tasks']}"
        )


if __name__ == "__main__":
    main()
