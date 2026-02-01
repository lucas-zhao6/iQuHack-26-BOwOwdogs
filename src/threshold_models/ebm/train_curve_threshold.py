#!/usr/bin/env python3
"""
Train the EBM curve-based threshold model.
"""

from __future__ import annotations

import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.threshold_models.features import get_feature_columns
from src.evaluation.cv_splits import load_cv_splits
from src.evaluation.scoring import THRESHOLD_RUNGS
from src.threshold_models.ebm.curve_threshold_model import (
    EBMCurveThresholdModel,
    TARGET_FIDELITY,
)

OUTPUT_DIR = PROJECT_ROOT / "outputs"
FEATURES_PATH = OUTPUT_DIR / "training_features.pkl"
CV_SPLITS_PATH = OUTPUT_DIR / "cv_splits.json"
MODEL_DIR = OUTPUT_DIR / "threshold_models" / "threshold_ebm_curve"


def load_data() -> pd.DataFrame:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Feature file not found: {FEATURES_PATH}\n"
            "Run 'python src/data/scripts/prep_features.py' first."
        )
    return pd.read_pickle(FEATURES_PATH)


def build_feature_matrix(df: pd.DataFrame):
    feature_cols = get_feature_columns(df.columns.tolist())
    valid_cols = []
    for col in feature_cols:
        if col in df.columns:
            series = df[col]
            if series.dtype in [np.float64, np.int64, float, int]:
                if series.notna().sum() > 0 and series.std() > 1e-10:
                    valid_cols.append(col)
    feature_cols = valid_cols

    X = df[feature_cols].copy()
    X["is_gpu"] = (df["backend"] == "GPU").astype(float)
    X["is_double"] = (df["precision"] == "double").astype(float)
    feature_cols = feature_cols + ["is_gpu", "is_double"]

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    return X.values, feature_cols


def extract_curve_targets(df: pd.DataFrame) -> dict[int, np.ndarray]:
    curves = {}
    for rung in THRESHOLD_RUNGS:
        col = f"sweep_fid_{rung}"
        if col in df.columns:
            curves[rung] = np.asarray(df[col].values.astype(float))
    return curves


def main() -> None:
    if not CV_SPLITS_PATH.exists():
        raise FileNotFoundError(
            f"CV splits not found: {CV_SPLITS_PATH}\n"
            "Run 'python src/evaluation/scripts/prep_cv_splits.py' first."
        )

    print("Loading training data...")
    df = load_data()
    valid = df["selected_threshold"].notna() & df["forward_wall_s"].notna()
    df = df[valid].reset_index(drop=True)

    print("Loading CV splits...")
    splits, meta = load_cv_splits(CV_SPLITS_PATH)
    if meta.get("n_samples") is not None and meta["n_samples"] != len(df):
        raise ValueError(
            "CV splits do not match current dataset size. "
            f"Splits have n_samples={meta['n_samples']} but data has {len(df)} rows."
        )

    print("Building feature matrix...")
    X, feature_cols = build_feature_matrix(df)
    X = np.asarray(X)
    y_curves = extract_curve_targets(df)
    print(
        f"Features: {len(feature_cols)} | Samples: {len(df)} | "
        f"Rungs: {len(y_curves)}"
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for fold, split in enumerate(splits):
        print(f"\nTraining fold {fold}...")
        train_idx = split["train_idx"]

        model = EBMCurveThresholdModel()
        model.fit(
            np.asarray(X[train_idx]),
            {r: np.asarray(y_curves[r][train_idx]) for r in y_curves},
        )

        model_path = MODEL_DIR / f"fold_{fold}.pkl"
        payload = {
            "model": model,
            "feature_cols": feature_cols,
            "rungs": THRESHOLD_RUNGS,
            "target_fidelity": TARGET_FIDELITY,
        }
        with model_path.open("wb") as f:
            pickle.dump(payload, f)
        print(f"Saved fold {fold} model: {model_path}")

    print("\nTraining full model...")
    full_model = EBMCurveThresholdModel()
    full_model.fit(np.asarray(X), {r: np.asarray(y_curves[r]) for r in y_curves})

    full_path = MODEL_DIR / "full_model.pkl"
    payload = {
        "model": full_model,
        "feature_cols": feature_cols,
        "rungs": THRESHOLD_RUNGS,
        "target_fidelity": TARGET_FIDELITY,
    }
    with full_path.open("wb") as f:
        pickle.dump(payload, f)
    print(f"Saved full model: {full_path}")


if __name__ == "__main__":
    main()
