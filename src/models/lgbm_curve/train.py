#!/usr/bin/env python3
"""Train the LGBM curve-fit fidelity model using fixed CV splits."""

from __future__ import annotations

import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_prep import build_feature_matrix
from src.evaluation.cv_splits import load_cv_splits
from src.models.lgbm_curve import LGBMCurveFidelityModel, DEFAULT_RUNGS
from src.models.lgbm.curve_predictor import extract_curve_targets

OUTPUT_DIR = PROJECT_ROOT / "outputs"
FEATURES_PATH = OUTPUT_DIR / "training_features.pkl"
CV_SPLITS_PATH = OUTPUT_DIR / "cv_splits.json"
MODEL_DIR = OUTPUT_DIR / "lgbm_curve"


def load_data() -> pd.DataFrame:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Feature file not found: {FEATURES_PATH}\n"
            "Run 'python src/data/scripts/prep_features.py' first."
        )
    return pd.read_pickle(FEATURES_PATH)


def main() -> None:
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

    X, feature_cols = build_feature_matrix(df)
    X = X.to_numpy()
    y_curves = extract_curve_targets(df, rungs=DEFAULT_RUNGS)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for fold, split in enumerate(splits):
        train_idx = split["train_idx"]
        test_idx = split["test_idx"]

        model = LGBMCurveFidelityModel()
        model.fit(X[train_idx], {r: y_curves[r][train_idx] for r in y_curves})

        model_path = MODEL_DIR / f"fold_{fold}.pkl"
        payload = {
            "model": model,
            "feature_cols": feature_cols,
            "rungs": DEFAULT_RUNGS,
        }
        with model_path.open("wb") as f:
            pickle.dump(payload, f)
        print(f"Saved fold {fold} model: {model_path}")

    full_model = LGBMCurveFidelityModel()
    full_model.fit(X, y_curves)

    full_path = MODEL_DIR / "full_model.pkl"
    payload = {
        "model": full_model,
        "feature_cols": feature_cols,
        "rungs": DEFAULT_RUNGS,
    }
    with full_path.open("wb") as f:
        pickle.dump(payload, f)
    print(f"Saved full model: {full_path}")


if __name__ == "__main__":
    main()
