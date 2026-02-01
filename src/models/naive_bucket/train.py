#!/usr/bin/env python3
"""Train NaiveBucketModel folds using fixed CV splits."""

from __future__ import annotations

import sys
import pickle
from pathlib import Path

import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.cv_splits import load_cv_splits
from src.models.naive_model import NaiveBucketModel

OUTPUT_DIR = PROJECT_ROOT / "outputs"
FEATURES_PATH = OUTPUT_DIR / "training_features.pkl"
CV_SPLITS_PATH = OUTPUT_DIR / "cv_splits.json"
MODEL_DIR = OUTPUT_DIR / "naive_bucket"


def load_data() -> pd.DataFrame:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Feature file not found: {FEATURES_PATH}\n"
            "Run 'python run_phase1.py' first."
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

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for fold, split in enumerate(splits):
        train_df = df.iloc[split["train_idx"]]
        model = NaiveBucketModel()
        model.fit(train_df, threshold_col="selected_threshold", runtime_col="forward_wall_s")
        model_path = MODEL_DIR / f"fold_{fold}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print(f"Saved fold {fold} model: {model_path}")

    full_model = NaiveBucketModel()
    full_model.fit(df, threshold_col="selected_threshold", runtime_col="forward_wall_s")
    full_path = MODEL_DIR / "full_model.pkl"
    with open(full_path, "wb") as f:
        pickle.dump(full_model, f)
    print(f"Saved full model: {full_path}")


if __name__ == "__main__":
    main()
