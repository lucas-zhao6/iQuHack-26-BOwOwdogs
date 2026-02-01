#!/usr/bin/env python3
"""
Generate fixed CV splits once and save to outputs/cv_splits.json.

This ensures all models train/evaluate on identical withheld circuits.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.cv_splits import build_cv_splits, save_cv_splits

OUTPUT_DIR = PROJECT_ROOT / "outputs"
FEATURES_PATH = OUTPUT_DIR / "training_features.pkl"
CV_SPLITS_PATH = OUTPUT_DIR / "cv_splits.json"
VALID_SIZE = 0.2
N_SPLITS = 5
SEED = 42


def load_data() -> pd.DataFrame:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Feature file not found: {FEATURES_PATH}\n"
            "Run 'python src/data/scripts/prep_features.py' first."
        )
    return pd.read_pickle(FEATURES_PATH)


def main() -> None:
    df = load_data()

    valid = df["selected_threshold"].notna() & df["forward_wall_s"].notna()
    df = df[valid].reset_index(drop=True)

    groups = df["file"].values
    y_threshold = df["selected_threshold"].values.astype(float)
    n_groups = df["file"].nunique()
    n_splits = min(N_SPLITS, n_groups)

    splitter = GroupKFold(n_splits=n_splits)
    X_placeholder = np.zeros((len(df), 1), dtype=float)

    splits = build_cv_splits(
        X_placeholder,
        y_threshold,
        groups,
        splitter,
        valid_size=VALID_SIZE,
        seed=SEED,
    )

    metadata = {
        "splitter": "GroupKFold",
        "n_splits": n_splits,
        "actual_splits": len(splits),
        "valid_size": VALID_SIZE,
        "seed": SEED,
        "coverage_rule": "train_label_coverage",
    }
    save_cv_splits(CV_SPLITS_PATH, splits, n_samples=len(df), metadata=metadata)

    print(f"Saved CV splits: {CV_SPLITS_PATH}")
    print(f"  Rows: {len(df)} | Circuits: {n_groups} | Splits: {n_splits}")


if __name__ == "__main__":
    main()
