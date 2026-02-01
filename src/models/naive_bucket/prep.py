#!/usr/bin/env python3
"""Prepare data and CV splits for NaiveBucket model."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.scripts.prep_features import main as prep_features

OUTPUT_DIR = PROJECT_ROOT / "outputs"
CV_SPLITS_PATH = OUTPUT_DIR / "cv_splits.json"


def main() -> None:
    prep_features()
    if not CV_SPLITS_PATH.exists():
        raise FileNotFoundError(
            f"CV splits not found: {CV_SPLITS_PATH}\n"
            "Run 'python src/evaluation/scripts/prep_cv_splits.py' first."
        )


if __name__ == "__main__":
    main()
