#!/usr/bin/env python3
"""Prepare data and CV splits for LGBM curve-fit fidelity model."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.scripts.prep_features import main as prep_features
from src.evaluation.scripts.prep_cv_splits import main as prep_splits


def main() -> None:
    prep_features()
    prep_splits()


if __name__ == "__main__":
    main()
