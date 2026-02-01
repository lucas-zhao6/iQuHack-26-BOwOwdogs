"""
Predictor wrapper for EBM curve-based threshold modeling.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterable, List

import numpy as np

from src.evaluation.scoring import THRESHOLD_RUNGS
from .curve_threshold_model import EBMCurveThresholdModel, TARGET_FIDELITY


class EBMCurveThresholdPredictor:
    """Wraps a curve model with metadata for threshold prediction."""

    def __init__(
        self,
        model: EBMCurveThresholdModel,
        feature_columns: List[str],
        rungs: Iterable[int] = THRESHOLD_RUNGS,
        target_fidelity: float = TARGET_FIDELITY,
    ) -> None:
        self.model = model
        self.feature_columns = list(feature_columns)
        self.rungs = list(rungs)
        self.target_fidelity = float(target_fidelity)

    def predict_threshold(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_threshold(
            np.asarray(X),
            target_fidelity=self.target_fidelity,
            thresholds=self.rungs,
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        with path.open("wb") as f:
            pickle.dump(
                {
                    "model": self.model,
                    "feature_cols": self.feature_columns,
                    "rungs": self.rungs,
                    "target_fidelity": self.target_fidelity,
                },
                f,
            )

    @classmethod
    def load(cls, path: str | Path) -> "EBMCurveThresholdPredictor":
        path = Path(path)
        with path.open("rb") as f:
            payload = pickle.load(f)
        if isinstance(payload, cls):
            return payload
        model = payload["model"]
        feature_cols = payload.get("feature_cols", [])
        rungs = payload.get("rungs", THRESHOLD_RUNGS)
        target_fidelity = payload.get("target_fidelity", TARGET_FIDELITY)
        return cls(
            model=model,
            feature_columns=feature_cols,
            rungs=rungs,
            target_fidelity=target_fidelity,
        )
