"""Runner that loads saved NaiveBucketModel folds."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pickle

from ..naive_bucket.model import NaiveBucketModel


class SavedNaiveBucketRunner:
    name = "saved_naive_bucket"
    uses_saved_model = True

    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)

    def _load(self, path: Path) -> NaiveBucketModel:
        with open(path, "rb") as f:
            return pickle.load(f)

    def predict_from_df(self, df) -> Tuple[np.ndarray, np.ndarray]:
        model_path = self.model_dir / "full_model.pkl"
        model = self._load(model_path)
        return model.predict(df)

    def predict_from_df_fold(self, df, fold_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        model_path = self.model_dir / f"fold_{fold_idx}.pkl"
        model = self._load(model_path)
        return model.predict(df)
