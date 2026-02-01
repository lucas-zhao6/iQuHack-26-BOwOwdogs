"""Runner that loads a saved CombinedPredictor from outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

from ..combined import CombinedPredictor
from ...data.data_prep import build_feature_matrix


class SavedCombinedRunner:
    name = "saved_combined"
    uses_saved_model = True

    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)

    def predict_from_df(self, df) -> Tuple[np.ndarray, np.ndarray]:
        model_path = self.model_dir / "full_model.pkl"
        model = CombinedPredictor.load(model_path)
        feature_cols = model.feature_columns
        X = build_feature_matrix(df, feature_cols)
        families = df["predicted_family"].tolist() if "predicted_family" in df.columns else None
        return model.predict(X, families=families)

    def predict_from_df_fold(self, df, fold_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        model_path = self.model_dir / f"fold_{fold_idx}.pkl"
        model = CombinedPredictor.load(model_path)
        feature_cols = model.feature_columns
        X = build_feature_matrix(df, feature_cols)
        families = df["predicted_family"].tolist() if "predicted_family" in df.columns else None
        return model.predict(X, families=families)
