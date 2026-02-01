import tempfile
import unittest

import numpy as np

try:
    from interpret.glassbox import ExplainableBoostingRegressor  # noqa: F401
except Exception:  # pragma: no cover
    ExplainableBoostingRegressor = None

from src.threshold_models.ebm.curve_threshold_model import (
    EBMCurveThresholdModel,
    EBMConfig,
    TARGET_FIDELITY,
)
from src.evaluation.scoring import THRESHOLD_RUNGS


@unittest.skipIf(ExplainableBoostingRegressor is None, "interpret not installed")
class TestEBMCurveThresholdModel(unittest.TestCase):
    def _make_data(self, n_samples: int = 40):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(n_samples, 5))
        X[:, 3] = (rng.random(n_samples) > 0.5).astype(float)  # b1
        X[:, 4] = (rng.random(n_samples) > 0.5).astype(float)  # b2

        base = 1 / (1 + np.exp(-X[:, 0]))
        y_curves = {}
        for rung in THRESHOLD_RUNGS:
            rung_idx = int(round(np.log2(float(rung))))
            y_curves[rung] = np.clip(base + 0.04 * rung_idx, 0.0, 1.0)
        return X, y_curves

    def test_predict_curve_length_and_monotone(self):
        X, y_curves = self._make_data()
        feature_names = [f"f{i}" for i in range(X.shape[1])]
        feature_names[3] = "b1"
        feature_names[4] = "b2"

        config = EBMConfig(
            ebm_params={"max_rounds": 50, "interactions": 0},
            binary_feature_names=("b1", "b2"),
        )
        model = EBMCurveThresholdModel(config=config)
        model.fit(X, y_curves, feature_names=feature_names)

        preds = model.predict_curve(X[0], b1_value=1.0, b2_value=0.0)
        self.assertEqual(len(preds), 9)
        self.assertTrue(np.all(np.diff(preds) >= -1e-9))

    def test_save_load_roundtrip(self):
        X, y_curves = self._make_data()
        feature_names = [f"f{i}" for i in range(X.shape[1])]
        feature_names[3] = "b1"
        feature_names[4] = "b2"

        config = EBMConfig(
            ebm_params={"max_rounds": 50, "interactions": 0},
            binary_feature_names=("b1", "b2"),
        )
        model = EBMCurveThresholdModel(config=config)
        model.fit(X, y_curves, feature_names=feature_names)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/model.pkl"
            model.save(path)
            loaded = EBMCurveThresholdModel.load(path)

            preds_a = model.predict_curve(X[1], b1_value=0.0, b2_value=1.0)
            preds_b = loaded.predict_curve(X[1], b1_value=0.0, b2_value=1.0)
            np.testing.assert_allclose(preds_a, preds_b, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
