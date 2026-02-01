"""Model registry for comparison."""

from pathlib import Path

from .lgbm_simple import LGBMSimpleRunner
from .naive_bucket import NaiveBucketRunner
from .saved_combined import SavedCombinedRunner
from .saved_naive_bucket import SavedNaiveBucketRunner

OUTPUTS_DIR = Path(__file__).resolve().parents[2] / "outputs"
DEFAULT_LGBM_DIR = OUTPUTS_DIR / "combined_model"
DEFAULT_NAIVE_DIR = OUTPUTS_DIR / "naive_bucket"

MODEL_REGISTRY = {
    "lgbm_simple": LGBMSimpleRunner,
    "naive_bucket": NaiveBucketRunner,
    "saved_combined": lambda: SavedCombinedRunner(DEFAULT_LGBM_DIR),
    "saved_naive_bucket": lambda: SavedNaiveBucketRunner(DEFAULT_NAIVE_DIR),
}
