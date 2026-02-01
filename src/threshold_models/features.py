"""
Shared feature column selection for LGBM models.
"""

from __future__ import annotations

from typing import List


def get_feature_columns(df_columns: List[str]) -> List[str]:
    """
    Select feature columns suitable for model training.

    Excludes:
    - Target columns (threshold, runtime, fidelity)
    - Metadata columns (file, backend, precision, family)
    - Sweep columns (per-rung data)
    """
    exclude_patterns = [
        "file", "backend", "precision", "family", "predicted_family", "true_family",
        "selected_threshold", "selected_fidelity", "selected_threshold_log2",
        "forward_wall_s", "forward_threshold", "forward_unique_outcomes", "forward_peak_rss_mb",
        "estimated_per_shot_s", "estimated_setup_s",
        "state_setup_wall_s", "state_setup_peak_rss_mb",
        "log_forward_wall_s", "log_setup_s", "log_per_shot_s",
        "is_gpu", "is_double",
        "verify_p_return_zero",
        "sweep_min_fidelity", "sweep_max_fidelity", "n_sweep_rungs",
        "fidelity_at_1", "rungs_to_099", "biggest_fid_jump", "biggest_jump_rung",
        "sweep_fid_", "sweep_wall_", "sweep_rss_",
    ]

    feature_cols = []
    for col in df_columns:
        exclude = False
        for pattern in exclude_patterns:
            if pattern in col:
                exclude = True
                break
        if not exclude:
            feature_cols.append(col)

    return feature_cols
