"""
Innovation 5: Threshold Sweep Curve Shape Analysis

Extracts features from the fidelity-threshold curve that characterize
how a circuit responds to increasing MPS bond dimension.

Key insights:
- Circuits that converge quickly (high fidelity at low thresholds) are "easy"
- Circuits that plateau below 0.99 may need very high thresholds
- The shape of the curve reveals entanglement structure

These features are extracted from TRAINING data only and used to
train a model that can predict curve shape from circuit features.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd


# Threshold rungs in order
THRESHOLD_RUNGS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]


def extract_sweep_curve(row: pd.Series) -> Tuple[List[int], List[float]]:
    """
    Extract the fidelity-threshold curve from a data row.

    Returns:
        rungs: List of threshold values [1, 2, 4, ..., 512]
        fidelities: List of corresponding fidelity values
    """
    rungs = []
    fidelities = []

    for rung in THRESHOLD_RUNGS:
        col = f"sweep_fid_{rung}"
        if col in row.index:
            fid = row[col]
            if pd.notna(fid) and np.isfinite(fid):
                rungs.append(rung)
                fidelities.append(float(fid))

    return rungs, fidelities


def compute_convergence_rate(rungs: List[int], fidelities: List[float]) -> float:
    """
    Compute how quickly fidelity converges to 1.0.

    Uses the slope in log-threshold space: d(fidelity) / d(log2(threshold))
    Higher values = faster convergence = easier circuit.
    """
    if len(rungs) < 2:
        return 0.0

    # Convert to log2 scale for threshold
    log_rungs = np.log2(rungs)
    fids = np.array(fidelities)

    # Find the steepest slope (maximum improvement rate)
    max_slope = 0.0
    for i in range(len(rungs) - 1):
        if log_rungs[i+1] != log_rungs[i]:
            slope = (fids[i+1] - fids[i]) / (log_rungs[i+1] - log_rungs[i])
            max_slope = max(max_slope, slope)

    return max_slope


def compute_area_under_curve(rungs: List[int], fidelities: List[float]) -> float:
    """
    Compute area under the fidelity curve (in log-threshold space).

    Higher area = better fidelity across all thresholds = easier circuit.
    Normalized to [0, 1] range.
    """
    if len(rungs) < 2:
        return 0.0

    # Use log2 scale for x-axis
    log_rungs = np.log2(rungs)
    fids = np.array(fidelities)

    # Trapezoidal integration
    area = np.trapz(fids, log_rungs)

    # Normalize by maximum possible area (fidelity=1 everywhere)
    max_area = log_rungs[-1] - log_rungs[0]
    if max_area > 0:
        return area / max_area
    return 0.0


def detect_plateau(rungs: List[int], fidelities: List[float],
                   plateau_threshold: float = 0.01) -> Tuple[bool, float, int]:
    """
    Detect if the fidelity curve plateaus before reaching 0.99.

    A plateau is detected when consecutive improvements are < plateau_threshold.

    Returns:
        has_plateau: True if curve plateaus below 0.99
        plateau_fidelity: The fidelity value where it plateaus (or max fidelity)
        plateau_rung: The threshold rung where plateau starts
    """
    if len(rungs) < 2:
        return False, 1.0, 1

    fids = np.array(fidelities)

    # Find where improvements become negligible
    for i in range(len(fids) - 1):
        improvement = fids[i+1] - fids[i]
        if improvement < plateau_threshold and fids[i] < 0.99:
            # Check if it stays plateaued
            remaining_improvements = np.diff(fids[i:])
            if np.all(remaining_improvements < plateau_threshold):
                return True, float(fids[i]), rungs[i]

    return False, float(fids[-1]), rungs[-1]


def compute_fidelity_variance(fidelities: List[float]) -> float:
    """
    Compute variance of fidelity values across rungs.

    Low variance = consistent fidelity (either always good or always bad)
    High variance = fidelity changes significantly with threshold
    """
    if len(fidelities) < 2:
        return 0.0
    return float(np.var(fidelities))


def compute_threshold_sensitivity(rungs: List[int], fidelities: List[float]) -> float:
    """
    Compute how sensitive the circuit is to threshold changes.

    Measured as the ratio of max fidelity jump to total fidelity range.
    High sensitivity = one critical threshold rung
    Low sensitivity = gradual improvement
    """
    if len(fidelities) < 2:
        return 0.0

    fids = np.array(fidelities)
    jumps = np.diff(fids)

    total_range = fids[-1] - fids[0]
    if total_range <= 0:
        return 0.0

    max_jump = np.max(jumps) if len(jumps) > 0 else 0.0
    return max_jump / total_range


def compute_early_fidelity(fidelities: List[float], n_early: int = 3) -> float:
    """
    Compute average fidelity at early (low) threshold rungs.

    High early fidelity = circuit is easy for MPS
    Low early fidelity = circuit needs high threshold
    """
    if len(fidelities) == 0:
        return 0.0

    early = fidelities[:n_early]
    return float(np.mean(early))


def compute_late_fidelity(fidelities: List[float], n_late: int = 3) -> float:
    """
    Compute average fidelity at late (high) threshold rungs.

    Should be close to 1.0 for most circuits.
    Low late fidelity = circuit may never converge (anomaly)
    """
    if len(fidelities) == 0:
        return 0.0

    late = fidelities[-n_late:]
    return float(np.mean(late))


def extract_sweep_features(row: pd.Series) -> Dict[str, float]:
    """
    Extract all sweep curve shape features from a data row.

    Returns dict of feature_name -> value.
    """
    rungs, fidelities = extract_sweep_curve(row)

    if len(rungs) == 0:
        # Return default values if no sweep data
        return {
            "sweep_convergence_rate": 0.0,
            "sweep_area_under_curve": 0.0,
            "sweep_has_plateau": 0.0,
            "sweep_plateau_fidelity": 1.0,
            "sweep_plateau_rung_log2": 0.0,
            "sweep_fidelity_variance": 0.0,
            "sweep_threshold_sensitivity": 0.0,
            "sweep_early_fidelity": 0.0,
            "sweep_late_fidelity": 1.0,
            "sweep_fidelity_range": 0.0,
            "sweep_n_rungs_below_05": 0.0,
            "sweep_n_rungs_below_09": 0.0,
            "sweep_n_rungs_below_099": 0.0,
        }

    # Basic stats
    fids = np.array(fidelities)
    fidelity_range = float(fids[-1] - fids[0]) if len(fids) > 0 else 0.0

    # Convergence and area
    convergence_rate = compute_convergence_rate(rungs, fidelities)
    area_under_curve = compute_area_under_curve(rungs, fidelities)

    # Plateau detection
    has_plateau, plateau_fid, plateau_rung = detect_plateau(rungs, fidelities)

    # Other features
    fid_variance = compute_fidelity_variance(fidelities)
    threshold_sensitivity = compute_threshold_sensitivity(rungs, fidelities)
    early_fidelity = compute_early_fidelity(fidelities)
    late_fidelity = compute_late_fidelity(fidelities)

    # Count rungs below thresholds
    n_below_05 = sum(1 for f in fidelities if f < 0.5)
    n_below_09 = sum(1 for f in fidelities if f < 0.9)
    n_below_099 = sum(1 for f in fidelities if f < 0.99)

    return {
        "sweep_convergence_rate": convergence_rate,
        "sweep_area_under_curve": area_under_curve,
        "sweep_has_plateau": float(has_plateau),
        "sweep_plateau_fidelity": plateau_fid,
        "sweep_plateau_rung_log2": np.log2(plateau_rung) if plateau_rung > 0 else 0.0,
        "sweep_fidelity_variance": fid_variance,
        "sweep_threshold_sensitivity": threshold_sensitivity,
        "sweep_early_fidelity": early_fidelity,
        "sweep_late_fidelity": late_fidelity,
        "sweep_fidelity_range": fidelity_range,
        "sweep_n_rungs_below_05": float(n_below_05),
        "sweep_n_rungs_below_09": float(n_below_09),
        "sweep_n_rungs_below_099": float(n_below_099),
    }


def add_sweep_features_to_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add sweep curve shape features to an existing dataframe.

    Args:
        df: DataFrame with sweep_fid_* columns

    Returns:
        DataFrame with additional sweep feature columns
    """
    # Extract features for each row
    sweep_features = df.apply(extract_sweep_features, axis=1)

    # Convert to DataFrame
    sweep_df = pd.DataFrame(sweep_features.tolist(), index=df.index)

    # Combine with original
    result = pd.concat([df, sweep_df], axis=1)

    return result


def get_sweep_feature_names() -> List[str]:
    """Return list of sweep feature column names."""
    return [
        "sweep_convergence_rate",
        "sweep_area_under_curve",
        "sweep_has_plateau",
        "sweep_plateau_fidelity",
        "sweep_plateau_rung_log2",
        "sweep_fidelity_variance",
        "sweep_threshold_sensitivity",
        "sweep_early_fidelity",
        "sweep_late_fidelity",
        "sweep_fidelity_range",
        "sweep_n_rungs_below_05",
        "sweep_n_rungs_below_09",
        "sweep_n_rungs_below_099",
    ]
