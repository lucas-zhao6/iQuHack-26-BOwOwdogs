"""
Training data loader.

Loads hackathon_public.json and extracts structured training labels:
- selected_threshold (minimum threshold achieving >= TARGET_FIDELITY)
- forward_wall_s (10,000-shot forward run time)
- timing decomposition (setup + per-shot)
- threshold sweep curves (fidelity at each rung)
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd


THRESHOLD_RUNGS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
THRESHOLD_LOG2 = {t: int(math.log2(t)) for t in THRESHOLD_RUNGS}
TARGET_FIDELITY = 0.75


def load_training_data(json_path: str | Path) -> Dict[str, Any]:
    """Load the raw JSON training data."""
    path = Path(json_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    return data


def build_circuit_metadata(data: Dict[str, Any]) -> pd.DataFrame:
    """Extract circuit metadata into a DataFrame."""
    rows = []
    for c in data["circuits"]:
        rows.append({
            "file": c["file"],
            "family": c["family"],
            "n_qubits": c["n_qubits"],
            "source_name": c["source"]["name"],
        })
    return pd.DataFrame(rows)


def build_results_dataframe(data: Dict[str, Any]) -> pd.DataFrame:
    """
    Build a flat DataFrame from the results array.

    Each row = one (circuit, backend, precision) configuration with:
    - Training labels (threshold, runtime)
    - Threshold sweep curve data
    - Timing decomposition
    """
    rows = []
    for r in data["results"]:
        if r.get("status") != "ok":
            continue

        row: Dict[str, Any] = {
            "file": r["file"],
            "backend": r["backend"],
            "precision": r["precision"],
        }

        # Selection info (recomputed using TARGET_FIDELITY when sweep data is available)
        sel = r.get("selection", {})
        row["selected_threshold"] = sel.get("selected_threshold")
        row["selected_fidelity"] = sel.get("selected_mirror_metric_value")
        if row["selected_threshold"] is not None:
            row["selected_threshold_log2"] = int(math.log2(row["selected_threshold"]))
        else:
            row["selected_threshold_log2"] = None

        # Forward run
        fwd = r.get("forward", {})
        row["forward_wall_s"] = fwd.get("run_wall_s")
        row["forward_threshold"] = fwd.get("threshold")
        row["forward_unique_outcomes"] = fwd.get("unique_outcomes")
        row["forward_peak_rss_mb"] = fwd.get("peak_rss_mb")

        # Timing decomposition
        timing = r.get("forward_timing_estimates", {})
        row["estimated_per_shot_s"] = timing.get("estimated_per_shot_s")
        row["estimated_setup_s"] = timing.get("estimated_setup_s")

        # State setup (1-shot) timing
        setup = r.get("state_setup", {})
        row["state_setup_wall_s"] = setup.get("run_wall_s")
        row["state_setup_peak_rss_mb"] = setup.get("peak_rss_mb")

        # Threshold sweep curve: fidelity and runtime at each rung
        sweep = r.get("threshold_sweep", [])
        for rung in THRESHOLD_RUNGS:
            row[f"sweep_fid_{rung}"] = None
            row[f"sweep_wall_{rung}"] = None
            row[f"sweep_rss_{rung}"] = None

        for entry in sweep:
            thr = entry.get("threshold")
            if thr in THRESHOLD_RUNGS:
                fid = entry.get("sdk_get_fidelity")
                wall = entry.get("run_wall_s")
                rss = entry.get("peak_rss_mb")
                note = entry.get("note", "")
                if note and "timeout" in note.lower():
                    continue  # skip timed-out entries
                row[f"sweep_fid_{thr}"] = fid
                row[f"sweep_wall_{thr}"] = wall
                row[f"sweep_rss_{thr}"] = rss

        # Sweep-derived features
        fidelities = []
        for rung in THRESHOLD_RUNGS:
            f = row.get(f"sweep_fid_{rung}")
            if f is not None:
                fidelities.append((rung, f))

        if fidelities:
            row["sweep_min_fidelity"] = min(f for _, f in fidelities)
            row["sweep_max_fidelity"] = max(f for _, f in fidelities)
            row["n_sweep_rungs"] = len(fidelities)

            # Fidelity at threshold=1 (baseline difficulty)
            row["fidelity_at_1"] = fidelities[0][1] if fidelities[0][0] == 1 else None

            # Convergence rate: how many rungs to reach TARGET_FIDELITY
            crossed = [rung for rung, f in fidelities if f >= TARGET_FIDELITY]
            row["rungs_to_099"] = THRESHOLD_LOG2.get(crossed[0], 10) if crossed else 10

            # Biggest fidelity jump
            if len(fidelities) >= 2:
                jumps = [
                    (fidelities[i + 1][1] - fidelities[i][1], fidelities[i + 1][0])
                    for i in range(len(fidelities) - 1)
                ]
                biggest_jump = max(jumps, key=lambda x: x[0])
                row["biggest_fid_jump"] = biggest_jump[0]
                row["biggest_jump_rung"] = biggest_jump[1]
            else:
                row["biggest_fid_jump"] = 0.0
                row["biggest_jump_rung"] = fidelities[0][0]
        else:
            row["sweep_min_fidelity"] = None
            row["sweep_max_fidelity"] = None
            row["n_sweep_rungs"] = 0
            row["fidelity_at_1"] = None
            row["rungs_to_099"] = 10
            row["biggest_fid_jump"] = None
            row["biggest_jump_rung"] = None

        # Override selection threshold based on TARGET_FIDELITY when sweep is available
        if fidelities:
            crossed = [rung for rung, f in fidelities if f >= TARGET_FIDELITY]
            if crossed:
                selected_thr = crossed[0]
            else:
                selected_thr = fidelities[-1][0]
            row["selected_threshold"] = selected_thr
            row["selected_threshold_log2"] = int(math.log2(selected_thr))
            row["selected_fidelity"] = next(
                (f for rung, f in fidelities if rung == selected_thr), None
            )

        # Verify
        verify = r.get("verify", {})
        row["verify_p_return_zero"] = verify.get("p_return_zero") if verify else None

        rows.append(row)

    return pd.DataFrame(rows)


def load_holdout_tasks(json_path: str | Path) -> pd.DataFrame:
    """Load the holdout task definitions."""
    path = Path(json_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    tasks = data["tasks"]
    return pd.DataFrame(tasks).rename(columns={"id": "task_id"})
