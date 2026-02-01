#!/usr/bin/env python3
"""
Single-circuit prediction CLI.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd

from src.data.feature_pipeline import extract_holdout_circuit_features
from src.data.data_prep import build_feature_matrix
from src.threshold_models.lgbm.predictor import ThresholdPredictor
from src.runtime_models.gpr.predictor import GPRRuntimePredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict threshold or runtime for a single circuit.")
    parser.add_argument("--mode", choices=["threshold", "runtime"], required=True)
    parser.add_argument("--qasm", required=True, help="Path to QASM file.")
    parser.add_argument("--backend", choices=["CPU", "GPU"], required=True)
    parser.add_argument("--precision", choices=["single", "double"], required=True)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Threshold rung (required for runtime mode).",
    )
    parser.add_argument(
        "--artifacts",
        default="outputs",
        help="Root directory containing trained model artifacts.",
    )
    return parser.parse_args()


def build_feature_row(qasm_path: Path, backend: str, precision: str) -> pd.DataFrame:
    feats: Dict[str, Any] = extract_holdout_circuit_features(qasm_path)
    feats["backend"] = backend
    feats["precision"] = precision
    return pd.DataFrame([feats])


def main() -> None:
    args = parse_args()
    qasm_path = Path(args.qasm)
    if not qasm_path.exists():
        raise FileNotFoundError(f"Missing QASM file: {qasm_path}")

    artifacts_dir = Path(args.artifacts)
    threshold_path = artifacts_dir / "threshold_models" / "threshold_lgbm" / "full_model.pkl"
    runtime_path = artifacts_dir / "runtime_models" / "runtime_gpr" / "full_model.pkl"

    if args.mode == "threshold":
        if not threshold_path.exists():
            raise FileNotFoundError(f"Threshold model not found: {threshold_path}")
        threshold_predictor = ThresholdPredictor.load(threshold_path)

        df = build_feature_row(qasm_path, args.backend, args.precision)
        X_thr, _ = build_feature_matrix(df, threshold_predictor.feature_columns)
        families = df["predicted_family"].tolist() if "predicted_family" in df.columns else None
        pred = threshold_predictor.predict(X_thr.to_numpy(), families=families)[0]
        print(int(pred))
        return

    if args.threshold is None:
        raise ValueError("--threshold is required for runtime mode.")
    if not runtime_path.exists():
        raise FileNotFoundError(f"Runtime model not found: {runtime_path}")

    runtime_predictor = GPRRuntimePredictor.load(runtime_path)
    df = build_feature_row(qasm_path, args.backend, args.precision)
    X_rt, _ = build_feature_matrix(df, runtime_predictor.feature_columns)
    pred = runtime_predictor.predict(X_rt.to_numpy(), np.array([args.threshold], dtype=float))[0]
    print(float(pred))


if __name__ == "__main__":
    main()
