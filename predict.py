#!/usr/bin/env python3
"""
Generate holdout predictions using pretrained models.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd

from src.data.data_loader import load_holdout_tasks
from src.data.feature_pipeline import extract_holdout_circuit_features
from src.data.data_prep import build_feature_matrix
from src.threshold_models.lgbm.predictor import ThresholdPredictor
from src.runtime_models.lgbm.predictor_uncertainty import RuntimePredictorWithThresholdUncertainty


def load_id_map(path: Path) -> Dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("entries", [])
    mapping = {}
    for entry in entries:
        task_id = entry.get("id")
        qasm_file = entry.get("qasm_file")
        if task_id and qasm_file:
            mapping[task_id] = qasm_file
    return mapping


def build_holdout_dataframe(
    tasks_df: pd.DataFrame,
    id_map: Dict[str, str],
    circuits_dir: Path,
    allow_missing_ids: bool = False,
) -> pd.DataFrame:
    rows: list[Dict[str, Any]] = []
    for _, task in tasks_df.iterrows():
        task_id = task["task_id"]
        qasm_name = id_map.get(task_id)
        if qasm_name is None:
            if allow_missing_ids:
                continue
            raise KeyError(f"Task id {task_id} missing from id map.")

        qasm_path = circuits_dir / qasm_name
        if not qasm_path.exists():
            raise FileNotFoundError(f"Missing QASM file: {qasm_path}")

        feats = extract_holdout_circuit_features(qasm_path)
        feats["backend"] = task["processor"]
        feats["precision"] = task["precision"]
        feats["task_id"] = task_id
        rows.append(feats)

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate holdout predictions.")
    parser.add_argument("--tasks", required=True, help="Path to holdout tasks JSON.")
    parser.add_argument("--circuits", required=True, help="Directory of holdout QASM files.")
    parser.add_argument("--id-map", required=True, help="Path to holdout id map JSON.")
    parser.add_argument("--out", required=True, help="Output path for predictions JSON.")
    parser.add_argument(
        "--artifacts",
        default="outputs",
        help="Root directory containing trained model artifacts.",
    )
    parser.add_argument(
        "--allow-missing-ids",
        action="store_true",
        help="Skip tasks missing from the id map instead of raising an error.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tasks_path = Path(args.tasks)
    circuits_dir = Path(args.circuits)
    id_map_path = Path(args.id_map)
    out_path = Path(args.out)
    artifacts_dir = Path(args.artifacts)

    tasks_df = load_holdout_tasks(tasks_path)
    id_map = load_id_map(id_map_path)

    df = build_holdout_dataframe(
        tasks_df,
        id_map,
        circuits_dir,
        allow_missing_ids=args.allow_missing_ids,
    )

    threshold_path = artifacts_dir / "threshold_models" / "threshold_lgbm" / "full_model.pkl"
    runtime_path = (
        artifacts_dir / "runtime_models" / "runtime_lgbm_uncertainty" / "full_model.pkl"
    )

    if not threshold_path.exists():
        raise FileNotFoundError(f"Threshold model not found: {threshold_path}")
    if not runtime_path.exists():
        raise FileNotFoundError(f"Runtime model not found: {runtime_path}")

    threshold_predictor = ThresholdPredictor.load(threshold_path)
    runtime_predictor = RuntimePredictorWithThresholdUncertainty.load(runtime_path)
    runtime_predictor.threshold_predictor = threshold_predictor

    X_thr, _ = build_feature_matrix(df, threshold_predictor.feature_columns)
    X_rt, _ = build_feature_matrix(df, runtime_predictor.feature_columns)

    families = df["predicted_family"].tolist() if "predicted_family" in df.columns else None
    pred_thresholds = threshold_predictor.predict(X_thr.to_numpy(), families=families)
    pred_runtime = runtime_predictor.predict(X_rt.to_numpy(), pred_thresholds)

    predictions = []
    for task_id, thr, rt in zip(df["task_id"].tolist(), pred_thresholds, pred_runtime):
        predictions.append(
            {
                "id": str(task_id),
                "predicted_threshold_min": int(thr),
                "predicted_forward_wall_s": float(rt) if np.isfinite(rt) else float("nan"),
            }
        )

    out_payload = {"predictions": predictions}
    out_path.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
