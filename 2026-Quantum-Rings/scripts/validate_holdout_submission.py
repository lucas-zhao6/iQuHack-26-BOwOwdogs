#!/usr/bin/env python3
"""
validate_holdout_submission.py

Validate and normalize a submission for the Circuit Fingerprint Challenge.

Hackathon-facing labels:
- processor: CPU or GPU
- precision: single or double

This script:
- checks all required task IDs are present exactly once
- enforces threshold rung choices
- enforces finite positive runtime
- writes a canonical normalized submission JSON

Usage:
  python scripts/validate_holdout_submission.py \
    --public data/holdout_public.json \
    --submission my_predictions.json \
    --write-normalized my_predictions.normalized.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Any


THRESHOLD_RUNGS = [1, 2, 4, 8, 16, 32, 64, 128, 256]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def normalize_predictions(obj: Any) -> List[Dict[str, Any]]:
    if isinstance(obj, list):
        preds = obj
    elif isinstance(obj, dict) and "predictions" in obj:
        preds = obj["predictions"]
    else:
        raise ValueError("Submission must be a list or an object with key 'predictions'.")

    if not isinstance(preds, list):
        raise ValueError("'predictions' must be a list.")

    normalized = []
    for p in preds:
        if not isinstance(p, dict):
            raise ValueError("Each prediction must be an object.")

        if "id" not in p:
            raise ValueError("Each prediction must include 'id'.")

        if "predicted_threshold_min" not in p:
            raise ValueError(f"Missing predicted_threshold_min for id={p.get('id')}")

        if "predicted_forward_wall_s" not in p:
            raise ValueError(f"Missing predicted_forward_wall_s for id={p.get('id')}")

        thr = p["predicted_threshold_min"]
        t = p["predicted_forward_wall_s"]

        if not isinstance(thr, int):
            raise ValueError(f"predicted_threshold_min must be int for id={p['id']}")

        if thr not in THRESHOLD_RUNGS:
            raise ValueError(
                f"id={p['id']}: predicted_threshold_min={thr} "
                f"is not one of {THRESHOLD_RUNGS}"
            )

        if not isinstance(t, (int, float)) or not math.isfinite(t) or t <= 0:
            raise ValueError(
                f"id={p['id']}: predicted_forward_wall_s must be a finite positive number"
            )

        normalized.append(
            {
                "id": p["id"],
                "predicted_threshold_min": int(thr),
                "predicted_forward_wall_s": float(t),
            }
        )

    return normalized


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--public", required=True, help="holdout_public.json")
    ap.add_argument("--submission", required=True, help="raw submission JSON")
    ap.add_argument("--write-normalized", help="output normalized JSON")
    args = ap.parse_args()

    public = load_json(Path(args.public))
    expected_ids = {task["id"] for task in public["tasks"]}

    submission_raw = load_json(Path(args.submission))
    preds = normalize_predictions(submission_raw)

    pred_ids = [p["id"] for p in preds]
    pred_id_set = set(pred_ids)

    print(f"Expected tasks: {len(expected_ids)}")
    print(f"Valid predictions parsed: {len(preds)}")
    print(f"Unique ids in submission: {len(pred_id_set)}")

    if len(pred_ids) != len(pred_id_set):
        raise ValueError("Duplicate task IDs found in submission.")

    missing = expected_ids - pred_id_set
    extra = pred_id_set - expected_ids

    if missing:
        raise ValueError(f"Missing predictions for task IDs: {sorted(missing)}")

    if extra:
        raise ValueError(f"Submission contains unknown task IDs: {sorted(extra)}")

    if args.write_normalized:
        out = {
            "predictions": sorted(preds, key=lambda x: x["id"])
        }
        save_json(Path(args.write_normalized), out)
        print(f"\nWrote normalized: {args.write_normalized}")


if __name__ == "__main__":
    main()

