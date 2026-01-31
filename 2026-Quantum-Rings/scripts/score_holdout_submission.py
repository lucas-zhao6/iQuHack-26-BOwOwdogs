#!/usr/bin/env python3
"""
score_holdout_submission.py

Organizer-only scoring script for the Circuit Fingerprint Challenge.

Scoring model (Model 1A):
- Missing prediction -> task_score = 0
- predicted_threshold < true_threshold -> task_score = 0 (fidelity violated)
- Otherwise:
    threshold_score = 2^(-steps_over)
    runtime_score = min(r, 1/r), r = pred_time / true_time
    task_score = threshold_score * runtime_score

Overall score = mean(task_score over all tasks)

Hackathon-facing labels:
- processor: CPU / GPU
- precision: single / double
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Any


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def rung_index(thr: int) -> int:
    return int(math.log2(thr))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", required=True, help="private holdout_truth.json")
    ap.add_argument("--public", required=True, help="holdout_public.json")
    ap.add_argument("--submission", required=True, help="normalized submission JSON")
    ap.add_argument("--out", required=True, help="output score report JSON")
    ap.add_argument("--topk", type=int, default=10, help="print top/bottom K tasks")
    args = ap.parse_args()

    truth = load_json(Path(args.truth))
    public = load_json(Path(args.public))
    submission = load_json(Path(args.submission))

    truth_map = {t["id"]: t for t in truth["labels"]}
    tasks = public["tasks"]
    preds = {p["id"]: p for p in submission["predictions"]}

    scores = []
    missing = 0

    for task in tasks:
        tid = task["id"]

        if tid not in preds:
            missing += 1
            scores.append((tid, 0.0, 0.0, 0.0))
            continue

        pred = preds[tid]
        true = truth_map[tid]

        pred_thr = pred["predicted_threshold_min"]
        true_thr = true["true_threshold_min"]

        # Fidelity constraint
        if pred_thr < true_thr:
            scores.append((tid, 0.0, 0.0, 0.0))
            continue

        steps_over = rung_index(pred_thr) - rung_index(true_thr)
        threshold_score = 2.0 ** (-steps_over)

        pred_t = pred["predicted_forward_wall_s"]
        true_t = true["true_forward_wall_s"]

        r = pred_t / true_t
        runtime_score = min(r, 1.0 / r)

        task_score = threshold_score * runtime_score
        scores.append((tid, task_score, threshold_score, runtime_score))

    overall = sum(s[1] for s in scores) / len(scores)

    scores_sorted = sorted(scores, key=lambda x: x[1])

    print(f"Tasks: {len(scores)}")
    print(f"Missing predictions: {missing}")
    print(f"Overall score: {overall:.6f}")

    print(f"\nWorst {args.topk}:")
    for s in scores_sorted[: args.topk]:
        print(f"  {s[0]}: task={s[1]:.4f} thr={s[2]:.4f} time={s[3]:.4f}")

    print(f"\nBest {args.topk}:")
    for s in scores_sorted[-args.topk :]:
        print(f"  {s[0]}: task={s[1]:.4f} thr={s[2]:.4f} time={s[3]:.4f}")

    report = {
        "overall_score": overall,
        "tasks": [
            {
                "id": s[0],
                "task_score": s[1],
                "threshold_score": s[2],
                "runtime_score": s[3],
            }
            for s in scores
        ],
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote report: {args.out}")


if __name__ == "__main__":
    main()

