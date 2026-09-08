"""
Evaluation metrics and reporting (PDF Section 6).

Tag / genre: Precision, Recall, F1 per tag; Macro-F1; Micro-F1; mean AUC-PR.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, f1_score


def macro_micro_f1(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Macro-F1 and Micro-F1 for multi-label tags."""
    y_pred = (y_prob >= threshold).astype(np.int32)
    return {
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "micro_f1": float(
            f1_score(y_true, y_pred, average="micro", zero_division=0)
        ),
    }


def mean_auc_pr(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Mean area under precision–recall curve over tags."""
    scores: list[float] = []
    for k in range(y_true.shape[1]):
        if y_true[:, k].sum() == 0:
            continue
        scores.append(float(average_precision_score(y_true[:, k], y_prob[:, k])))
    return float(np.mean(scores)) if scores else 0.0


def emotion_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    """MAE and R² for valence / arousal (DEAM) — filled in later tasks."""
    raise NotImplementedError("Implement with Task 3 / DEAM.")


def retrieval_recall_at_k(
    similarity: Any,
    ks: list[int] | None = None,
) -> dict[str, float]:
    """Caption→Audio and Audio→Caption R@K — Task 4."""
    if ks is None:
        ks = [1, 5, 10]
    raise NotImplementedError("Implement in Task 4 / evaluation stage.")


def save_metrics(metrics: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained models.")
    parser.add_argument("--task", type=int, choices=[1, 2, 3, 4], required=True)
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()
    if args.task == 1:
        print(
            "Task 1 metrics are written during training "
            "(results/metrics.json and results/plots/). "
            f"Optional checkpoint: {args.checkpoint}"
        )
        return
    raise NotImplementedError(f"Evaluation for task {args.task} comes in later stages.")


if __name__ == "__main__":
    main()
