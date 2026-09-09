"""
Evaluation metrics and reporting.

Tag / genre: Precision, Recall, F1 per tag; Macro-F1; Micro-F1; mean AUC-PR.
Emotion: MAE + R2 for valence / arousal (DEAM).
Retrieval: Caption↔Audio R@K.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, r2_score

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
    """Mean area under precision-recall curve over tags."""
    scores: list[float] = []
    for k in range(y_true.shape[1]):
        if y_true[:, k].sum() == 0:
            continue
        scores.append(float(average_precision_score(y_true[:, k], y_prob[:, k])))
    return float(np.mean(scores)) if scores else 0.0

def emotion_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    """MAE and R2 for valence / arousal (DEAM)."""
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if yt.size == 0:
        return {"mae": float("nan"), "r2": float("nan")}
    mae = float(np.mean(np.abs(yt - yp)))
    if yt.size < 2 or float(np.std(yt)) < 1e-12:
        r2 = float("nan")
    else:
        r2 = float(r2_score(yt, yp))
    return {"mae": mae, "r2": r2}

def emotion_metrics_va(
    v_true: Any,
    v_pred: Any,
    a_true: Any,
    a_pred: Any,
) -> dict[str, float]:
    """Combined valence/arousal MAE + R2."""
    vm = emotion_metrics(v_true, v_pred)
    am = emotion_metrics(a_true, a_pred)
    return {
        "mae_valence": vm["mae"],
        "r2_valence": vm["r2"],
        "mae_arousal": am["mae"],
        "r2_arousal": am["r2"],
    }

def retrieval_recall_at_k(
    similarity: Any,
    ks: list[int] | None = None,
) -> dict[str, float]:
    """
    Diagonal-as-positive retrieval: row i should rank column i highly.
    `similarity`: (N, N) matrix (queries × gallery).
    """
    if ks is None:
        ks = [1, 5, 10]
    sim = np.asarray(similarity, dtype=np.float64)
    n = sim.shape[0]
    out: dict[str, float] = {}
    ranks = np.argsort(-sim, axis=1)
    for k in ks:
        kk = min(k, n)
        hits = sum(1 for i in range(n) if i in set(ranks[i, :kk].tolist()))
        out[f"R@{k}"] = float(hits) / float(max(n, 1))
    return out

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
    metrics_path = Path("results/metrics.json")
    if not metrics_path.exists():
        raise SystemExit("results/metrics.json missing - train first.")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    key = {
        1: "task1_bert_musiccaps",
        2: "task2_fma_small",
        3: "task3_gnn_bert_fusion",
        4: "task4_contrastive_musiccaps",
    }[args.task]
    block = metrics.get(key, {})
    print(json.dumps(block if not args.checkpoint else {"checkpoint": args.checkpoint, **block}, indent=2)[:4000])
    print(
        f"\n[evaluate] Task {args.task} metrics loaded from {metrics_path}. "
        "Full recompute: python -m src.postprocess"
    )

if __name__ == "__main__":
    main()
