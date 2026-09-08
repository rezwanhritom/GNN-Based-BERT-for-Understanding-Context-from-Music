"""
Training entrypoint for Tasks 1–4 (Algorithms 1–4).

Usage:
  python -m src.train --task 1 --config config.yaml
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.bert_encoder import BertMusicTagClassifier, build_tokenizer, tokenize_batch
from src.evaluate import macro_micro_f1, mean_auc_pr, save_metrics


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(cfg: dict[str, Any]) -> torch.device:
    want = str(cfg.get("project", {}).get("device", "cuda"))
    if want == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def parse_aspect_list(value: Any) -> list[str]:
    """MusicCaps aspect_list is a stringified Python list."""
    if isinstance(value, list):
        return [str(x).strip().lower() for x in value if str(x).strip()]
    if pd.isna(value):
        return []
    text = str(value).strip()
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(x).strip().lower() for x in parsed if str(x).strip()]
    except (SyntaxError, ValueError):
        pass
    return [t.strip().lower() for t in text.split(",") if t.strip()]


def build_musiccaps_tag_proxy(
    csv_path: Path,
    top_k: int,
    seed: int,
    val_ratio: float = 0.1,
) -> dict[str, Any]:
    """
    MusicCaps caption → tag proxy (PDF Task 1):
      X_text = caption, y = multi-hot over top-K aspect tags.
    Split: is_audioset_eval → test; remaining → train/val.
    """
    df = pd.read_csv(csv_path)
    df["aspects"] = df["aspect_list"].map(parse_aspect_list)
    df["caption"] = df["caption"].fillna("").astype(str)

    train_pool = df.loc[~df["is_audioset_eval"]].copy()
    test_df = df.loc[df["is_audioset_eval"]].copy()

    aspect_counts: Counter[str] = Counter()
    for aspects in train_pool["aspects"]:
        aspect_counts.update(set(aspects))
    vocab = [a for a, _ in aspect_counts.most_common(top_k)]
    tag_to_id = {t: i for i, t in enumerate(vocab)}

    rng = np.random.default_rng(seed)
    idx = np.arange(len(train_pool))
    rng.shuffle(idx)
    n_val = max(1, int(len(idx) * val_ratio))
    val_idx = set(idx[:n_val].tolist())
    train_df = train_pool.iloc[[i for i in range(len(train_pool)) if i not in val_idx]]
    val_df = train_pool.iloc[list(val_idx)]

    def encode_split(frame: pd.DataFrame) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for _, row in frame.iterrows():
            y = np.zeros(len(vocab), dtype=np.float32)
            for a in row["aspects"]:
                if a in tag_to_id:
                    y[tag_to_id[a]] = 1.0
            if y.sum() == 0:
                continue
            rows.append(
                {
                    "ytid": str(row["ytid"]),
                    "caption": row["caption"],
                    "labels": y,
                }
            )
        return rows

    return {
        "tag_to_id": tag_to_id,
        "id_to_tag": {i: t for t, i in tag_to_id.items()},
        "train": encode_split(train_df),
        "val": encode_split(val_df),
        "test": encode_split(test_df),
    }


class MusicCapsTagDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        return {
            "caption": row["caption"],
            "labels": torch.from_numpy(row["labels"]),
            "ytid": row["ytid"],
        }


def collate_musiccaps(
    batch: list[dict[str, Any]],
    tokenizer: Any,
    max_length: int,
) -> dict[str, Any]:
    texts = [b["caption"] for b in batch]
    tok = tokenize_batch(texts, tokenizer, max_length=max_length)
    labels = torch.stack([b["labels"] for b in batch], dim=0)
    return {
        "input_ids": tok["input_ids"],
        "attention_mask": tok["attention_mask"],
        "labels": labels,
        "captions": texts,
        "ytids": [b["ytid"] for b in batch],
    }


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    all_true: list[np.ndarray] = []
    all_prob: list[np.ndarray] = []
    for batch in loader:
        logits = model(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
        )
        prob = torch.sigmoid(logits).cpu().numpy()
        all_prob.append(prob)
        all_true.append(batch["labels"].numpy())
    y_true = np.concatenate(all_true, axis=0)
    y_prob = np.concatenate(all_prob, axis=0)
    metrics = macro_micro_f1(y_true, y_prob)
    metrics["auc_pr"] = mean_auc_pr(y_true, y_prob)
    return metrics


def plot_f1_curves(
    history: list[dict[str, float]],
    out_path: Path,
) -> None:
    epochs = [h["epoch"] for h in history]
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, [h["val_macro_f1"] for h in history], label="val Macro-F1")
    plt.plot(epochs, [h["val_micro_f1"] for h in history], label="val Micro-F1")
    plt.plot(epochs, [h["train_loss"] for h in history], label="train loss", alpha=0.7)
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Task 1: BERT tag F1 vs epochs")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


@torch.no_grad()
def example_predictions(
    model: nn.Module,
    rows: list[dict[str, Any]],
    tokenizer: Any,
    id_to_tag: dict[int, str],
    device: torch.device,
    max_length: int,
    n: int = 5,
    threshold: float = 0.5,
) -> list[dict[str, Any]]:
    model.eval()
    examples: list[dict[str, Any]] = []
    for row in rows[:n]:
        tok = tokenize_batch([row["caption"]], tokenizer, max_length=max_length)
        logits = model(tok["input_ids"].to(device), tok["attention_mask"].to(device))
        prob = torch.sigmoid(logits)[0].cpu().numpy()
        true_ids = np.where(row["labels"] >= 0.5)[0].tolist()
        pred_ids = np.where(prob >= threshold)[0].tolist()
        examples.append(
            {
                "ytid": row["ytid"],
                "caption": row["caption"][:300],
                "true_tags": [id_to_tag[i] for i in true_ids],
                "pred_tags": [id_to_tag[i] for i in pred_ids],
                "pred_scores": {id_to_tag[i]: float(prob[i]) for i in pred_ids},
            }
        )
    return examples


def train_task1(cfg: dict[str, Any]) -> None:
    """Algorithm 1 — BERT multi-label tag classifier on MusicCaps caption→tag proxy."""
    set_seed(int(cfg.get("project", {}).get("seed", 42)))
    device = resolve_device(cfg)
    print(f"[task1] device={device}")

    csv_path = Path(cfg["datasets"]["paths"]["raw"]) / "musiccaps" / "musiccaps-public.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing MusicCaps CSV: {csv_path}")

    top_k = int(cfg.get("eval", {}).get("top_k_tags", 50))
    data = build_musiccaps_tag_proxy(
        csv_path,
        top_k=top_k,
        seed=int(cfg.get("project", {}).get("seed", 42)),
    )
    splits_out = Path(cfg["datasets"]["paths"]["splits"]) / "musiccaps_tag_proxy_splits.json"
    splits_out.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        "tag_to_id": data["tag_to_id"],
        "counts": {
            "train": len(data["train"]),
            "val": len(data["val"]),
            "test": len(data["test"]),
        },
        "train_ytids": [r["ytid"] for r in data["train"]],
        "val_ytids": [r["ytid"] for r in data["val"]],
        "test_ytids": [r["ytid"] for r in data["test"]],
    }
    with splits_out.open("w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
    print(
        f"[task1] MusicCaps top-{top_k} tags | "
        f"train={len(data['train'])} val={len(data['val'])} test={len(data['test'])}"
    )

    model_name = cfg.get("model", {}).get("bert_name", "bert-base-uncased")
    max_length = int(cfg.get("text", {}).get("max_length", 128))
    tokenizer = build_tokenizer(model_name)
    model = BertMusicTagClassifier(
        model_name=model_name,
        num_labels=top_k,
        freeze_bert=bool(cfg.get("model", {}).get("freeze_bert", False)),
    ).to(device)

    train_cfg = cfg.get("train", {})
    batch_size = int(train_cfg.get("batch_size", 16))
    epochs = int(train_cfg.get("epochs", 20))
    lr = float(train_cfg.get("learning_rate", 2e-5))
    weight_decay = float(train_cfg.get("weight_decay", 0.01))
    num_workers = int(train_cfg.get("num_workers", 0))
    if sys.platform == "win32":
        num_workers = 0

    def make_loader(rows: list[dict[str, Any]], shuffle: bool) -> DataLoader:
        ds = MusicCapsTagDataset(rows)
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=lambda b: collate_musiccaps(b, tokenizer, max_length),
        )

    train_loader = make_loader(data["train"], shuffle=True)
    val_loader = make_loader(data["val"], shuffle=False)
    test_loader = make_loader(data["test"], shuffle=False)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss()

    out_cfg = cfg.get("output", {})
    ckpt_dir = Path(out_cfg.get("checkpoint_dir", "results/checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = Path(out_cfg.get("plots_dir", "results/plots"))
    metrics_file = Path(out_cfg.get("metrics_file", "results/metrics.json"))

    history: list[dict[str, float]] = []
    best_macro = -1.0
    best_path = ckpt_dir / "task1_bert_best.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        for batch in tqdm(train_loader, desc=f"task1 epoch {epoch}/{epochs}"):
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
            )
            loss = criterion(logits, batch["labels"].to(device))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        val_metrics = evaluate_loader(model, val_loader, device)
        row = {
            "epoch": float(epoch),
            "train_loss": float(np.mean(losses)),
            "val_macro_f1": val_metrics["macro_f1"],
            "val_micro_f1": val_metrics["micro_f1"],
            "val_auc_pr": val_metrics["auc_pr"],
        }
        history.append(row)
        print(
            f"[task1] epoch={epoch} loss={row['train_loss']:.4f} "
            f"val_macro_f1={row['val_macro_f1']:.4f} "
            f"val_micro_f1={row['val_micro_f1']:.4f} "
            f"val_auc_pr={row['val_auc_pr']:.4f}"
        )
        if val_metrics["macro_f1"] > best_macro:
            best_macro = val_metrics["macro_f1"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "tag_to_id": data["tag_to_id"],
                    "config_model_name": model_name,
                    "num_labels": top_k,
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                },
                best_path,
            )

    # Load best and evaluate test
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    test_metrics = evaluate_loader(model, test_loader, device)
    examples = example_predictions(
        model,
        data["test"],
        tokenizer,
        data["id_to_tag"],
        device,
        max_length=max_length,
        n=5,
    )

    plot_path = plots_dir / "task1_f1_curves.png"
    plot_f1_curves(history, plot_path)

    examples_path = Path(out_cfg.get("results_dir", "results")) / "task1_example_predictions.json"
    with examples_path.open("w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2)

    payload = {
        "task1_bert_musiccaps": {
            "top_k_tags": top_k,
            "best_val_macro_f1": best_macro,
            "test": test_metrics,
            "history": history,
            "checkpoint": str(best_path),
            "f1_curve_plot": str(plot_path),
            "examples": str(examples_path),
        }
    }
    # merge with existing metrics if present
    if metrics_file.exists():
        try:
            prev = json.loads(metrics_file.read_text(encoding="utf-8"))
            if isinstance(prev, dict):
                prev.update(payload)
                payload = prev
        except json.JSONDecodeError:
            pass
    save_metrics(payload, metrics_file)
    print(f"[task1] test metrics: {test_metrics}")
    print(f"[task1] wrote {metrics_file}, {plot_path}, {examples_path}")


def train_task2(cfg: dict[str, Any]) -> None:
    raise NotImplementedError("Implement in Task 2 stage.")


def train_task3(cfg: dict[str, Any]) -> None:
    raise NotImplementedError("Implement in Task 3 stage.")


def train_task4(cfg: dict[str, Any]) -> None:
    raise NotImplementedError("Implement in Task 4 stage.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train GNN–BERT music context models (CSE425)."
    )
    parser.add_argument("--task", type=int, required=True, choices=[1, 2, 3, 4])
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    {
        1: train_task1,
        2: train_task2,
        3: train_task3,
        4: train_task4,
    }[args.task](cfg)


if __name__ == "__main__":
    main()
