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
    """Algorithm 2 — GNN on FMA-small segment graphs + CNN mel baseline (B2)."""
    from torch_geometric.loader import DataLoader as GeoDataLoader

    from src.gnn_model import CNNMelBaseline, MusicGAT, MusicGraphSAGE
    from src.graph_builder import build_segment_graph, save_graph, save_graph_json_summary

    set_seed(int(cfg.get("project", {}).get("seed", 42)))
    device = resolve_device(cfg)
    print(f"[task2] device={device}")

    paths = cfg["datasets"]["paths"]
    processed_dir = Path(paths["processed"]) / "fma_small"
    splits_path = Path(paths["splits"]) / "fma_small_splits.json"
    if not splits_path.exists():
        raise FileNotFoundError(f"Missing splits: {splits_path}")
    with splits_path.open(encoding="utf-8") as f:
        splits = json.load(f)

    graph_cfg = cfg.get("graph", {})
    sim_tau = float(graph_cfg.get("similarity_threshold", 0.7))
    temporal_edges = bool(graph_cfg.get("temporal_edges", True))
    similarity_edges = bool(graph_cfg.get("similarity_edges", True))

    def rows_to_graphs(rows: list[dict[str, Any]]) -> list[Any]:
        graphs = []
        for row in rows:
            tid = int(row["track_id"])
            npz_path = processed_dir / f"{tid:06d}.npz"
            if not npz_path.exists():
                continue
            arr = np.load(npz_path)
            seg = arr["segment_vectors"]
            g = build_segment_graph(
                seg,
                similarity_threshold=sim_tau,
                temporal_edges=temporal_edges,
                similarity_edges=similarity_edges,
                y=int(row["genre_id"]),
                track_id=tid,
            )
            graphs.append(g)
        return graphs

    train_graphs = rows_to_graphs(splits["train"])
    val_graphs = rows_to_graphs(splits["val"])
    test_graphs = rows_to_graphs(splits["test"])
    num_classes = len(splits["genre_to_id"])
    in_channels = int(train_graphs[0].x.size(1))
    print(
        f"[task2] graphs train={len(train_graphs)} val={len(val_graphs)} "
        f"test={len(test_graphs)} in_dim={in_channels} classes={num_classes}"
    )

    # Submission: ≥ 20 example graphs
    graph_samples_dir = Path(paths["processed"]) / "graph_samples"
    for i, g in enumerate(train_graphs[:20]):
        save_graph(g, graph_samples_dir / f"sample_{i:02d}.pt")
        save_graph_json_summary(g, graph_samples_dir / f"sample_{i:02d}.json")
    print(f"[task2] wrote 20 graph samples -> {graph_samples_dir}")

    train_cfg = cfg.get("train", {})
    batch_size = int(train_cfg.get("batch_size", 16))
    epochs = int(train_cfg.get("epochs", 20))
    # GNN/CNN from scratch: use 1e-3 (config 2e-5 is BERT-oriented)
    lr = float(train_cfg.get("gnn_learning_rate", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 0.01))

    model_cfg = cfg.get("model", {})
    gnn_type = str(model_cfg.get("gnn_type", "graphsage")).lower()
    hidden = int(model_cfg.get("gnn_hidden_dim", 128))
    n_layers = int(model_cfg.get("gnn_layers", 2))
    dropout = float(model_cfg.get("gnn_dropout", 0.2))

    if gnn_type == "gat":
        gnn = MusicGAT(
            in_channels, hidden, n_layers, num_classes, dropout=dropout
        ).to(device)
    else:
        gnn = MusicGraphSAGE(
            in_channels, hidden, n_layers, num_classes, dropout=dropout
        ).to(device)

    train_loader = GeoDataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = GeoDataLoader(val_graphs, batch_size=batch_size, shuffle=False)
    test_loader = GeoDataLoader(test_graphs, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(gnn.parameters(), lr=lr, weight_decay=weight_decay)

    def eval_gnn(loader: Any) -> dict[str, float]:
        gnn.eval()
        correct, total = 0, 0
        losses: list[float] = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                logits = gnn(batch.x, batch.edge_index, batch.batch)
                loss = criterion(logits, batch.y)
                losses.append(float(loss.item()))
                pred = logits.argmax(dim=-1)
                correct += int((pred == batch.y).sum().item())
                total += int(batch.y.numel())
        return {
            "loss": float(np.mean(losses)) if losses else 0.0,
            "accuracy": correct / max(total, 1),
            "macro_f1": _multiclass_f1(gnn, loader, device, num_classes),
        }

    history: list[dict[str, float]] = []
    best_acc = -1.0
    out_cfg = cfg.get("output", {})
    ckpt_dir = Path(out_cfg.get("checkpoint_dir", "results/checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_gnn_path = ckpt_dir / "task2_gnn_best.pt"

    for epoch in range(1, epochs + 1):
        gnn.train()
        losses: list[float] = []
        for batch in tqdm(train_loader, desc=f"task2-gnn {epoch}/{epochs}"):
            batch = batch.to(device)
            opt.zero_grad(set_to_none=True)
            logits = gnn(batch.x, batch.edge_index, batch.batch)
            loss = criterion(logits, batch.y)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        val_m = eval_gnn(val_loader)
        row = {
            "epoch": float(epoch),
            "train_loss": float(np.mean(losses)),
            "val_loss": val_m["loss"],
            "val_accuracy": val_m["accuracy"],
            "val_macro_f1": val_m["macro_f1"],
        }
        history.append(row)
        print(
            f"[task2-gnn] epoch={epoch} loss={row['train_loss']:.4f} "
            f"val_acc={row['val_accuracy']:.4f} val_macro_f1={row['val_macro_f1']:.4f}"
        )
        if val_m["accuracy"] > best_acc:
            best_acc = val_m["accuracy"]
            torch.save({"model_state": gnn.state_dict(), "gnn_type": gnn_type}, best_gnn_path)

    ckpt = torch.load(best_gnn_path, map_location=device, weights_only=False)
    gnn.load_state_dict(ckpt["model_state"])
    test_gnn = eval_gnn(test_loader)

    # --- Baseline B2: CNN on mel ---
    print("[task2] training CNN mel baseline (B2)...")
    cnn = CNNMelBaseline(num_classes=num_classes, n_mels=int(cfg["audio"]["n_mels"])).to(device)
    cnn_opt = torch.optim.AdamW(cnn.parameters(), lr=lr, weight_decay=weight_decay)

    def load_mel_items(rows: list[dict[str, Any]]) -> list[tuple[np.ndarray, int]]:
        items = []
        for row in rows:
            tid = int(row["track_id"])
            npz_path = processed_dir / f"{tid:06d}.npz"
            if not npz_path.exists():
                continue
            mel = np.load(npz_path)["mel"].astype(np.float32)
            items.append((mel, int(row["genre_id"])))
        return items

    train_mel = load_mel_items(splits["train"])
    val_mel = load_mel_items(splits["val"])
    test_mel = load_mel_items(splits["test"])

    def mel_batches(items: list[tuple[np.ndarray, int]], shuffle: bool):
        idxs = list(range(len(items)))
        if shuffle:
            random.shuffle(idxs)
        for start in range(0, len(idxs), batch_size):
            batch_idx = idxs[start : start + batch_size]
            mels = [items[i][0] for i in batch_idx]
            ys = [items[i][1] for i in batch_idx]
            max_t = max(m.shape[1] for m in mels)
            n_mels = mels[0].shape[0]
            stacked = np.zeros((len(mels), n_mels, max_t), dtype=np.float32)
            for i, m in enumerate(mels):
                stacked[i, :, : m.shape[1]] = m
            yield torch.from_numpy(stacked), torch.tensor(ys, dtype=torch.long)

    def eval_cnn(items: list[tuple[np.ndarray, int]]) -> dict[str, float]:
        cnn.eval()
        correct, total = 0, 0
        all_true: list[int] = []
        all_pred: list[int] = []
        with torch.no_grad():
            for xb, yb in mel_batches(items, shuffle=False):
                logits = cnn(xb.to(device))
                pred = logits.argmax(dim=-1).cpu()
                correct += int((pred == yb).sum().item())
                total += int(yb.numel())
                all_true.extend(yb.tolist())
                all_pred.extend(pred.tolist())
        return {
            "accuracy": correct / max(total, 1),
            "macro_f1": float(
                __import__("sklearn.metrics", fromlist=["f1_score"]).f1_score(
                    all_true, all_pred, average="macro", zero_division=0
                )
            ),
        }

    best_cnn_acc = -1.0
    best_cnn_path = ckpt_dir / "task2_cnn_best.pt"
    cnn_history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        cnn.train()
        losses = []
        for xb, yb in tqdm(
            list(mel_batches(train_mel, shuffle=True)),
            desc=f"task2-cnn {epoch}/{epochs}",
        ):
            cnn_opt.zero_grad(set_to_none=True)
            logits = cnn(xb.to(device))
            loss = criterion(logits, yb.to(device))
            loss.backward()
            cnn_opt.step()
            losses.append(float(loss.item()))
        val_c = eval_cnn(val_mel)
        cnn_history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(np.mean(losses)),
                "val_accuracy": val_c["accuracy"],
                "val_macro_f1": val_c["macro_f1"],
            }
        )
        print(
            f"[task2-cnn] epoch={epoch} loss={np.mean(losses):.4f} "
            f"val_acc={val_c['accuracy']:.4f} val_macro_f1={val_c['macro_f1']:.4f}"
        )
        if val_c["accuracy"] > best_cnn_acc:
            best_cnn_acc = val_c["accuracy"]
            torch.save({"model_state": cnn.state_dict()}, best_cnn_path)

    cnn.load_state_dict(
        torch.load(best_cnn_path, map_location=device, weights_only=False)["model_state"]
    )
    test_cnn = eval_cnn(test_mel)

    # Majority baseline B1
    train_labels = [int(r["genre_id"]) for r in splits["train"]]
    majority = int(Counter(train_labels).most_common(1)[0][0])
    test_labels = [int(r["genre_id"]) for r in splits["test"] if (processed_dir / f"{int(r['track_id']):06d}.npz").exists()]
    maj_acc = float(np.mean([1.0 if y == majority else 0.0 for y in test_labels]))
    maj_pred = [majority] * len(test_labels)
    maj_f1 = float(
        __import__("sklearn.metrics", fromlist=["f1_score"]).f1_score(
            test_labels, maj_pred, average="macro", zero_division=0
        )
    )

    plots_dir = Path(out_cfg.get("plots_dir", "results/plots"))
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plots_dir / "task2_gnn_vs_cnn.png"
    plt.figure(figsize=(7, 4))
    plt.plot([h["epoch"] for h in history], [h["val_macro_f1"] for h in history], label="GNN val Macro-F1")
    plt.plot(
        [h["epoch"] for h in cnn_history],
        [h["val_macro_f1"] for h in cnn_history],
        label="CNN val Macro-F1",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title("Task 2: GNN vs CNN (FMA-small)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    payload_update = {
        "task2_fma_small": {
            "gnn": {"test": test_gnn, "best_val_accuracy": best_acc, "history": history, "checkpoint": str(best_gnn_path)},
            "cnn_mel_baseline": {
                "test": test_cnn,
                "best_val_accuracy": best_cnn_acc,
                "history": cnn_history,
                "checkpoint": str(best_cnn_path),
            },
            "majority_baseline": {"test_accuracy": maj_acc, "test_macro_f1": maj_f1},
            "graph_samples_dir": str(graph_samples_dir),
            "comparison_plot": str(plot_path),
        }
    }
    metrics_file = Path(out_cfg.get("metrics_file", "results/metrics.json"))
    payload: dict[str, Any] = payload_update
    if metrics_file.exists():
        try:
            prev = json.loads(metrics_file.read_text(encoding="utf-8"))
            if isinstance(prev, dict):
                prev.update(payload_update)
                payload = prev
        except json.JSONDecodeError:
            pass
    save_metrics(payload, metrics_file)
    print(f"[task2] GNN test: {test_gnn}")
    print(f"[task2] CNN test: {test_cnn}")
    print(f"[task2] Majority test acc={maj_acc:.4f} macro_f1={maj_f1:.4f}")
    print(f"[task2] wrote {metrics_file}, {plot_path}")


def _multiclass_f1(
    model: nn.Module,
    loader: Any,
    device: torch.device,
    num_classes: int,
) -> float:
    from sklearn.metrics import f1_score

    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.batch)
            ps.extend(logits.argmax(dim=-1).cpu().tolist())
            ys.extend(batch.y.cpu().tolist())
    return float(f1_score(ys, ps, average="macro", labels=list(range(num_classes)), zero_division=0))


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
