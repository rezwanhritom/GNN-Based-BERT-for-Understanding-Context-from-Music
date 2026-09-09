"""
Training entrypoint for Tasks 1-4 (Algorithms 1-4).

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
    MusicCaps caption -> tag proxy :
      X_text = caption, y = multi-hot over top-K aspect tags.
    Split: is_audioset_eval -> test; remaining -> train/val.
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
    """Task 1 - BERT multi-label tag classifier on MusicCaps caption->tag proxy."""
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
    """Task 2 - GNN on FMA-small segment graphs + CNN mel baseline (B2)."""
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
    """
    Task 3 - GNN-BERT fusion on FMA-small (graph + track text -> genre).
    Ablations: BERT-only, GNN-only, early_concat, cross_attention.
    DEAM emotion aux skipped (annotations not present under data/raw/deam).
    """
    from torch_geometric.data import Batch
    from torch_geometric.loader import DataLoader as GeoDataLoader

    from src.fusion_model import GNNBertFusionModel
    from src.graph_builder import build_segment_graph

    set_seed(int(cfg.get("project", {}).get("seed", 42)))
    device = resolve_device(cfg)
    print(f"[task3] device={device}")

    paths = cfg["datasets"]["paths"]
    processed_dir = Path(paths["processed"]) / "fma_small"
    splits_path = Path(paths["splits"]) / "fma_small_splits.json"
    with splits_path.open(encoding="utf-8") as f:
        splits = json.load(f)

    tracks = pd.read_csv(
        Path(paths["raw"]) / "fma" / "fma_metadata" / "tracks.csv",
        index_col=0,
        header=[0, 1],
    )

    def track_text(tid: int) -> str:
        row = tracks.loc[int(tid)]
        title = str(row[("track", "title")]) if ("track", "title") in tracks.columns else ""
        artist = str(row[("artist", "name")]) if ("artist", "name") in tracks.columns else ""
        album = str(row[("album", "title")]) if ("album", "title") in tracks.columns else ""
        atags = str(row[("artist", "tags")]) if ("artist", "tags") in tracks.columns else ""
        parts = [
            title if title and title != "nan" else "",
            f"Artist: {artist}" if artist and artist != "nan" else "",
            f"Album: {album}" if album and album != "nan" else "",
            f"Tags: {atags}" if atags and atags not in {"nan", "[]", ""} else "",
        ]
        text = ". ".join(p for p in parts if p).strip()
        return text if text else "unknown track"

    graph_cfg = cfg.get("graph", {})
    sim_tau = float(graph_cfg.get("similarity_threshold", 0.7))

    def build_items(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        items = []
        for row in rows:
            tid = int(row["track_id"])
            npz_path = processed_dir / f"{tid:06d}.npz"
            if not npz_path.exists():
                continue
            arr = np.load(npz_path)
            g = build_segment_graph(
                arr["segment_vectors"],
                similarity_threshold=sim_tau,
                temporal_edges=bool(graph_cfg.get("temporal_edges", True)),
                similarity_edges=bool(graph_cfg.get("similarity_edges", True)),
                y=int(row["genre_id"]),
                track_id=tid,
            )
            items.append(
                {
                    "graph": g,
                    "text": track_text(tid),
                    "genre_id": int(row["genre_id"]),
                    "genre": str(row["genre"]),
                    "track_id": tid,
                }
            )
        return items

    train_items = build_items(splits["train"])
    val_items = build_items(splits["val"])
    test_items = build_items(splits["test"])
    num_classes = len(splits["genre_to_id"])
    id_to_genre = {int(v): k for k, v in splits["genre_to_id"].items()}
    in_channels = int(train_items[0]["graph"].x.size(1))
    print(
        f"[task3] pairs train={len(train_items)} val={len(val_items)} "
        f"test={len(test_items)} classes={num_classes}"
    )

    model_cfg = cfg.get("model", {})
    text_cfg = cfg.get("text", {})
    train_cfg = cfg.get("train", {})
    model_name = str(model_cfg.get("bert_name", "bert-base-uncased"))
    max_length = int(text_cfg.get("max_length", 128))
    batch_size = int(train_cfg.get("batch_size", 16))
    epochs = int(train_cfg.get("epochs", 20))
    lr = float(train_cfg.get("learning_rate", 2e-5))
    weight_decay = float(train_cfg.get("weight_decay", 0.01))
    tokenizer = build_tokenizer(model_name)

    def make_loader(items: list[dict[str, Any]], shuffle: bool) -> DataLoader:
        class FusionDataset(Dataset):
            def __len__(self) -> int:
                return len(items)

            def __getitem__(self, idx: int) -> dict[str, Any]:
                return items[idx]

        def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
            graphs = Batch.from_data_list([b["graph"] for b in batch])
            tok = tokenize_batch([b["text"] for b in batch], tokenizer, max_length)
            y = torch.tensor([b["genre_id"] for b in batch], dtype=torch.long)
            return {
                "graph": graphs,
                "input_ids": tok["input_ids"],
                "attention_mask": tok["attention_mask"],
                "y": y,
                "texts": [b["text"] for b in batch],
                "track_ids": [b["track_id"] for b in batch],
                "genres": [b["genre"] for b in batch],
            }

        return DataLoader(
            FusionDataset(),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            collate_fn=collate,
        )

    train_loader = make_loader(train_items, True)
    val_loader = make_loader(val_items, False)
    test_loader = make_loader(test_items, False)

    criterion = nn.CrossEntropyLoss()
    out_cfg = cfg.get("output", {})
    ckpt_dir = Path(out_cfg.get("checkpoint_dir", "results/checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = Path(out_cfg.get("plots_dir", "results/plots"))
    plots_dir.mkdir(parents=True, exist_ok=True)

    ablation_modes = ["bert_only", "gnn_only", "early_concat", "cross_attention"]
    ablation_results: dict[str, Any] = {}

    def eval_fusion(model: nn.Module, loader: DataLoader) -> dict[str, float]:
        from sklearn.metrics import f1_score

        model.eval()
        ys, ps = [], []
        losses: list[float] = []
        with torch.no_grad():
            for batch in loader:
                g = batch["graph"].to(device)
                logits = model(
                    g.x,
                    g.edge_index,
                    g.batch,
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                )
                loss = criterion(logits, batch["y"].to(device))
                losses.append(float(loss.item()))
                pred = logits.argmax(dim=-1).cpu().tolist()
                ps.extend(pred)
                ys.extend(batch["y"].tolist())
        return {
            "loss": float(np.mean(losses)) if losses else 0.0,
            "accuracy": float(np.mean([a == b for a, b in zip(ys, ps)])),
            "macro_f1": float(
                f1_score(ys, ps, average="macro", labels=list(range(num_classes)), zero_division=0)
            ),
        }

    for mode in ablation_modes:
        best_path = ckpt_dir / f"task3_{mode}_best.pt"
        model = GNNBertFusionModel(
            in_channels=in_channels,
            num_classes=num_classes,
            bert_name=model_name,
            fusion=mode,
            gnn_type=str(model_cfg.get("gnn_type", "graphsage")),
            gnn_hidden=int(model_cfg.get("gnn_hidden_dim", 128)),
            gnn_layers=int(model_cfg.get("gnn_layers", 2)),
            gnn_dropout=float(model_cfg.get("gnn_dropout", 0.2)),
            freeze_bert=bool(model_cfg.get("freeze_bert", False)),
        ).to(device)

        history: list[dict[str, float]] = []
        best_f1 = -1.0

        if best_path.exists():
            print(f"[task3] resume/skip training for {mode} (found {best_path})")
            ckpt = torch.load(best_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state"])
            val_m = eval_fusion(model, val_loader)
            test_m = eval_fusion(model, test_loader)
            best_f1 = float(val_m["macro_f1"])
            ablation_results[mode] = {
                "test": test_m,
                "best_val_macro_f1": best_f1,
                "history": history,
                "checkpoint": str(best_path),
                "resumed": True,
            }
            print(f"[task3-{mode}] resumed test: {test_m}")
            continue

        print(f"[task3] training fusion={mode}")
        # GNN-only: higher LR; BERT modes: config LR
        mode_lr = 1e-3 if mode == "gnn_only" else lr
        opt = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=mode_lr,
            weight_decay=weight_decay,
        )

        for epoch in range(1, epochs + 1):
            model.train()
            losses = []
            for batch in tqdm(train_loader, desc=f"task3-{mode} {epoch}/{epochs}"):
                g = batch["graph"].to(device)
                opt.zero_grad(set_to_none=True)
                logits = model(
                    g.x,
                    g.edge_index,
                    g.batch,
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                )
                loss = criterion(logits, batch["y"].to(device))
                loss.backward()
                opt.step()
                losses.append(float(loss.item()))
            val_m = eval_fusion(model, val_loader)
            history.append(
                {
                    "epoch": float(epoch),
                    "train_loss": float(np.mean(losses)),
                    "val_accuracy": val_m["accuracy"],
                    "val_macro_f1": val_m["macro_f1"],
                }
            )
            print(
                f"[task3-{mode}] epoch={epoch} loss={np.mean(losses):.4f} "
                f"val_acc={val_m['accuracy']:.4f} val_macro_f1={val_m['macro_f1']:.4f}"
            )
            if val_m["macro_f1"] > best_f1:
                best_f1 = val_m["macro_f1"]
                torch.save({"model_state": model.state_dict(), "fusion": mode}, best_path)

        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        test_m = eval_fusion(model, test_loader)
        ablation_results[mode] = {
            "test": test_m,
            "best_val_macro_f1": best_f1,
            "history": history,
            "checkpoint": str(best_path),
            "resumed": False,
        }
        print(f"[task3-{mode}] test: {test_m}")

    # t-SNE of z from best cross-attention model
    from sklearn.manifold import TSNE

    best_mode = "cross_attention"
    model = GNNBertFusionModel(
        in_channels=in_channels,
        num_classes=num_classes,
        bert_name=model_name,
        fusion=best_mode,
        gnn_type=str(model_cfg.get("gnn_type", "graphsage")),
        gnn_hidden=int(model_cfg.get("gnn_hidden_dim", 128)),
        gnn_layers=int(model_cfg.get("gnn_layers", 2)),
        gnn_dropout=float(model_cfg.get("gnn_dropout", 0.2)),
    ).to(device)
    model.load_state_dict(
        torch.load(
            ablation_results[best_mode]["checkpoint"],
            map_location=device,
            weights_only=False,
        )["model_state"]
    )
    model.eval()
    zs, labels, case_pool = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            g = batch["graph"].to(device)
            logits, z = model(
                g.x,
                g.edge_index,
                g.batch,
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                return_z=True,
            )
            pred = logits.argmax(dim=-1).cpu().tolist()
            zs.append(z.cpu().numpy())
            labels.extend(batch["y"].tolist())
            for i in range(len(batch["track_ids"])):
                case_pool.append(
                    {
                        "track_id": int(batch["track_ids"][i]),
                        "text": batch["texts"][i][:240],
                        "true_genre": batch["genres"][i],
                        "pred_genre": id_to_genre.get(int(pred[i]), str(pred[i])),
                        "correct": bool(int(pred[i]) == int(batch["y"][i])),
                    }
                )

    Z = np.concatenate(zs, axis=0)
    # Cap t-SNE size for speed
    n_tsne = min(len(labels), 1000)
    idx = np.random.default_rng(42).choice(len(labels), size=n_tsne, replace=False)
    emb = TSNE(n_components=2, perplexity=30, random_state=42, init="pca").fit_transform(Z[idx])
    y_plot = np.array(labels)[idx]
    plt.figure(figsize=(8, 6))
    for gid, gname in sorted(id_to_genre.items()):
        m = y_plot == gid
        if m.any():
            plt.scatter(emb[m, 0], emb[m, 1], s=12, alpha=0.7, label=gname)
    plt.legend(markerscale=1.5, fontsize=8)
    plt.title("Task 3: t-SNE of fusion z (coloured by genre)")
    plt.tight_layout()
    tsne_path = plots_dir / "task3_tsne_genre.png"
    plt.savefig(tsne_path, dpi=150)
    plt.close()

    # 3 case studies (prefer one correct + mix)
    correct = [c for c in case_pool if c["correct"]]
    wrong = [c for c in case_pool if not c["correct"]]
    cases = (correct[:2] + wrong[:1]) if correct and wrong else case_pool[:3]
    # Enrich with edge counts from saved graphs
    for c in cases:
        npz = np.load(processed_dir / f"{c['track_id']:06d}.npz")
        g = build_segment_graph(npz["segment_vectors"], similarity_threshold=sim_tau)
        c["num_nodes"] = int(g.num_nodes)
        c["num_edges"] = int(g.edge_index.size(1))
        c["graph_path_note"] = (
            f"Segment graph with {c['num_nodes']} nodes; "
            f"temporal + similarity edges (tau={sim_tau})."
        )
    cases_path = Path(out_cfg.get("results_dir", "results")) / "task3_case_studies.json"
    with cases_path.open("w", encoding="utf-8") as f:
        json.dump(cases[:3], f, indent=2)

    # Ablation bar plot
    fig_path = plots_dir / "task3_ablation_macro_f1.png"
    names = list(ablation_results.keys())
    vals = [ablation_results[n]["test"]["macro_f1"] for n in names]
    plt.figure(figsize=(7, 4))
    plt.bar(names, vals)
    plt.ylabel("Test Macro-F1")
    plt.title("Task 3 ablations")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()

    payload_update = {
        "task3_gnn_bert_fusion": {
            "dataset": "fma_small",
            "ablations": ablation_results,
            "tsne_plot": str(tsne_path),
            "ablation_plot": str(fig_path),
            "case_studies": str(cases_path),
            "note": "DEAM L_aux not applied (annotations not in data/raw/deam).",
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
    print(f"[task3] wrote {metrics_file}, {tsne_path}, {cases_path}")

def train_task4(cfg: dict[str, Any]) -> None:
    """
    Task 4 - Contrastive dual-encoder on MusicCaps (graph ↔ caption).
    Downloads missing audio clips via yt-dlp when needed.
    """
    from torch_geometric.data import Batch

    from src.audio_features import extract_track_features
    from src.contrastive import DualEncoderContrastive, info_nce_loss, retrieval_metrics
    from src.graph_builder import build_segment_graph

    set_seed(int(cfg.get("project", {}).get("seed", 42)))
    device = resolve_device(cfg)
    print(f"[task4] device={device}")

    paths = cfg["datasets"]["paths"]
    raw_mc = Path(paths["raw"]) / "musiccaps"
    csv_path = raw_mc / "musiccaps-public.csv"
    audio_dir = raw_mc / "audio"
    processed_dir = Path(paths["processed"]) / "musiccaps"
    audio_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing MusicCaps CSV: {csv_path}")

    df = pd.read_csv(csv_path)

    def clip_stem(row: pd.Series) -> str:
        return f"{row['ytid']}_{int(row['start_s'])}_{int(row['end_s'])}"

    def find_audio(row: pd.Series) -> Path | None:
        """Accept segment-named files or HF mirror `{ytid}.wav`."""
        stem = clip_stem(row)
        ytid = str(row["ytid"])
        for name in (stem, ytid):
            for ext in (".wav", ".mp3", ".m4a", ".webm", ".opus", ".ogg"):
                p = audio_dir / f"{name}{ext}"
                if p.exists() and p.stat().st_size > 1000:
                    return p
            matches = list(audio_dir.glob(f"{name}.*"))
            if matches:
                return matches[0]
        return None

    # Use existing on-disk MusicCaps audio only (no re-download)
    min_clips = int(cfg.get("train", {}).get("task4_min_clips", 200))
    have = sum(1 for _, row in df.iterrows() if find_audio(row) is not None)
    print(f"[task4] existing MusicCaps audio clips: {have}")
    if have < 50:
        raise RuntimeError(
            f"Too few MusicCaps audio clips ({have}). Place clips under {audio_dir}."
        )

    graph_cfg = cfg.get("graph", {})
    sim_tau = float(graph_cfg.get("similarity_threshold", 0.7))

    pairs: list[dict[str, Any]] = []
    for _, row in tqdm(list(df.iterrows()), desc="build musiccaps pairs"):
        stem = clip_stem(row)
        audio_path = find_audio(row)
        if audio_path is None:
            continue
        npz_path = processed_dir / f"{stem}.npz"
        try:
            if not npz_path.exists():
                feats = extract_track_features(audio_path, cfg)
                np.savez_compressed(
                    npz_path,
                    segment_vectors=feats["segment_vectors"],
                    mel=feats["mel"],
                    chroma=feats["chroma"],
                )
            arr = np.load(npz_path)
            g = build_segment_graph(
                arr["segment_vectors"],
                similarity_threshold=sim_tau,
                temporal_edges=bool(graph_cfg.get("temporal_edges", True)),
                similarity_edges=bool(graph_cfg.get("similarity_edges", True)),
            )
            pairs.append(
                {
                    "stem": stem,
                    "ytid": str(row["ytid"]),
                    "caption": str(row["caption"]),
                    "aspect_list": str(row.get("aspect_list", "")),
                    "is_eval": bool(row["is_audioset_eval"]),
                    "graph": g,
                }
            )
        except Exception as exc:  # noqa: BLE001
            tqdm.write(f"[warn] {stem}: {exc}")

    train_pairs = [p for p in pairs if not p["is_eval"]]
    test_pairs = [p for p in pairs if p["is_eval"]]
    rng = np.random.default_rng(int(cfg.get("project", {}).get("seed", 42)))
    idx = np.arange(len(train_pairs))
    rng.shuffle(idx)
    n_val = max(1, int(0.1 * len(idx)))
    val_pairs = [train_pairs[i] for i in idx[:n_val]]
    train_pairs = [train_pairs[i] for i in idx[n_val:]]
    print(
        f"[task4] pairs train={len(train_pairs)} val={len(val_pairs)} "
        f"test={len(test_pairs)}"
    )
    if len(train_pairs) < 16 or len(test_pairs) < 8:
        raise RuntimeError("Not enough MusicCaps pairs after preprocessing.")

    model_cfg = cfg.get("model", {})
    text_cfg = cfg.get("text", {})
    train_cfg = cfg.get("train", {})
    model_name = str(model_cfg.get("bert_name", "bert-base-uncased"))
    max_length = int(text_cfg.get("max_length", 128))
    batch_size = int(train_cfg.get("batch_size", 16))
    epochs = int(train_cfg.get("epochs", 20))
    lr = float(train_cfg.get("learning_rate", 2e-5))
    weight_decay = float(train_cfg.get("weight_decay", 0.01))
    temperature = float(train_cfg.get("temperature", 0.07))
    proj_dim = int(model_cfg.get("projection_dim", 256))
    tokenizer = build_tokenizer(model_name)
    in_channels = int(train_pairs[0]["graph"].x.size(1))

    def make_loader(items: list[dict[str, Any]], shuffle: bool) -> DataLoader:
        class PairDS(Dataset):
            def __len__(self) -> int:
                return len(items)

            def __getitem__(self, i: int) -> dict[str, Any]:
                return items[i]

        def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
            graphs = Batch.from_data_list([b["graph"] for b in batch])
            tok = tokenize_batch([b["caption"] for b in batch], tokenizer, max_length)
            return {
                "graph": graphs,
                "input_ids": tok["input_ids"],
                "attention_mask": tok["attention_mask"],
                "captions": [b["caption"] for b in batch],
                "stems": [b["stem"] for b in batch],
            }

        return DataLoader(
            PairDS(),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            collate_fn=collate,
            drop_last=shuffle,
        )

    train_loader = make_loader(train_pairs, True)
    # For retrieval we embed full splits
    model = DualEncoderContrastive(
        in_channels=in_channels,
        bert_name=model_name,
        gnn_type=str(model_cfg.get("gnn_type", "graphsage")),
        gnn_hidden=int(model_cfg.get("gnn_hidden_dim", 128)),
        gnn_layers=int(model_cfg.get("gnn_layers", 2)),
        gnn_dropout=float(model_cfg.get("gnn_dropout", 0.2)),
        projection_dim=proj_dim,
        temperature=temperature,
        freeze_bert=bool(model_cfg.get("freeze_bert", False)),
    ).to(device)

    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    out_cfg = cfg.get("output", {})
    ckpt_dir = Path(out_cfg.get("checkpoint_dir", "results/checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "task4_contrastive_best.pt"
    best_r5 = -1.0
    history: list[dict[str, float]] = []

    @torch.no_grad()
    def embed_split(items: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
        model.eval()
        gs, ts, stems = [], [], []
        loader = make_loader(items, shuffle=False)
        for batch in loader:
            g = batch["graph"].to(device)
            ge, te = model(
                g.x,
                g.edge_index,
                g.batch,
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
            )
            gs.append(ge.cpu())
            ts.append(te.cpu())
            stems.extend(batch["stems"])
        return torch.cat(gs, dim=0), torch.cat(ts, dim=0), stems

    def eval_retrieval(items: list[dict[str, Any]]) -> dict[str, float]:
        g_emb, t_emb, _ = embed_split(items)
        sim_g2t = g_emb @ t_emb.t()
        sim_t2g = t_emb @ g_emb.t()
        m_g2t = retrieval_metrics(sim_g2t)
        m_t2g = retrieval_metrics(sim_t2g)
        return {
            **{f"audio2caption_{k}": v for k, v in m_g2t.items()},
            **{f"caption2audio_{k}": v for k, v in m_t2g.items()},
        }

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"task4 {epoch}/{epochs}"):
            g = batch["graph"].to(device)
            opt.zero_grad(set_to_none=True)
            ge, te = model(
                g.x,
                g.edge_index,
                g.batch,
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
            )
            loss = info_nce_loss(ge, te, temperature=temperature)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        val_m = eval_retrieval(val_pairs if len(val_pairs) >= 8 else test_pairs[:64])
        row = {
            "epoch": float(epoch),
            "train_loss": float(np.mean(losses)),
            **val_m,
        }
        history.append(row)
        r5 = val_m.get("caption2audio_R@5", 0.0)
        print(
            f"[task4] epoch={epoch} loss={row['train_loss']:.4f} "
            f"c2a_R@1={val_m.get('caption2audio_R@1', 0):.3f} "
            f"c2a_R@5={r5:.3f} a2c_R@5={val_m.get('audio2caption_R@5', 0):.3f}"
        )
        if r5 > best_r5:
            best_r5 = r5
            torch.save({"model_state": model.state_dict()}, best_path)

    model.load_state_dict(
        torch.load(best_path, map_location=device, weights_only=False)["model_state"]
    )
    test_m = eval_retrieval(test_pairs)
    print(f"[task4] test retrieval: {test_m}")

    # 10 qualitative retrieval examples (caption -> top-3 audio)
    g_emb, t_emb, stems = embed_split(test_pairs)
    sim = t_emb @ g_emb.t()  # caption -> audio
    caption_by_stem = {p["stem"]: p["caption"] for p in test_pairs}
    examples = []
    n_ex = min(10, len(test_pairs))
    for i in range(n_ex):
        top3 = sim[i].topk(min(3, sim.size(1))).indices.tolist()
        examples.append(
            {
                "query_caption": caption_by_stem[stems[i]][:300],
                "query_stem": stems[i],
                "top3_matched_clips": [
                    {
                        "stem": stems[j],
                        "caption": caption_by_stem[stems[j]][:200],
                        "score": float(sim[i, j].item()),
                        "is_correct": bool(j == i),
                    }
                    for j in top3
                ],
            }
        )
    retrieval_dir = Path(out_cfg.get("retrieval_dir", "results/retrieval_examples"))
    retrieval_dir.mkdir(parents=True, exist_ok=True)
    examples_path = retrieval_dir / "task4_caption_to_audio_examples.json"
    with examples_path.open("w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2)

    # Zero-shot tag prediction from captions vs Task 3 supervised reference (from src.evaluate import macro_micro_f1

    tag_data = build_musiccaps_tag_proxy(
        csv_path,
        top_k=int(cfg.get("eval", {}).get("top_k_tags", 50)),
        seed=int(cfg.get("project", {}).get("seed", 42)),
    )
    id_to_tag = tag_data["id_to_tag"]
    tag_names = [id_to_tag[i] for i in range(len(id_to_tag))]
    model.eval()
    with torch.no_grad():
        tag_tok = tokenize_batch(tag_names, tokenizer, max_length=32)
        tag_emb = model.encode_text(
            tag_tok["input_ids"].to(device),
            tag_tok["attention_mask"].to(device),
        ).cpu()
        y_true, y_prob = [], []
        for row in tag_data["test"][:2000]:
            tok = tokenize_batch([row["caption"]], tokenizer, max_length=max_length)
            c_emb = model.encode_text(
                tok["input_ids"].to(device),
                tok["attention_mask"].to(device),
            ).cpu()
            scores = (c_emb @ tag_emb.t()).squeeze(0).numpy()
            prob = 1.0 / (1.0 + np.exp(-5.0 * scores))
            y_prob.append(prob.astype(np.float32))
            y_true.append(row["labels"])
        y_true_a = np.stack(y_true, axis=0)
        y_prob_a = np.stack(y_prob, axis=0)
        zs_metrics = macro_micro_f1(y_true_a, y_prob_a, threshold=0.5)

    metrics_file = Path(out_cfg.get("metrics_file", "results/metrics.json"))
    task3_ref = None
    task1_ref = None
    if metrics_file.exists():
        try:
            prev = json.loads(metrics_file.read_text(encoding="utf-8"))
            t3 = prev.get("task3_gnn_bert_fusion", {})
            best_mode = t3.get("best_mode")
            if best_mode and "ablations" in t3:
                task3_ref = {
                    "best_mode": best_mode,
                    **t3["ablations"][best_mode].get("test", {}),
                }
            task1_ref = prev.get("task1_bert_musiccaps", {}).get("test")
        except json.JSONDecodeError:
            prev = {}
    else:
        prev = {}

    plots_dir = Path(out_cfg.get("plots_dir", "results/plots"))
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plots_dir / "task4_retrieval_r_at_k.png"
    ks = [1, 5, 10]
    c2a = [test_m.get(f"caption2audio_R@{k}", 0.0) for k in ks]
    a2c = [test_m.get(f"audio2caption_R@{k}", 0.0) for k in ks]
    x = np.arange(len(ks))
    plt.figure(figsize=(6, 4))
    plt.bar(x - 0.15, c2a, width=0.3, label="Caption->Audio")
    plt.bar(x + 0.15, a2c, width=0.3, label="Audio->Caption")
    plt.xticks(x, [f"R@{k}" for k in ks])
    plt.ylim(0, 1)
    plt.ylabel("Recall")
    plt.title("Task 4 MusicCaps retrieval")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    payload_update = {
        "task4_contrastive_musiccaps": {
            "n_pairs": {"train": len(train_pairs), "val": len(val_pairs), "test": len(test_pairs)},
            "test_retrieval": test_m,
            "history": history,
            "checkpoint": str(best_path),
            "retrieval_examples": str(examples_path),
            "retrieval_plot": str(plot_path),
            "zero_shot_tag_from_captions": zs_metrics,
            "task3_supervised_tag_reference": task3_ref,
            "task1_supervised_tag_reference": task1_ref,
        }
    }
    if isinstance(prev, dict):
        prev.update(payload_update)
        payload = prev
    else:
        payload = payload_update
    save_metrics(payload, metrics_file)
    print(f"[task4] zero-shot tag F1: {zs_metrics} | Task3 ref: {task3_ref}")
    print(f"[task4] wrote {metrics_file}, {examples_path}, {plot_path}")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train GNN-BERT music context models."
    )
    parser.add_argument("--task", type=int, required=True, choices=[1, 2, 3, 4])
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.task == 1:
        train_task1(cfg)
    elif args.task == 2:
        from src.train_tasks_23 import train_task2 as _t2

        _t2(cfg)
    elif args.task == 3:
        from src.train_tasks_23 import train_task3 as _t3

        _t3(cfg)
    else:
        train_task4(cfg)

if __name__ == "__main__":
    main()
