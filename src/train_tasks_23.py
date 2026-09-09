"""
Task 2 (chord graphs) + Task 3 (multi-label + DEAM L_aux) training.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
from sklearn.metrics import average_precision_score, f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.bert_encoder import build_tokenizer, tokenize_batch
from src.evaluate import save_metrics
from src.fusion_model import GNNBertFusionModel
from src.gnn_model import CNNMelBaseline, MusicGAT, MusicGraphSAGE
from src.graph_builder import (
    build_chord_transition_graph,
    build_segment_graph,
    save_graph,
    save_graph_json_summary,
)
from src.multilabel_data import (
    build_fma_multilabel_vocab,
    fma_multilabel_vector,
    load_deam_static_annotations,
    write_deam_splits,
)
import yaml

def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        return yaml.safe_load(f)

def set_seed(seed: int) -> None:
    import random

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

def _build_graph_from_npz(
    arr: np.lib.npyio.NpzFile,
    graph_cfg: dict[str, Any],
    y: int | None = None,
    track_id: int | None = None,
) -> Any:
    gtype = str(graph_cfg.get("type", "chord_transition")).lower()
    if gtype == "chord_transition":
        return build_chord_transition_graph(arr["chroma"], y=y, track_id=track_id)
    return build_segment_graph(
        arr["segment_vectors"],
        similarity_threshold=float(graph_cfg.get("similarity_threshold", 0.7)),
        temporal_edges=bool(graph_cfg.get("temporal_edges", True)),
        similarity_edges=bool(graph_cfg.get("similarity_edges", True)),
        y=y,
        track_id=track_id,
    )

def train_task2(cfg: dict[str, Any]) -> None:
    """Task 2 - GraphSAGE/GAT on chord-transition graphs + CNN + majority."""
    from torch_geometric.loader import DataLoader as GeoDataLoader

    set_seed(int(cfg.get("project", {}).get("seed", 42)))
    device = resolve_device(cfg)
    print(f"[task2] device={device}")

    paths = cfg["datasets"]["paths"]
    primary = str(cfg["datasets"]["primary_audio"])
    processed_dir = Path(paths["processed"]) / primary
    splits_path = Path(paths["splits"]) / f"{primary}_splits.json"
    with splits_path.open(encoding="utf-8") as f:
        splits = json.load(f)
    graph_cfg = cfg.get("graph", {})

    def rows_to_graphs(rows: list[dict[str, Any]]) -> list[Any]:
        graphs = []
        for row in rows:
            tid = int(row["track_id"])
            npz_path = processed_dir / f"{tid:06d}.npz"
            if not npz_path.exists():
                continue
            arr = np.load(npz_path)
            g = _build_graph_from_npz(arr, graph_cfg, y=int(row["genre_id"]), track_id=tid)
            graphs.append(g)
        return graphs

    train_graphs = rows_to_graphs(splits["train"])
    val_graphs = rows_to_graphs(splits["val"])
    test_graphs = rows_to_graphs(splits["test"])
    num_classes = len(splits["genre_to_id"])
    in_channels = int(train_graphs[0].x.size(1))
    print(
        f"[task2] graph_type={graph_cfg.get('type')} train={len(train_graphs)} "
        f"val={len(val_graphs)} test={len(test_graphs)} in_dim={in_channels}"
    )

    graph_samples_dir = Path(paths["processed"]) / "graph_samples"
    graph_samples_dir.mkdir(parents=True, exist_ok=True)
    for i, g in enumerate(train_graphs[:20]):
        save_graph(g, graph_samples_dir / f"chord_sample_{i:02d}.pt")
        save_graph_json_summary(g, graph_samples_dir / f"chord_sample_{i:02d}.json")
    if bool(graph_cfg.get("also_build_segment_graphs", True)):
        n_seg = 0
        for row in splits["train"]:
            if n_seg >= 20:
                break
            tid = int(row["track_id"])
            npz_path = processed_dir / f"{tid:06d}.npz"
            if not npz_path.exists():
                continue
            arr = np.load(npz_path)
            sg = build_segment_graph(
                arr["segment_vectors"],
                similarity_threshold=float(graph_cfg.get("similarity_threshold", 0.7)),
                temporal_edges=True,
                similarity_edges=True,
                y=int(row["genre_id"]),
                track_id=tid,
            )
            save_graph(sg, graph_samples_dir / f"segment_sample_{n_seg:02d}.pt")
            save_graph_json_summary(sg, graph_samples_dir / f"segment_sample_{n_seg:02d}.json")
            n_seg += 1
    print(f"[task2] wrote graph samples -> {graph_samples_dir}")

    train_cfg = cfg.get("train", {})
    batch_size = int(train_cfg.get("batch_size", 16))
    epochs = int(train_cfg.get("epochs", 20))
    lr = float(train_cfg.get("gnn_learning_rate", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 0.01))
    model_cfg = cfg.get("model", {})
    gnn_type = str(model_cfg.get("gnn_type", "graphsage")).lower()
    hidden = int(model_cfg.get("gnn_hidden_dim", 128))
    n_layers = int(model_cfg.get("gnn_layers", 2))
    dropout = float(model_cfg.get("gnn_dropout", 0.2))

    gnn_cls = MusicGAT if gnn_type == "gat" else MusicGraphSAGE
    gnn = gnn_cls(in_channels, hidden, n_layers, num_classes, dropout=dropout).to(device)

    train_loader = GeoDataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = GeoDataLoader(val_graphs, batch_size=batch_size, shuffle=False)
    test_loader = GeoDataLoader(test_graphs, batch_size=batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(gnn.parameters(), lr=lr, weight_decay=weight_decay)

    def eval_gnn(loader: Any) -> dict[str, float]:
        gnn.eval()
        ys, ps, losses = [], [], []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                logits = gnn(batch.x, batch.edge_index, batch.batch)
                losses.append(float(criterion(logits, batch.y).item()))
                pred = logits.argmax(dim=-1)
                ys.append(batch.y.cpu().numpy())
                ps.append(pred.cpu().numpy())
        y_true = np.concatenate(ys)
        y_pred = np.concatenate(ps)
        return {
            "loss": float(np.mean(losses)),
            "accuracy": float((y_true == y_pred).mean()),
            "macro_f1": float(
                f1_score(y_true, y_pred, average="macro", labels=list(range(num_classes)), zero_division=0)
            ),
        }

    out_cfg = cfg.get("output", {})
    ckpt_dir = Path(out_cfg.get("checkpoint_dir", "results/checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_gnn_path = ckpt_dir / "task2_gnn_best.pt"
    best_acc = -1.0
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        gnn.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"task2-gnn {epoch}/{epochs}"):
            batch = batch.to(device)
            opt.zero_grad(set_to_none=True)
            logits = gnn(batch.x, batch.edge_index, batch.batch)
            loss = criterion(logits, batch.y)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        val_m = eval_gnn(val_loader)
        history.append({"epoch": float(epoch), "train_loss": float(np.mean(losses)), **{f"val_{k}": v for k, v in val_m.items()}})
        print(f"[task2] epoch={epoch} loss={np.mean(losses):.4f} val_acc={val_m['accuracy']:.3f}")
        if val_m["accuracy"] > best_acc:
            best_acc = val_m["accuracy"]
            torch.save({"model_state": gnn.state_dict(), "gnn_type": gnn_type, "graph_type": graph_cfg.get("type")}, best_gnn_path)

    gnn.load_state_dict(torch.load(best_gnn_path, map_location=device, weights_only=False)["model_state"])
    test_gnn = eval_gnn(test_loader)

    # CNN mel baseline B2
    cnn = CNNMelBaseline(num_classes=num_classes, n_mels=int(cfg["audio"]["n_mels"])).to(device)
    cnn_opt = torch.optim.AdamW(cnn.parameters(), lr=lr, weight_decay=weight_decay)

    def load_mel_items(rows: list[dict[str, Any]]) -> list[tuple[np.ndarray, int]]:
        items = []
        for row in rows:
            npz_path = processed_dir / f"{int(row['track_id']):06d}.npz"
            if not npz_path.exists():
                continue
            arr = np.load(npz_path)
            items.append((np.asarray(arr["mel"], dtype=np.float32), int(row["genre_id"])))
        return items

    train_mel = load_mel_items(splits["train"])
    test_mel = load_mel_items(splits["test"])
    val_mel = load_mel_items(splits["val"])

    def mel_batches(items: list[tuple[np.ndarray, int]], shuffle: bool):
        idx = np.arange(len(items))
        if shuffle:
            np.random.default_rng(42).shuffle(idx)
        for start in range(0, len(idx), batch_size):
            sl = idx[start : start + batch_size]
            # pad time dim
            mels = [items[i][0] for i in sl]
            ys = torch.tensor([items[i][1] for i in sl], dtype=torch.long)
            t_max = max(m.shape[1] for m in mels)
            stacked = np.zeros((len(mels), mels[0].shape[0], t_max), dtype=np.float32)
            for j, m in enumerate(mels):
                stacked[j, :, : m.shape[1]] = m
            yield torch.from_numpy(stacked), ys

    def eval_cnn(items: list[tuple[np.ndarray, int]]) -> dict[str, float]:
        cnn.eval()
        ys, ps = [], []
        with torch.no_grad():
            for xb, yb in mel_batches(items, False):
                logits = cnn(xb.to(device))
                pred = logits.argmax(dim=-1).cpu().numpy()
                ys.append(yb.numpy())
                ps.append(pred)
        y_true = np.concatenate(ys)
        y_pred = np.concatenate(ps)
        return {
            "accuracy": float((y_true == y_pred).mean()),
            "macro_f1": float(
                f1_score(y_true, y_pred, average="macro", labels=list(range(num_classes)), zero_division=0)
            ),
        }

    best_cnn_path = ckpt_dir / "task2_cnn_best.pt"
    best_cnn_acc = -1.0
    cnn_hist = []
    for epoch in range(1, epochs + 1):
        cnn.train()
        losses = []
        for xb, yb in tqdm(list(mel_batches(train_mel, True)), desc=f"task2-cnn {epoch}/{epochs}"):
            cnn_opt.zero_grad(set_to_none=True)
            logits = cnn(xb.to(device))
            loss = criterion(logits, yb.to(device))
            loss.backward()
            cnn_opt.step()
            losses.append(float(loss.item()))
        val_m = eval_cnn(val_mel)
        cnn_hist.append({"epoch": float(epoch), "train_loss": float(np.mean(losses)), "val_accuracy": val_m["accuracy"]})
        if val_m["accuracy"] > best_cnn_acc:
            best_cnn_acc = val_m["accuracy"]
            torch.save({"model_state": cnn.state_dict()}, best_cnn_path)
        print(f"[task2-cnn] epoch={epoch} val_acc={val_m['accuracy']:.3f}")

    cnn.load_state_dict(torch.load(best_cnn_path, map_location=device, weights_only=False)["model_state"])
    test_cnn = eval_cnn(test_mel)

    train_labels = [int(r["genre_id"]) for r in splits["train"]]
    test_labels = [int(r["genre_id"]) for r in splits["test"] if (processed_dir / f"{int(r['track_id']):06d}.npz").exists()]
    majority = int(Counter(train_labels).most_common(1)[0][0])
    maj_pred = [majority] * len(test_labels)
    maj_acc = float(np.mean([1.0 if y == majority else 0.0 for y in test_labels]))
    maj_f1 = float(f1_score(test_labels, maj_pred, average="macro", labels=list(range(num_classes)), zero_division=0))

    plots_dir = Path(out_cfg.get("plots_dir", "results/plots"))
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plots_dir / "task2_gnn_vs_cnn.png"
    names = ["Majority", "GNN", "CNN mel"]
    accs = [maj_acc, test_gnn["accuracy"], test_cnn["accuracy"]]
    plt.figure(figsize=(6, 4))
    plt.bar(names, accs)
    plt.ylim(0, 1)
    plt.ylabel("Test accuracy")
    plt.title(f"Task 2 ({graph_cfg.get('type')}) vs baselines")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    metrics_file = Path(out_cfg.get("metrics_file", "results/metrics.json"))
    prev = json.loads(metrics_file.read_text(encoding="utf-8")) if metrics_file.exists() else {}
    prev["task2_fma_small"] = {
        "graph_type": graph_cfg.get("type"),
        "gnn": {"test": test_gnn, "best_val_accuracy": best_acc, "history": history, "checkpoint": str(best_gnn_path)},
        "cnn_mel_baseline": {"test": test_cnn, "best_val_accuracy": best_cnn_acc, "history": cnn_hist, "checkpoint": str(best_cnn_path)},
        "majority_baseline": {"test_accuracy": maj_acc, "test_macro_f1": maj_f1},
        "graph_samples_dir": str(graph_samples_dir),
        "comparison_plot": str(plot_path),
    }
    save_metrics(prev, metrics_file)
    print(f"[task2] done GNN={test_gnn} CNN={test_cnn} majority_acc={maj_acc:.3f}")

def _multilabel_metrics(y_true: np.ndarray, y_prob: np.ndarray, thr: float = 0.5) -> dict[str, float]:
    y_pred = (y_prob >= thr).astype(np.int32)
    macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    micro = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    try:
        auc = float(average_precision_score(y_true, y_prob, average="macro"))
    except ValueError:
        auc = 0.0
    return {"macro_f1": macro, "micro_f1": micro, "auc_pr": auc}

def train_task3(cfg: dict[str, Any]) -> None:
    """Task 3 - multi-label GNN-BERT fusion + DEAM emotion aux + ablations."""
    from torch_geometric.data import Batch

    set_seed(int(cfg.get("project", {}).get("seed", 42)))
    device = resolve_device(cfg)
    print(f"[task3] device={device}")

    paths = cfg["datasets"]["paths"]
    primary = str(cfg["datasets"]["primary_audio"])
    processed_dir = Path(paths["processed"]) / primary
    splits_path = Path(paths["splits"]) / f"{primary}_splits.json"
    with splits_path.open(encoding="utf-8") as f:
        splits = json.load(f)

    tracks = pd.read_csv(
        Path(paths["raw"]) / "fma" / "fma_metadata" / "tracks.csv",
        index_col=0,
        header=[0, 1],
    )
    subset = primary.replace("fma_", "")
    top_k = int(cfg.get("eval", {}).get("top_k_tags", 50))
    vocab = build_fma_multilabel_vocab(tracks, subset=subset, top_k_tags=top_k)
    vocab_path = Path(paths["splits"]) / f"{primary}_multilabel_vocab.json"
    with vocab_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "genre_ids": vocab["genre_ids"],
                "tag_to_idx": vocab["tag_to_idx"],
                "n_genre": vocab["n_genre"],
                "n_tag": vocab["n_tag"],
                "n_labels": vocab["n_labels"],
            },
            f,
            indent=2,
        )
    print(f"[task3] multilabel labels={vocab['n_labels']} (genres={vocab['n_genre']} tags={vocab['n_tag']})")

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

    def build_items(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        items = []
        for row in rows:
            tid = int(row["track_id"])
            npz_path = processed_dir / f"{tid:06d}.npz"
            if not npz_path.exists():
                continue
            arr = np.load(npz_path)
            g = _build_graph_from_npz(arr, graph_cfg, track_id=tid)
            y = fma_multilabel_vector(tracks, tid, vocab)
            items.append(
                {
                    "graph": g,
                    "text": track_text(tid),
                    "labels": y,
                    "track_id": tid,
                    "genre": str(row.get("genre", "")),
                    "has_emotion": False,
                    "valence": 0.0,
                    "arousal": 0.0,
                }
            )
        return items

    train_items = build_items(splits["train"])
    val_items = build_items(splits["val"])
    test_items = build_items(splits["test"])

    # DEAM L_aux items
    deam_splits_path = Path(paths["splits"]) / "deam_splits.json"
    if not deam_splits_path.exists():
        write_deam_splits(Path(paths["raw"]) / "deam", deam_splits_path)
    with deam_splits_path.open(encoding="utf-8") as f:
        deam_splits = json.load(f)
    deam_proc = Path(paths["processed"]) / "deam"
    deam_meta = load_deam_static_annotations(Path(paths["raw"]) / "deam")

    def build_deam_items(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        items = []
        for row in rows:
            sid = int(row["song_id"])
            npz_path = deam_proc / f"{sid}.npz"
            if not npz_path.exists():
                continue
            arr = np.load(npz_path)
            g = _build_graph_from_npz(arr, graph_cfg, track_id=sid)
            items.append(
                {
                    "graph": g,
                    "text": f"DEAM song {sid}",
                    "labels": np.zeros(vocab["n_labels"], dtype=np.float32),
                    "track_id": sid,
                    "genre": "",
                    "has_emotion": True,
                    "valence": float(row["valence"]),
                    "arousal": float(row["arousal"]),
                }
            )
        return items

    deam_train = build_deam_items(deam_splits["train"])
    deam_val = build_deam_items(deam_splits["val"])
    print(
        f"[task3] FMA train/val/test={len(train_items)}/{len(val_items)}/{len(test_items)} "
        f"DEAM train/val={len(deam_train)}/{len(deam_val)}"
    )
    if len(train_items) < 16:
        raise RuntimeError("Not enough FMA items for Task 3")

    model_cfg = cfg.get("model", {})
    text_cfg = cfg.get("text", {})
    train_cfg = cfg.get("train", {})
    model_name = str(model_cfg.get("bert_name", "bert-base-uncased"))
    max_length = int(text_cfg.get("max_length", 128))
    batch_size = int(train_cfg.get("batch_size", 16))
    epochs = int(train_cfg.get("epochs", 20))
    lr = float(train_cfg.get("learning_rate", 2e-5))
    weight_decay = float(train_cfg.get("weight_decay", 0.01))
    alpha = float(train_cfg.get("alpha_valence", 0.5))
    beta = float(train_cfg.get("beta_arousal", 0.5))
    tokenizer = build_tokenizer(model_name)
    in_channels = int(train_items[0]["graph"].x.size(1))
    num_labels = int(vocab["n_labels"])

    def make_loader(items: list[dict[str, Any]], shuffle: bool) -> DataLoader:
        class DS(Dataset):
            def __len__(self) -> int:
                return len(items)

            def __getitem__(self, idx: int) -> dict[str, Any]:
                return items[idx]

        def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
            graphs = Batch.from_data_list([b["graph"] for b in batch])
            tok = tokenize_batch([b["text"] for b in batch], tokenizer, max_length)
            return {
                "graph": graphs,
                "input_ids": tok["input_ids"],
                "attention_mask": tok["attention_mask"],
                "labels": torch.tensor(np.stack([b["labels"] for b in batch]), dtype=torch.float32),
                "has_emotion": torch.tensor([1.0 if b["has_emotion"] else 0.0 for b in batch]),
                "valence": torch.tensor([b["valence"] for b in batch], dtype=torch.float32),
                "arousal": torch.tensor([b["arousal"] for b in batch], dtype=torch.float32),
                "texts": [b["text"] for b in batch],
                "track_ids": [b["track_id"] for b in batch],
                "genres": [b["genre"] for b in batch],
            }

        return DataLoader(DS(), batch_size=batch_size, shuffle=shuffle, num_workers=0, collate_fn=collate)

    bce = nn.BCEWithLogitsLoss(reduction="none")
    mse = nn.MSELoss(reduction="none")

    def run_epoch(model: nn.Module, loader: DataLoader, opt: torch.optim.Optimizer | None) -> float:
        train_mode = opt is not None
        model.train(train_mode)
        losses = []
        for batch in loader:
            g = batch["graph"].to(device)
            labels = batch["labels"].to(device)
            has_e = batch["has_emotion"].to(device)
            v_true = batch["valence"].to(device)
            a_true = batch["arousal"].to(device)
            if train_mode:
                opt.zero_grad(set_to_none=True)
            logits, v_hat, a_hat = model(
                g.x,
                g.edge_index,
                g.batch,
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                return_emotion=True,
            )
            # tag loss only where not pure-emotion DEAM rows (or always; DEAM labels are zero)
            tag_loss = bce(logits, labels).mean(dim=1)
            tag_mask = 1.0 - has_e
            tag_term = (tag_loss * tag_mask).sum() / tag_mask.sum().clamp_min(1.0)
            v_term = (mse(v_hat, v_true) * has_e).sum() / has_e.sum().clamp_min(1.0)
            a_term = (mse(a_hat, a_true) * has_e).sum() / has_e.sum().clamp_min(1.0)
            loss = tag_term + alpha * v_term + beta * a_term
            if train_mode:
                loss.backward()
                opt.step()
            losses.append(float(loss.item()))
        return float(np.mean(losses)) if losses else 0.0

    @torch.no_grad()
    def eval_tags(model: nn.Module, items: list[dict[str, Any]]) -> dict[str, float]:
        model.eval()
        loader = make_loader(items, False)
        probs, truths = [], []
        for batch in loader:
            g = batch["graph"].to(device)
            logits = model(
                g.x,
                g.edge_index,
                g.batch,
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
            )
            probs.append(torch.sigmoid(logits).cpu().numpy())
            truths.append(batch["labels"].numpy())
        y_prob = np.concatenate(probs, axis=0)
        y_true = np.concatenate(truths, axis=0)
        return _multilabel_metrics(y_true, y_prob)

    @torch.no_grad()
    def eval_emotion(model: nn.Module, items: list[dict[str, Any]]) -> dict[str, float]:
        from src.evaluate import emotion_metrics_va

        if not items:
            return {
                "mae_valence": float("nan"),
                "mae_arousal": float("nan"),
                "r2_valence": float("nan"),
                "r2_arousal": float("nan"),
            }
        model.eval()
        loader = make_loader(items, False)
        v_true_all, v_pred_all, a_true_all, a_pred_all = [], [], [], []
        for batch in loader:
            g = batch["graph"].to(device)
            _, v_hat, a_hat = model(
                g.x,
                g.edge_index,
                g.batch,
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                return_emotion=True,
            )
            v_true_all.extend(batch["valence"].cpu().numpy().tolist())
            a_true_all.extend(batch["arousal"].cpu().numpy().tolist())
            v_pred_all.extend(v_hat.detach().cpu().numpy().reshape(-1).tolist())
            a_pred_all.extend(a_hat.detach().cpu().numpy().reshape(-1).tolist())
        return emotion_metrics_va(v_true_all, v_pred_all, a_true_all, a_pred_all)

    modes = ["bert_only", "gnn_only", "early_concat", "cross_attention"]
    out_cfg = cfg.get("output", {})
    ckpt_dir = Path(out_cfg.get("checkpoint_dir", "results/checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = Path(out_cfg.get("plots_dir", "results/plots"))
    plots_dir.mkdir(parents=True, exist_ok=True)
    ablation_results: dict[str, Any] = {}

    # Mix DEAM into training each epoch
    mixed_train = train_items + deam_train

    for mode in modes:
        best_path = ckpt_dir / f"task3_{mode}_best.pt"
        model = GNNBertFusionModel(
            in_channels=in_channels,
            num_labels=num_labels,
            bert_name=model_name,
            fusion=mode,
            gnn_type=str(model_cfg.get("gnn_type", "graphsage")),
            gnn_hidden=int(model_cfg.get("gnn_hidden_dim", 128)),
            gnn_layers=int(model_cfg.get("gnn_layers", 2)),
            gnn_dropout=float(model_cfg.get("gnn_dropout", 0.2)),
            freeze_bert=bool(model_cfg.get("freeze_bert", False)),
            predict_emotion=True,
        ).to(device)
        # BERT modes use small LR; gnn_only can use higher
        mode_lr = lr if mode != "gnn_only" else float(train_cfg.get("gnn_learning_rate", 1e-3))
        opt = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=mode_lr,
            weight_decay=weight_decay,
        )
        best_f1 = -1.0
        history = []
        train_loader = make_loader(mixed_train, True)
        for epoch in range(1, epochs + 1):
            tr_loss = run_epoch(model, train_loader, opt)
            val_m = eval_tags(model, val_items)
            emo_m = eval_emotion(model, deam_val) if deam_val else {}
            history.append({"epoch": float(epoch), "train_loss": tr_loss, **val_m, **emo_m})
            print(
                f"[task3 {mode}] epoch={epoch} loss={tr_loss:.4f} "
                f"val_macro_f1={val_m['macro_f1']:.3f} emo={emo_m}"
            )
            if val_m["macro_f1"] > best_f1:
                best_f1 = val_m["macro_f1"]
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "fusion": mode,
                        "num_labels": num_labels,
                        "vocab_path": str(vocab_path),
                        "in_channels": in_channels,
                    },
                    best_path,
                )

        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        test_m = eval_tags(model, test_items)
        test_emo = eval_emotion(model, build_deam_items(deam_splits["test"]))
        ablation_results[mode] = {
            "test": {**test_m, **test_emo},
            "best_val_macro_f1": best_f1,
            "history": history,
            "checkpoint": str(best_path),
        }
        print(f"[task3 {mode}] test={test_m} emotion={test_emo}")

    # t-SNE of best cross-attention z
    best_mode = max(ablation_results.keys(), key=lambda m: ablation_results[m]["test"]["macro_f1"])
    best_ckpt = torch.load(ablation_results[best_mode]["checkpoint"], map_location=device, weights_only=False)
    model = GNNBertFusionModel(
        in_channels=in_channels,
        num_labels=num_labels,
        bert_name=model_name,
        fusion=best_ckpt["fusion"],
        gnn_type=str(model_cfg.get("gnn_type", "graphsage")),
        gnn_hidden=int(model_cfg.get("gnn_hidden_dim", 128)),
        gnn_layers=int(model_cfg.get("gnn_layers", 2)),
        gnn_dropout=float(model_cfg.get("gnn_dropout", 0.2)),
        predict_emotion=True,
    ).to(device)
    model.load_state_dict(best_ckpt["model_state"])
    model.eval()

    zs, genres = [], []
    with torch.no_grad():
        for batch in make_loader(test_items[:512], False):
            g = batch["graph"].to(device)
            _, z = model(
                g.x,
                g.edge_index,
                g.batch,
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                return_z=True,
            )
            zs.append(z.cpu().numpy())
            genres.extend(batch["genres"])
    z_all = np.concatenate(zs, axis=0)
    tsne = TSNE(n_components=2, perplexity=min(30, max(5, len(z_all) // 4)), random_state=42)
    emb2 = tsne.fit_transform(z_all)
    genre_names = sorted(set(genres))
    color_map = {g: i for i, g in enumerate(genre_names)}
    colors = [color_map.get(g, 0) for g in genres]
    tsne_path = plots_dir / "task3_tsne_genre.png"
    plt.figure(figsize=(7, 5))
    sc = plt.scatter(emb2[:, 0], emb2[:, 1], c=colors, cmap="tab10", s=12, alpha=0.8)
    plt.title(f"Task 3 t-SNE of z ({best_mode})")
    plt.tight_layout()
    plt.savefig(tsne_path, dpi=150)
    plt.close()

    abl_path = plots_dir / "task3_ablation_macro_f1.png"
    modes_sorted = list(ablation_results.keys())
    vals = [ablation_results[m]["test"]["macro_f1"] for m in modes_sorted]
    plt.figure(figsize=(7, 4))
    plt.bar(modes_sorted, vals)
    plt.ylim(0, 1)
    plt.ylabel("Test Macro-F1")
    plt.title("Task 3 ablations")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(abl_path, dpi=150)
    plt.close()

    # 3 case studies
    cases = []
    for item in test_items[:200]:
        if len(cases) >= 3:
            break
        g = Batch.from_data_list([item["graph"]]).to(device)
        tok = tokenize_batch([item["text"]], tokenizer, max_length)
        with torch.no_grad():
            logits = model(
                g.x,
                g.edge_index,
                g.batch,
                tok["input_ids"].to(device),
                tok["attention_mask"].to(device),
            )
            prob = torch.sigmoid(logits)[0].cpu().numpy()
        top = np.argsort(-prob)[:5]
        id_to_name = {}
        for gid, idx in vocab["genre_to_idx"].items():
            id_to_name[idx] = f"genre:{gid}"
        for t, idx in vocab["tag_to_idx"].items():
            id_to_name[vocab["n_genre"] + idx] = f"tag:{t}"
        cases.append(
            {
                "track_id": item["track_id"],
                "text": item["text"][:300],
                "true_genre": item["genre"],
                "top_pred_labels": [{"name": id_to_name.get(int(i), str(i)), "score": float(prob[i])} for i in top],
                "num_nodes": int(item["graph"].num_nodes),
                "num_edges": int(item["graph"].edge_index.size(1)),
                "graph_path_note": f"{graph_cfg.get('type')} graph; fusion={best_mode}.",
            }
        )
    case_path = Path(out_cfg.get("results_dir", "results")) / "task3_case_studies.json"
    with case_path.open("w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2)

    metrics_file = Path(out_cfg.get("metrics_file", "results/metrics.json"))
    prev = json.loads(metrics_file.read_text(encoding="utf-8")) if metrics_file.exists() else {}
    prev["task3_gnn_bert_fusion"] = {
        "dataset": primary,
        "graph_type": graph_cfg.get("type"),
        "multilabel": {"n_labels": num_labels, "n_genre": vocab["n_genre"], "n_tag": vocab["n_tag"], "vocab": str(vocab_path)},
        "deam_aux": {
            "n_train": len(deam_train),
            "n_val": len(deam_val),
            "alpha_valence": alpha,
            "beta_arousal": beta,
        },
        "ablations": ablation_results,
        "best_mode": best_mode,
        "tsne_plot": str(tsne_path),
        "ablation_plot": str(abl_path),
        "case_studies": str(case_path),
        "note": "Multi-label genre+tags with DEAM L_aux valence/arousal.",
    }
    save_metrics(prev, metrics_file)
    print(f"[task3] wrote metrics; best_mode={best_mode}")

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--task", type=int, choices=[2, 3], required=True)
    p.add_argument("--config", default="config.yaml")
    args = p.parse_args()
    cfg = load_config(args.config)
    if args.task == 2:
        train_task2(cfg)
    else:
        train_task3(cfg)
