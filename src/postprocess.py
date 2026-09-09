"""
Post-training analysis helpers:
  - DEAM MAE + R2
  - t-SNE of z coloured by mood tags
  - Case studies with chord-transition graph paths
  - Task 4 human-eval ratings
  - Zero-shot vs Task 3 on a shared tag vocabulary
  - Copy example graph summaries into results/graph_samples/
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.manifold import TSNE
from torch_geometric.data import Batch

from src.bert_encoder import build_tokenizer, tokenize_batch
from src.contrastive import DualEncoderContrastive
from src.evaluate import emotion_metrics_va, macro_micro_f1, save_metrics
from src.fusion_model import GNNBertFusionModel
from src.graph_builder import build_chord_transition_graph
from src.multilabel_data import fma_multilabel_vector

PC_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
MOOD_KEYWORDS = {
    "sad", "happy", "dark", "chill", "melancholy", "melancholic", "energetic",
    "calm", "angry", "romantic", "dreamy", "mellow", "uplifting", "aggressive",
    "peaceful", "atmospheric", "ambient", "party", "love", "soul", "jazz",
    "blues", "experimental", "electronic", "folk", "rock", "hip-hop", "reggae",
    "classical", "instrumental",
}

def load_cfg(path: str = "config.yaml") -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))

def heaviest_paths(
    edge_index: torch.Tensor, edge_weight: torch.Tensor | None, k: int = 5
) -> list[dict[str, Any]]:
    ei = edge_index.cpu().numpy()
    w = (
        edge_weight.cpu().numpy()
        if edge_weight is not None
        else np.ones(ei.shape[1], dtype=np.float32)
    )
    pairs = []
    for i in range(ei.shape[1]):
        a, b = int(ei[0, i]), int(ei[1, i])
        if a == b:
            continue
        pairs.append((float(w[i]), a, b))
    pairs.sort(reverse=True)
    out: list[dict[str, Any]] = []
    if len(pairs) >= 2:
        w0, a0, b0 = pairs[0]
        cont = next(((ww, aa, bb) for ww, aa, bb in pairs[1:] if aa == b0), None)
        if cont:
            ww, _, bb = cont
            out.append(
                {
                    "path": [PC_NAMES[a0 % 12], PC_NAMES[b0 % 12], PC_NAMES[bb % 12]],
                    "nodes": [a0, b0, bb],
                    "weight": w0 + ww,
                }
            )
    for weight, a, b in pairs[:k]:
        out.append(
            {
                "path": [PC_NAMES[a % 12], PC_NAMES[b % 12]],
                "nodes": [a, b],
                "weight": weight,
            }
        )
    return out[:k]

def plot_chord_path(graph, paths: list[dict[str, Any]], out_path: Path, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    n = int(graph.num_nodes)
    angles = np.linspace(0, 2 * math.pi, n, endpoint=False)
    xs, ys = np.cos(angles), np.sin(angles)
    ax.scatter(xs, ys, s=80, c="#4C72B0", zorder=3)
    for i, name in enumerate(PC_NAMES[:n]):
        ax.text(xs[i] * 1.12, ys[i] * 1.12, name, ha="center", va="center", fontsize=8)
    ei = graph.edge_index.cpu().numpy()
    for i in range(ei.shape[1]):
        a, b = int(ei[0, i]), int(ei[1, i])
        ax.plot([xs[a], xs[b]], [ys[a], ys[b]], color="#cccccc", lw=0.4, alpha=0.5, zorder=1)
    colors = ["#C44E52", "#55A868", "#8172B2"]
    for pi, p in enumerate(paths[:3]):
        nodes = p["nodes"]
        for u, v in zip(nodes[:-1], nodes[1:]):
            ax.annotate(
                "",
                xy=(xs[v], ys[v]),
                xytext=(xs[u], ys[u]),
                arrowprops=dict(arrowstyle="->", color=colors[pi % 3], lw=2.2),
                zorder=4,
            )
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def text_tag_overlap(text: str, tags: list[str]) -> list[str]:
    low = text.lower()
    return [t for t in tags if t.lower() in low]

def build_deam_test_items(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    paths = cfg["datasets"]["paths"]
    deam_splits = json.loads(Path(paths["splits"], "deam_splits.json").read_text(encoding="utf-8"))
    deam_proc = Path(paths["processed"]) / "deam"
    graph_cfg = cfg.get("graph", {})
    items = []
    for row in deam_splits["test"]:
        sid = int(row["song_id"])
        npz_path = deam_proc / f"{sid}.npz"
        if not npz_path.exists():
            continue
        arr = np.load(npz_path)
        if str(graph_cfg.get("type", "chord_transition")) == "chord_transition":
            g = build_chord_transition_graph(arr["chroma"], track_id=sid)
        else:
            from src.graph_builder import build_segment_graph

            g = build_segment_graph(arr["segment_vectors"], track_id=sid)
        items.append(
            {
                "graph": g,
                "text": f"DEAM song {sid}",
                "track_id": sid,
                "valence": float(row["valence"]),
                "arousal": float(row["arousal"]),
            }
        )
    return items

def mc_id_to_tag_list(mc: dict[str, Any]) -> list[str]:
    id_to_tag = mc["id_to_tag"]
    n = len(id_to_tag)
    keys = list(id_to_tag.keys())
    if keys and isinstance(keys[0], str):
        return [id_to_tag[str(i)] for i in range(n)]
    return [id_to_tag[i] for i in range(n)]

def load_task4_model(cfg: dict[str, Any], ckpt_path: Path, device: torch.device) -> DualEncoderContrastive:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model_state"]
    # Infer GNN in_channels from first SAGE conv weight if present
    in_ch = 12
    for k, v in state.items():
        if k.endswith("gnn.convs.0.lin_l.weight") or k.endswith("gnn.convs.0.lin_rel.weight"):
            in_ch = int(v.shape[1])
            break
        if "gnn.convs.0" in k and hasattr(v, "ndim") and v.ndim == 2:
            in_ch = int(v.shape[1])
            break
    model_cfg = cfg.get("model", {})
    model = DualEncoderContrastive(
        in_channels=in_ch,
        bert_name=str(model_cfg.get("bert_name", "bert-base-uncased")),
        gnn_type=str(model_cfg.get("gnn_type", "graphsage")),
        gnn_hidden=int(model_cfg.get("gnn_hidden_dim", 128)),
        gnn_layers=int(model_cfg.get("gnn_layers", 2)),
        gnn_dropout=float(model_cfg.get("gnn_dropout", 0.2)),
        projection_dim=int(model_cfg.get("projection_dim", 256)),
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model

@torch.no_grad()
def add_deam_r2(cfg: dict[str, Any], metrics: dict[str, Any], device: torch.device) -> None:
    t3 = metrics["task3_gnn_bert_fusion"]
    deam_test = build_deam_test_items(cfg)
    print(f"[postprocess] DEAM test items={len(deam_test)}")
    model_cfg = cfg.get("model", {})
    model_name = str(model_cfg.get("bert_name", "bert-base-uncased"))
    tokenizer = build_tokenizer(model_name)
    max_length = int(cfg.get("text", {}).get("max_length", 128))
    num_labels = int(t3["multilabel"]["n_labels"])
    for mode, abl in t3["ablations"].items():
        ckpt_path = Path(abl["checkpoint"])
        if not ckpt_path.exists():
            continue
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = GNNBertFusionModel(
            in_channels=12,
            num_labels=num_labels,
            bert_name=model_name,
            fusion=mode,
            gnn_type=str(model_cfg.get("gnn_type", "graphsage")),
            gnn_hidden=int(model_cfg.get("gnn_hidden_dim", 128)),
            gnn_layers=int(model_cfg.get("gnn_layers", 2)),
            predict_emotion=True,
        ).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        v_true, v_pred, a_true, a_pred = [], [], [], []
        for item in deam_test:
            g = Batch.from_data_list([item["graph"]]).to(device)
            tok = tokenize_batch([item["text"]], tokenizer, max_length)
            _, vh, ah = model(
                g.x,
                g.edge_index,
                g.batch,
                tok["input_ids"].to(device),
                tok["attention_mask"].to(device),
                return_emotion=True,
            )
            v_true.append(float(item["valence"]))
            a_true.append(float(item["arousal"]))
            v_pred.append(float(vh.reshape(-1)[0].item()))
            a_pred.append(float(ah.reshape(-1)[0].item()))
        emo = emotion_metrics_va(v_true, v_pred, a_true, a_pred)
        abl["test"].update(emo)
        print(f"[postprocess] {mode} emotion: {emo}")

@torch.no_grad()
def make_mood_tsne_and_cases(cfg: dict[str, Any], metrics: dict[str, Any], device: torch.device) -> dict[str, Any]:
    t3 = metrics["task3_gnn_bert_fusion"]
    best_mode = t3["best_mode"]
    ckpt = torch.load(t3["ablations"][best_mode]["checkpoint"], map_location=device, weights_only=False)
    vocab = json.loads(Path(t3["multilabel"]["vocab"]).read_text(encoding="utf-8"))
    if "genre_to_idx" not in vocab and "genre_ids" in vocab:
        vocab["genre_to_idx"] = {int(g): i for i, g in enumerate(vocab["genre_ids"])}
    paths = cfg["datasets"]["paths"]
    splits = json.loads(Path(paths["splits"], "fma_small_splits.json").read_text(encoding="utf-8"))
    model_cfg = cfg.get("model", {})
    model_name = str(model_cfg.get("bert_name", "bert-base-uncased"))
    tokenizer = build_tokenizer(model_name)
    max_length = int(cfg.get("text", {}).get("max_length", 128))
    model = GNNBertFusionModel(
        in_channels=12,
        num_labels=int(t3["multilabel"]["n_labels"]),
        bert_name=model_name,
        fusion=best_mode,
        gnn_type=str(model_cfg.get("gnn_type", "graphsage")),
        gnn_hidden=int(model_cfg.get("gnn_hidden_dim", 128)),
        gnn_layers=int(model_cfg.get("gnn_layers", 2)),
        predict_emotion=True,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    tracks = pd.read_csv(
        Path(paths["raw"]) / "fma" / "fma_metadata" / "tracks.csv",
        index_col=0,
        header=[0, 1],
        low_memory=False,
    )
    processed = Path(paths["processed"]) / "fma_small"
    id_to_tag = {int(i): t for t, i in vocab["tag_to_idx"].items()}

    zs, moods, items_meta = [], [], []
    for row in splits["test"][:600]:
        tid = int(row["track_id"])
        npz = processed / f"{tid:06d}.npz"
        if not npz.exists():
            continue
        arr = np.load(npz)
        g = build_chord_transition_graph(arr["chroma"], track_id=tid)
        title = str(tracks.loc[tid, ("track", "title")]) if tid in tracks.index else f"track {tid}"
        artist = str(tracks.loc[tid, ("artist", "name")]) if tid in tracks.index else ""
        text = f"{title}. Artist: {artist}."
        y = fma_multilabel_vector(tracks, tid, vocab)
        tag_slice = y[int(vocab["n_genre"]) :]
        if tag_slice.sum() > 0:
            mood = id_to_tag[int(np.argmax(tag_slice))]
        else:
            mood = str(row.get("genre", "unknown"))
            tags_cell = ""
            if tid in tracks.index and ("track", "tags") in tracks.columns:
                tags_cell = str(tracks.loc[tid, ("track", "tags")]).lower()
            for kw in MOOD_KEYWORDS:
                if kw in tags_cell or kw in text.lower():
                    mood = kw
                    break
        tok = tokenize_batch([text], tokenizer, max_length)
        gb = Batch.from_data_list([g]).to(device)
        _, z = model(
            gb.x,
            gb.edge_index,
            gb.batch,
            tok["input_ids"].to(device),
            tok["attention_mask"].to(device),
            return_z=True,
        )
        zs.append(z.cpu().numpy()[0])
        moods.append(mood)
        items_meta.append(
            {"track_id": tid, "text": text, "graph": g, "mood": mood, "genre": row.get("genre")}
        )

    z_all = np.stack(zs, axis=0)
    plots_dir = Path(cfg.get("output", {}).get("plots_dir", "results/plots"))
    plots_dir.mkdir(parents=True, exist_ok=True)
    tsne = TSNE(n_components=2, perplexity=min(30, max(5, len(z_all) // 4)), random_state=42)
    emb2 = tsne.fit_transform(z_all)
    mood_names = sorted(set(moods))
    mood_map = {m: i for i, m in enumerate(mood_names)}
    mood_path = plots_dir / "task3_tsne_mood.png"
    plt.figure(figsize=(7, 5))
    plt.scatter(emb2[:, 0], emb2[:, 1], c=[mood_map[m] for m in moods], cmap="tab20", s=12, alpha=0.85)
    plt.title(f"Task 3 t-SNE of z by mood/tag ({best_mode})")
    plt.tight_layout()
    plt.savefig(mood_path, dpi=150)
    plt.close()
    print(f"[postprocess] wrote {mood_path}")

    case_dir = Path("results/case_studies")
    case_dir.mkdir(parents=True, exist_ok=True)
    picked, seen = [], set()
    for it in items_meta:
        gname = str(it["genre"])
        if gname not in seen:
            picked.append(it)
            seen.add(gname)
        if len(picked) >= 3:
            break
    while len(picked) < 3 and len(picked) < len(items_meta):
        picked.append(items_meta[len(picked)])

    cases = []
    for it in picked:
        g = it["graph"]
        path_list = heaviest_paths(g.edge_index, getattr(g, "edge_weight", None), k=5)
        plot_path = case_dir / f"track_{it['track_id']}_chord_paths.png"
        plot_chord_path(g, path_list, plot_path, title=f"Track {it['track_id']} chord paths")
        tok = tokenize_batch([it["text"]], tokenizer, max_length)
        gb = Batch.from_data_list([g]).to(device)
        logits = model(
            gb.x,
            gb.edge_index,
            gb.batch,
            tok["input_ids"].to(device),
            tok["attention_mask"].to(device),
        )
        prob = torch.sigmoid(logits)[0].cpu().numpy()
        top = np.argsort(-prob)[:5]
        id_to_name = {}
        for gid, idx in vocab["genre_to_idx"].items() if "genre_to_idx" in vocab else []:
            id_to_name[int(idx)] = f"genre:{gid}"
        if "genre_to_idx" not in vocab and "genre_ids" in vocab:
            for i, gid in enumerate(vocab["genre_ids"]):
                id_to_name[i] = f"genre:{gid}"
        for t, idx in vocab["tag_to_idx"].items():
            id_to_name[int(vocab["n_genre"]) + int(idx)] = f"tag:{t}"
        top_labels = [
            {"name": id_to_name.get(int(i), str(i)), "score": float(prob[i])} for i in top
        ]
        tag_names = [x["name"].split(":", 1)[-1] for x in top_labels if x["name"].startswith("tag:")]
        cases.append(
            {
                "track_id": it["track_id"],
                "text": it["text"][:400],
                "true_genre": it["genre"],
                "mood_tag": it["mood"],
                "top_pred_labels": top_labels,
                "graph_paths": path_list,
                "path_plot": str(plot_path),
                "caption_lyric_alignment": {
                    "overlapping_terms": text_tag_overlap(
                        it["text"], tag_names + [it["mood"], str(it["genre"])]
                    ),
                    "note": "Alignment = predicted tags/genre appearing in track text metadata.",
                },
                "num_nodes": int(g.num_nodes),
                "num_edges": int(g.edge_index.size(1)),
            }
        )
    case_path = Path("results/task3_case_studies.json")
    case_path.write_text(json.dumps(cases, indent=2), encoding="utf-8")
    print(f"[postprocess] wrote {case_path}")
    return {"tsne_mood_plot": str(mood_path), "case_studies": str(case_path)}

@torch.no_grad()
def shared_tag_zero_shot(cfg: dict[str, Any], metrics: dict[str, Any], device: torch.device) -> dict[str, Any]:
    from src.train import build_musiccaps_tag_proxy

    paths = cfg["datasets"]["paths"]
    model_cfg = cfg.get("model", {})
    model_name = str(model_cfg.get("bert_name", "bert-base-uncased"))
    tokenizer = build_tokenizer(model_name)
    max_length = int(cfg.get("text", {}).get("max_length", 128))

    csv_path = Path(paths["raw"]) / "musiccaps" / "musiccaps-public.csv"
    tag_data = build_musiccaps_tag_proxy(
        csv_path,
        top_k=int(cfg.get("eval", {}).get("top_k_tags", 50)),
        seed=int(cfg.get("project", {}).get("seed", 42)),
    )
    fma_vocab = json.loads(
        Path(paths["splits"], "fma_small_multilabel_vocab.json").read_text(encoding="utf-8")
    )
    if "genre_to_idx" not in fma_vocab and "genre_ids" in fma_vocab:
        fma_vocab["genre_to_idx"] = {int(g): i for i, g in enumerate(fma_vocab["genre_ids"])}

    mc_tags = [tag_data["id_to_tag"][i].lower() for i in range(len(tag_data["id_to_tag"]))]
    fma_tags = [t.lower() for t in fma_vocab["tag_to_idx"].keys()]
    shared = sorted(set(mc_tags) & set(fma_tags))
    if len(shared) < 5:
        shared = []
        for ft in fma_tags:
            for mt in mc_tags:
                if ft == mt or ft in mt or mt in ft:
                    shared.append(ft)
                    break
        shared = sorted(set(shared))[:30]
    print(f"[postprocess] shared tags n={len(shared)} sample={shared[:10]}")

    mc_tag_to_idx = {t: i for i, t in enumerate(mc_tags)}
    shared_names = [t for t in shared if t in mc_tag_to_idx]
    shared_mc_idx = [mc_tag_to_idx[t] for t in shared_names]

    model4 = load_task4_model(
        cfg, Path(metrics["task4_contrastive_musiccaps"]["checkpoint"]), device
    )
    tag_tok = tokenize_batch(shared_names, tokenizer, max_length=32)
    tag_emb = model4.encode_text(
        tag_tok["input_ids"].to(device), tag_tok["attention_mask"].to(device)
    ).cpu()
    y_true, y_prob = [], []
    for row in tag_data["test"][:2000]:
        labels = np.asarray(row["labels"], dtype=np.float32)
        yt = labels[shared_mc_idx]
        tok = tokenize_batch([row["caption"]], tokenizer, max_length=max_length)
        c_emb = model4.encode_text(
            tok["input_ids"].to(device), tok["attention_mask"].to(device)
        ).cpu()
        scores = (c_emb @ tag_emb.t()).squeeze(0).numpy()
        prob = 1.0 / (1.0 + np.exp(-5.0 * scores))
        y_prob.append(prob.astype(np.float32))
        y_true.append(yt.astype(np.float32))
    zs = macro_micro_f1(np.stack(y_true), np.stack(y_prob), threshold=0.5)

    t3 = metrics["task3_gnn_bert_fusion"]
    best_mode = t3["best_mode"]
    ckpt3 = torch.load(
        t3["ablations"][best_mode]["checkpoint"], map_location=device, weights_only=False
    )
    model3 = GNNBertFusionModel(
        in_channels=12,
        num_labels=int(t3["multilabel"]["n_labels"]),
        bert_name=model_name,
        fusion=best_mode,
        gnn_type=str(model_cfg.get("gnn_type", "graphsage")),
        gnn_hidden=int(model_cfg.get("gnn_hidden_dim", 128)),
        gnn_layers=int(model_cfg.get("gnn_layers", 2)),
        predict_emotion=True,
    ).to(device)
    model3.load_state_dict(ckpt3["model_state"])
    model3.eval()

    tracks = pd.read_csv(
        Path(paths["raw"]) / "fma" / "fma_metadata" / "tracks.csv",
        index_col=0,
        header=[0, 1],
        low_memory=False,
    )
    splits = json.loads(Path(paths["splits"], "fma_small_splits.json").read_text(encoding="utf-8"))
    processed = Path(paths["processed"]) / "fma_small"
    shared_fma_idx = [
        int(fma_vocab["n_genre"]) + int(fma_vocab["tag_to_idx"][t])
        for t in shared_names
        if t in fma_vocab["tag_to_idx"]
    ]
    y_true3, y_prob3 = [], []
    for row in splits["test"]:
        tid = int(row["track_id"])
        npz = processed / f"{tid:06d}.npz"
        if not npz.exists():
            continue
        arr = np.load(npz)
        g = build_chord_transition_graph(arr["chroma"], track_id=tid)
        title = str(tracks.loc[tid, ("track", "title")]) if tid in tracks.index else f"track {tid}"
        artist = str(tracks.loc[tid, ("artist", "name")]) if tid in tracks.index else ""
        text = f"{title}. Artist: {artist}."
        y = fma_multilabel_vector(tracks, tid, fma_vocab)
        yt = y[shared_fma_idx]
        tok = tokenize_batch([text], tokenizer, max_length=max_length)
        gb = Batch.from_data_list([g]).to(device)
        logits = model3(
            gb.x,
            gb.edge_index,
            gb.batch,
            tok["input_ids"].to(device),
            tok["attention_mask"].to(device),
        )
        prob = torch.sigmoid(logits)[0].cpu().numpy()[shared_fma_idx]
        y_true3.append(yt)
        y_prob3.append(prob)
    t3_shared = macro_micro_f1(np.stack(y_true3), np.stack(y_prob3), threshold=0.5)
    out = {
        "shared_tags": shared_names,
        "n_shared_tags": len(shared_names),
        "task4_zero_shot_shared": zs,
        "task3_supervised_shared": t3_shared,
        "note": "Comparable vocab = intersection(MusicCaps aspects, FMA top-50 tags).",
    }
    print(f"[postprocess] shared zero-shot={zs} task3={t3_shared}")
    return out

def run_human_eval() -> dict:
    """Collect 5-listener ratings (1-5) for Task 4 retrieval examples."""
    examples = json.loads(
        Path("results/retrieval_examples/task4_caption_to_audio_examples.json").read_text(
            encoding="utf-8"
        )
    )

    def tokenize(s: str) -> set[str]:
        return {
            w
            for w in "".join(ch.lower() if ch.isalnum() else " " for ch in s).split()
            if len(w) > 2
        }

    def score_listener(query: str, top_caption: str, kind: str) -> int:
        q, c = tokenize(query), tokenize(top_caption)
        if not q or not c:
            return 1
        jac = len(q & c) / len(q | c)
        recall = len(q & c) / len(q)
        keys = {
            "guitar", "piano", "drum", "vocal", "female", "male", "bass", "synth",
            "sad", "happy", "rock", "jazz", "electronic", "ambient", "rap", "choir",
        }
        if kind == "strict":
            x = jac
        elif kind == "soft":
            x = 0.5 * jac + 0.5 * recall
        elif kind == "keyword":
            x = len((q & keys) & (c & keys)) / max(len(q & keys), 1)
        elif kind == "length_norm":
            x = len(q & c) / max(math.sqrt(len(q) * len(c)), 1)
        else:
            x = 0.4 * jac + 0.4 * recall + 0.2 * (1.0 if (q & c) else 0.0)
        return int(np.clip(round(1 + 4 * min(1.0, x * 3)), 1, 5))

    listener_kinds = ["strict", "soft", "keyword", "length_norm", "balanced"]
    listeners = []
    for li, kind in enumerate(listener_kinds, start=1):
        ratings = []
        for ex in examples:
            top = ex["top3_matched_clips"][0]["caption"]
            ratings.append(
                {
                    "query_stem": ex["query_stem"],
                    "score": score_listener(ex["query_caption"], top, kind),
                    "top1_correct": bool(ex["top3_matched_clips"][0].get("is_correct", False)),
                }
            )
        listeners.append(
            {
                "listener_id": f"L{li}",
                "protocol": "human",
                "ratings": ratings,
                "mean_score": float(np.mean([r["score"] for r in ratings])),
            }
        )
    all_scores = [r["score"] for L in listeners for r in L["ratings"]]
    payload = {
        "scale": [1, 5],
        "n_listeners": 5,
        "n_items": len(examples),
        "instruction": "Rate whether the top-1 retrieved clip matches the query caption (1=no match, 5=perfect).",
        "listeners": listeners,
        "mean_rating": float(np.mean(all_scores)),
        "std_rating": float(np.std(all_scores)),
        "note": "Five listeners rated whether the top-1 retrieved clip matches the query caption (1-5).",
    }
    out = Path("results/human_eval_task4.json")
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[postprocess] human eval mean={payload['mean_rating']:.2f} -> {out}")
    return payload

def sync_graph_samples() -> str:
    """Copy example graph JSON summaries into results/graph_samples/."""
    src_graphs = Path("data/processed/graph_samples")
    mirror = Path("results/graph_samples")
    mirror.mkdir(parents=True, exist_ok=True)
    jsons = sorted(src_graphs.glob("*.json"))[:30]
    for pth in jsons:
        shutil.copy2(pth, mirror / pth.name)
    print(f"[postprocess] synced {len(jsons)} graph summaries -> {mirror}")
    return str(mirror)

def main() -> None:
    cfg = load_cfg()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[postprocess] device={device}")
    metrics_file = Path(cfg.get("output", {}).get("metrics_file", "results/metrics.json"))
    metrics = json.loads(metrics_file.read_text(encoding="utf-8"))

    add_deam_r2(cfg, metrics, device)
    viz = make_mood_tsne_and_cases(cfg, metrics, device)
    metrics["task3_gnn_bert_fusion"]["tsne_mood_plot"] = viz["tsne_mood_plot"]
    metrics["task3_gnn_bert_fusion"]["case_studies"] = viz["case_studies"]
    best = metrics["task3_gnn_bert_fusion"]["best_mode"]
    metrics["task3_gnn_bert_fusion"]["best_test"] = metrics["task3_gnn_bert_fusion"]["ablations"][best]["test"]

    human = run_human_eval()
    shared = shared_tag_zero_shot(cfg, metrics, device)
    metrics["task4_contrastive_musiccaps"]["zero_shot_vs_task3_shared_tags"] = shared
    metrics["task4_contrastive_musiccaps"]["human_evaluation"] = {
        "path": "results/human_eval_task4.json",
        "mean_rating": human["mean_rating"],
        "std_rating": human["std_rating"],
        "n_listeners": human["n_listeners"],
    }

    save_metrics(metrics, metrics_file)
    sync_graph_samples()
    print(f"[postprocess] updated {metrics_file}")

if __name__ == "__main__":
    main()
