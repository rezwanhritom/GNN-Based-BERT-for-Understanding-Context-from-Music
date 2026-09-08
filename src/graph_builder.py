"""
Music structure graph construction.

Per PDF Section 3 (Graph construction):
  - Chord-transition graph: nodes = unique chords; edges = transitions weighted by count
  - Segment graph: nodes = time segments;
    edges = temporal adjacency + cosine similarity of MFCC/chroma > τ
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch_geometric.data import Data


def _cosine_matrix(x: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity for rows of x (n, d)."""
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    x_n = x / norms
    return x_n @ x_n.T


def build_segment_graph(
    segment_features: np.ndarray | list[Any],
    similarity_threshold: float = 0.7,
    temporal_edges: bool = True,
    similarity_edges: bool = True,
    y: int | None = None,
    track_id: int | None = None,
    **kwargs: Any,
) -> Data:
    """
    Segment graph G = (V, E).
    Nodes = time segments; node features = segment vectors.
    Edges = temporal adjacency + cosine similarity > τ.
    Returns a PyTorch Geometric Data object.
    """
    if isinstance(segment_features, list):
        x_np = np.stack([np.asarray(s, dtype=np.float32).reshape(-1) for s in segment_features], axis=0)
    else:
        x_np = np.asarray(segment_features, dtype=np.float32)
        if x_np.ndim == 1:
            x_np = x_np.reshape(1, -1)

    n = int(x_np.shape[0])
    edges: set[tuple[int, int]] = set()

    if temporal_edges and n >= 2:
        for i in range(n - 1):
            edges.add((i, i + 1))
            edges.add((i + 1, i))

    if similarity_edges and n >= 2:
        sim = _cosine_matrix(x_np)
        for i in range(n):
            for j in range(i + 1, n):
                if sim[i, j] > similarity_threshold:
                    edges.add((i, j))
                    edges.add((j, i))

    if not edges:
        # Single-node or isolated — self-loop so GNN message passing is defined
        edges.add((0, 0))

    edge_index = torch.tensor(sorted(edges), dtype=torch.long).t().contiguous()
    x = torch.from_numpy(x_np)
    data = Data(x=x, edge_index=edge_index)
    if y is not None:
        data.y = torch.tensor([int(y)], dtype=torch.long)
    if track_id is not None:
        data.track_id = torch.tensor([int(track_id)], dtype=torch.long)
    data.num_nodes = n
    return data


def build_chord_transition_graph(
    chroma_or_chords: Any,
    n_chords: int = 12,
    y: int | None = None,
    track_id: int | None = None,
    **kwargs: Any,
) -> Data:
    """
    Chord-transition graph: nodes = pitch-class / chord bins;
    edges = observed transitions weighted by count.
    `chroma_or_chords`: chroma (12, T) → argmax over time as chord sequence.
    """
    arr = np.asarray(chroma_or_chords, dtype=np.float32)
    if arr.ndim == 2 and arr.shape[0] == n_chords:
        chord_seq = np.argmax(arr, axis=0)
    else:
        chord_seq = arr.astype(np.int64).reshape(-1)

    counts = np.zeros((n_chords, n_chords), dtype=np.float32)
    for a, b in zip(chord_seq[:-1], chord_seq[1:]):
        counts[int(a), int(b)] += 1.0

    # Node features: outgoing transition histogram (+ self count)
    x = torch.from_numpy(counts.copy())
    src, dst = np.nonzero(counts)
    if len(src) == 0:
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        edge_weight = torch.tensor([1.0], dtype=torch.float32)
    else:
        edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)
        edge_weight = torch.from_numpy(counts[src, dst])

    data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight)
    if y is not None:
        data.y = torch.tensor([int(y)], dtype=torch.long)
    if track_id is not None:
        data.track_id = torch.tensor([int(track_id)], dtype=torch.long)
    data.num_nodes = n_chords
    return data


def save_graph(graph: Data, path: str | Path) -> None:
    """Save graph sample as .pt (submission: ≥ 20 example graphs)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(graph, path)


def load_graph(path: str | Path) -> Data:
    """Load a saved graph sample."""
    return torch.load(Path(path), map_location="cpu", weights_only=False)


def save_graph_json_summary(graph: Data, path: str | Path) -> None:
    """Lightweight JSON summary alongside .pt samples."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "num_nodes": int(graph.num_nodes) if graph.num_nodes is not None else int(graph.x.size(0)),
        "num_edges": int(graph.edge_index.size(1)),
        "x_dim": int(graph.x.size(1)),
        "y": int(graph.y.item()) if hasattr(graph, "y") and graph.y is not None else None,
        "track_id": int(graph.track_id.item())
        if hasattr(graph, "track_id") and graph.track_id is not None
        else None,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
