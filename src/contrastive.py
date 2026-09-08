"""
Contrastive dual-encoder GNN–BERT for MusicCaps (Task 4).

Per PDF Section 4.4 and Algorithm 4 (InfoNCE):
  g_i ← Normalize(GNN(G_i))
  t_i ← Normalize(BERT_CLS(caption_i))
  S_ij = g_i^T t_j / τ
  L_NCE = −(1/N) Σ_i log[ exp(S_ii) / Σ_j exp(S_ij) ]
  Metrics: Caption→Audio and Audio→Caption R@1, R@5, R@10
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from src.gnn_model import MusicGAT, MusicGraphSAGE


class DualEncoderContrastive(nn.Module):
    """Shared embedding space between audio graphs and captions."""

    def __init__(
        self,
        in_channels: int,
        bert_name: str = "bert-base-uncased",
        gnn_type: str = "graphsage",
        gnn_hidden: int = 128,
        gnn_layers: int = 2,
        gnn_dropout: float = 0.2,
        projection_dim: int = 256,
        temperature: float = 0.07,
        freeze_bert: bool = False,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        if gnn_type == "gat":
            self.gnn = MusicGAT(
                in_channels, gnn_hidden, gnn_layers, num_classes=gnn_hidden, dropout=gnn_dropout
            )
        else:
            self.gnn = MusicGraphSAGE(
                in_channels, gnn_hidden, gnn_layers, num_classes=gnn_hidden, dropout=gnn_dropout
            )
        self.bert = AutoModel.from_pretrained(bert_name)
        text_dim = int(self.bert.config.hidden_size)
        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False
        self.graph_proj = nn.Linear(gnn_hidden, projection_dim)
        self.text_proj = nn.Linear(text_dim, projection_dim)

    def encode_graph(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """g ← Normalize(proj(GNN(G)))."""
        g = self.gnn.encode(x, edge_index, batch)
        g = self.graph_proj(g)
        return F.normalize(g, dim=-1)

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """t ← Normalize(proj(BERT_CLS(caption)))."""
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        t = out.last_hidden_state[:, 0, :]
        t = self.text_proj(t)
        return F.normalize(t, dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        g = self.encode_graph(x, edge_index, batch)
        t = self.encode_text(input_ids, attention_mask)
        return g, t


def info_nce_loss(
    graph_emb: torch.Tensor,
    text_emb: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Symmetric InfoNCE for paired (graph, caption):
      L = 0.5 * (L_g2t + L_t2g)
    where sim(u, v) = u^T v (embeddings already L2-normalized).
    """
    logits = graph_emb @ text_emb.t() / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_g2t = F.cross_entropy(logits, labels)
    loss_t2g = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_g2t + loss_t2g)


def recall_at_k(
    similarity: torch.Tensor,
    k: int,
) -> float:
    """Retrieval R@K from similarity matrix (rows = queries, cols = gallery)."""
    # Diagonal is the positive pair when rows/cols are aligned
    n = similarity.size(0)
    topk = similarity.topk(k=min(k, similarity.size(1)), dim=1).indices
    targets = torch.arange(n, device=similarity.device).unsqueeze(1)
    hits = (topk == targets).any(dim=1).float()
    return float(hits.mean().item())


def retrieval_metrics(similarity: torch.Tensor, ks: list[int] | None = None) -> dict[str, float]:
    if ks is None:
        ks = [1, 5, 10]
    out: dict[str, float] = {}
    for k in ks:
        out[f"R@{k}"] = recall_at_k(similarity, k)
    return out
