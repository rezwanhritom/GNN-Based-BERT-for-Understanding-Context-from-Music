"""
GNN-BERT fusion for multi-context understanding (Task 3).

  Cross-attention fusion (recommended):
    A = softmax(Q K^T / √d),  Q = g W_Q,  K = H_text W_K
    z = CONCAT(g, A H_text),  ŷ = sigma(W z)
  Multi-task:
    L = L_tags + alpha||v - v̂||^2 + beta||a - â||^2
  Ablations: BERT-only, GNN-only, early concat, cross-attention.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel

from src.gnn_model import MusicGAT, MusicGraphSAGE

class CrossAttentionFusion(nn.Module):
    """Cross-attention between graph readout g and BERT token states H_text."""

    def __init__(self, graph_dim: int, text_dim: int, d_model: int = 256) -> None:
        super().__init__()
        self.d_model = d_model
        self.w_q = nn.Linear(graph_dim, d_model)
        self.w_k = nn.Linear(text_dim, d_model)
        self.w_v = nn.Linear(text_dim, d_model)
        self.out_dim = graph_dim + d_model

    def forward(
        self,
        g: torch.Tensor,
        h_text: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self.w_q(g).unsqueeze(1)
        k = self.w_k(h_text)
        v = self.w_v(h_text)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_model**0.5)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e4)
        a = torch.softmax(scores, dim=-1)
        attended = torch.matmul(a, v).squeeze(1)
        return torch.cat([g, attended], dim=-1)

class EarlyConcatFusion(nn.Module):
    """Ablation: z = CONCAT(g, t) where t is BERT CLS."""

    def __init__(self, graph_dim: int, text_dim: int) -> None:
        super().__init__()
        self.out_dim = graph_dim + text_dim

    def forward(self, g: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.cat([g, t], dim=-1)

class GNNBertFusionModel(nn.Module):
    """
    End-to-end GNN-BERT fusion.
    Multi-label tag head + optional valence/arousal regression (DEAM L_aux).
    fusion: 'cross_attention' | 'early_concat' | 'gnn_only' | 'bert_only'
    """

    def __init__(
        self,
        in_channels: int,
        num_labels: int,
        bert_name: str = "bert-base-uncased",
        fusion: str = "cross_attention",
        gnn_type: str = "graphsage",
        gnn_hidden: int = 128,
        gnn_layers: int = 2,
        gnn_dropout: float = 0.2,
        freeze_bert: bool = False,
        d_model: int = 256,
        predict_emotion: bool = True,
    ) -> None:
        super().__init__()
        self.fusion = fusion
        self.num_labels = num_labels
        self.predict_emotion = predict_emotion

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

        self.cross = CrossAttentionFusion(gnn_hidden, text_dim, d_model=d_model)
        self.early = EarlyConcatFusion(gnn_hidden, text_dim)

        if fusion == "cross_attention":
            clf_in = self.cross.out_dim
        elif fusion == "early_concat":
            clf_in = self.early.out_dim
        elif fusion == "gnn_only":
            clf_in = gnn_hidden
        elif fusion == "bert_only":
            clf_in = text_dim
        else:
            raise ValueError(f"Unknown fusion mode: {fusion}")

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(clf_in, num_labels)
        self.valence_head = nn.Linear(clf_in, 1) if predict_emotion else None
        self.arousal_head = nn.Linear(clf_in, 1) if predict_emotion else None

    def encode_graph(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        return self.gnn.encode(x, edge_index, batch)

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state
        t = h[:, 0, :]
        return h, t

    def fuse(
        self,
        g: torch.Tensor,
        h_text: torch.Tensor,
        t: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.fusion == "cross_attention":
            return self.cross(g, h_text, attention_mask)
        if self.fusion == "early_concat":
            return self.early(g, t)
        if self.fusion == "gnn_only":
            return g
        return t

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_z: bool = False,
        return_emotion: bool = False,
    ) -> torch.Tensor | tuple:
        g = self.encode_graph(x, edge_index, batch)
        h_text, t = self.encode_text(input_ids, attention_mask)
        z = self.fuse(g, h_text, t, attention_mask)
        z = self.dropout(z)
        logits = self.classifier(z)
        if return_emotion and self.predict_emotion:
            v = self.valence_head(z).squeeze(-1)
            a = self.arousal_head(z).squeeze(-1)
            if return_z:
                return logits, v, a, z
            return logits, v, a
        if return_z:
            return logits, z
        return logits
