"""
GNN encoder on music structure graphs (Task 2).

  GraphSAGE update:
    h_i^(l+1) = sigma( W^(l) · CONCAT( h_i^(l), MEAN_{jinN(i)} h_j^(l) ) )
  Graph readout (mean pooling):
    g = (1/|V|) Σ_i h_i^(L),  ŷ = sigma(W g + b)
  Alternative: GAT. Implemented with PyTorch Geometric.

Also includes Baseline B2: CNN on mel-spectrogram (no graph, no text).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv, global_mean_pool

class MusicGraphSAGE(nn.Module):
    """GraphSAGE encoder + mean-pool readout for genre prediction."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        num_layers: int = 2,
        num_classes: int = 8,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        h = x
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return global_mean_pool(h, batch)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        g = self.encode(x, edge_index, batch)
        return self.classifier(g)

class MusicGAT(nn.Module):
    """Optional GAT encoder."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        num_layers: int = 2,
        num_classes: int = 8,
        heads: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels // heads, heads=heads, dropout=dropout))
        for _ in range(num_layers - 1):
            self.convs.append(
                GATConv(hidden_channels, hidden_channels // heads, heads=heads, dropout=dropout)
            )
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        h = x
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return global_mean_pool(h, batch)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        g = self.encode(x, edge_index, batch)
        return self.classifier(g)

class CNNMelBaseline(nn.Module):
    """
    Baseline B2 (Section 8): CNN on mel-spectrogram (no graph, no text).
    Input: (B, 1, n_mels, T) with adaptive pooling for variable T.
    """

    def __init__(self, num_classes: int = 8, n_mels: int = 128) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
        self.n_mels = n_mels

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # mel: (B, n_mels, T) or (B, 1, n_mels, T)
        if mel.dim() == 3:
            mel = mel.unsqueeze(1)
        return self.classifier(self.features(mel))
