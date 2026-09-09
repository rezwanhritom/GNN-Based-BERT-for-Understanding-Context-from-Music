"""
BERT text encoder for music tags / captions / lyrics.

  t = BERT_CLS(X_text)
  ŷ_k = sigma(w_k^T t + b_k)
  L_BERT = binary cross-entropy per tag
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class BertMusicTagClassifier(nn.Module):
    """Task 1: BERT multi-label tag classifier."""

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_labels: int = 50,
        freeze_bert: bool = False,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels
        self.freeze_bert = freeze_bert
        self.bert = AutoModel.from_pretrained(model_name)
        hidden = int(self.bert.config.hidden_size)
        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        t = BERT_CLS(X_text)
        ŷ = sigma(W t + b)  - returns logits; apply sigmoid outside for BCE-with-logits.
        """
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # CLS token
        t = out.last_hidden_state[:, 0, :]
        t = self.dropout(t)
        return self.classifier(t)

def build_tokenizer(model_name: str = "bert-base-uncased") -> Any:
    return AutoTokenizer.from_pretrained(model_name)

def tokenize_batch(
    texts: list[str],
    tokenizer: Any,
    max_length: int = 128,
) -> dict[str, torch.Tensor]:
    """Tokenize tags/captions/lyrics."""
    encoded = tokenizer(
        texts,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
    }
