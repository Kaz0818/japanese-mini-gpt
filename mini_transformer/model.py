from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class MiniTransformerConfig:
    vocab_size: int
    block_size: int = 64
    embedding_dim: int = 128
    num_layers: int = 2
    num_heads: int = 4
    feed_forward_dim: int = 512
    dropout: float = 0.1
    tie_embeddings: bool = False


class CausalSelfAttention(nn.Module):
    def __init__(self, config: MiniTransformerConfig) -> None:
        super().__init__()
        if config.embedding_dim % config.num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")

        self.num_heads = config.num_heads
        self.head_dim = config.embedding_dim // config.num_heads
        self.query_key_value = nn.Linear(config.embedding_dim, 3 * config.embedding_dim)
        self.output_projection = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.attention_dropout = nn.Dropout(config.dropout)
        self.residual_dropout = nn.Dropout(config.dropout)

        # Shape: [1, 1, block, block]. True means the key position is visible.
        causal_mask = torch.tril(
            torch.ones(config.block_size, config.block_size, dtype=torch.bool)
        )
        self.register_buffer(
            "causal_mask",
            causal_mask.view(1, 1, config.block_size, config.block_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, embedding_dim = x.shape

        qkv = self.query_key_value(x)
        query, key, value = qkv.split(embedding_dim, dim=2)

        # Shape: [batch, heads, block, head_dim].
        query = query.view(
            batch_size, sequence_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key = key.view(
            batch_size, sequence_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        value = value.view(
            batch_size, sequence_length, self.num_heads, self.head_dim
        ).transpose(1, 2)

        attention_scores = query @ key.transpose(-2, -1)
        attention_scores = attention_scores / (self.head_dim**0.5)
        visible_positions = self.causal_mask[:, :, :sequence_length, :sequence_length]
        attention_scores = attention_scores.masked_fill(~visible_positions, float("-inf"))

        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)
        attended_values = attention_weights @ value

        # Shape returns from [batch, heads, block, head_dim] to [batch, block, embed].
        attended_values = attended_values.transpose(1, 2).contiguous()
        attended_values = attended_values.view(
            batch_size, sequence_length, embedding_dim
        )
        output = self.output_projection(attended_values)
        return self.residual_dropout(output)


class FeedForward(nn.Module):
    def __init__(self, config: MiniTransformerConfig) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config.embedding_dim, config.feed_forward_dim),
            nn.GELU(),
            nn.Linear(config.feed_forward_dim, config.embedding_dim),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, config: MiniTransformerConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.embedding_dim)
        self.attention = CausalSelfAttention(config)
        self.feed_forward_norm = nn.LayerNorm(config.embedding_dim)
        self.feed_forward = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x))
        x = x + self.feed_forward(self.feed_forward_norm(x))
        return x


class MiniTransformerDecoder(nn.Module):
    def __init__(self, config: MiniTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.position_embedding = nn.Embedding(
            config.block_size,
            config.embedding_dim,
        )
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_layers)]
        )
        self.final_norm = nn.LayerNorm(config.embedding_dim)
        self.language_modeling_head = nn.Linear(
            config.embedding_dim,
            config.vocab_size,
            bias=False,
        )
        if config.tie_embeddings:
            self.language_modeling_head.weight = self.token_embedding.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length = input_ids.shape
        if sequence_length > self.config.block_size:
            raise ValueError(
                f"sequence length {sequence_length} exceeds block_size "
                f"{self.config.block_size}"
            )

        positions = torch.arange(
            sequence_length,
            device=input_ids.device,
            dtype=torch.long,
        )

        # token_embeddings: [batch, block, embed]
        # position_embeddings: [block, embed], broadcast across the batch.
        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(positions)
        x = self.dropout(token_embeddings + position_embeddings)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        logits = self.language_modeling_head(x)
        return logits
