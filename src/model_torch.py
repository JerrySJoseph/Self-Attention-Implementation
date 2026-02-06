"""
PyTorch Transformer Model

Mirror of the MLX model for cloud GPU training (NVIDIA).
Optimized for H100 with Flash Attention and torch.compile.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    """Model configuration."""
    vocab_size: int = 32000
    embedding_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    ff_dim: int = 2048
    context_length: int = 1024
    dropout: float = 0.1
    rope_theta: float = 10000.0

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    def estimate_parameters(self) -> int:
        """Estimate total parameters."""
        embed = self.vocab_size * self.embedding_dim
        per_layer = (
            4 * self.embedding_dim * self.embedding_dim +  # QKV + O
            3 * self.embedding_dim * self.ff_dim  # SwiGLU FFN
        )
        total = embed + self.num_layers * per_layer
        return total


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute in float32 for stability
        x_f32 = x.float()
        rms = torch.sqrt(torch.mean(x_f32 ** 2, dim=-1, keepdim=True) + self.eps)
        x_norm = x_f32 / rms
        return (x_norm * self.weight).to(x.dtype)


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""

    def __init__(self, dim: int, max_seq_len: int = 8192, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        # Precompute frequencies
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Precompute cos/sin cache
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        positions = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(positions, self.inv_freq)
        # Shape: (seq_len, dim/2) -> (seq_len, dim) via interleaving
        cos = torch.cos(freqs).repeat(1, 2)
        sin = torch.sin(freqs).repeat(1, 2)
        self.register_buffer("cos_cache", cos, persistent=False)
        self.register_buffer("sin_cache", sin, persistent=False)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """Apply rotary embedding.

        Args:
            x: (batch, heads, seq_len, head_dim)
            offset: Position offset for KV cache

        Returns:
            Rotated tensor
        """
        seq_len = x.shape[2]

        # Extend cache if needed
        if offset + seq_len > self.cos_cache.shape[0]:
            self._build_cache(offset + seq_len)

        cos = self.cos_cache[offset:offset + seq_len].to(x.dtype)
        sin = self.sin_cache[offset:offset + seq_len].to(x.dtype)

        # Rotate: split into pairs and apply rotation
        x1, x2 = x[..., ::2], x[..., 1::2]
        rotated = torch.stack([-x2, x1], dim=-1).flatten(-2)

        return x * cos + rotated * sin


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with RoPE."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.embedding_dim // config.num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(config.embedding_dim, config.embedding_dim, bias=False)
        self.k_proj = nn.Linear(config.embedding_dim, config.embedding_dim, bias=False)
        self.v_proj = nn.Linear(config.embedding_dim, config.embedding_dim, bias=False)
        self.out_proj = nn.Linear(config.embedding_dim, config.embedding_dim, bias=False)

        self.rope = RotaryEmbedding(self.head_dim, config.context_length, config.rope_theta)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to (batch, heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        offset = cache[0].shape[2] if cache is not None else 0
        q = self.rope(q, offset)
        k = self.rope(k, offset)

        # Handle KV cache
        if cache is not None:
            k = torch.cat([cache[0], k], dim=2)
            v = torch.cat([cache[1], v], dim=2)

        new_cache = (k, v) if use_cache else None

        # Attention with Flash Attention when available
        if hasattr(F, 'scaled_dot_product_attention'):
            # Use PyTorch's Flash Attention implementation
            is_causal = cache is None  # Only causal for prefill, not for cached generation
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=is_causal,
            )
        else:
            # Fallback to manual attention
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

            # Causal mask
            if cache is None:
                mask = torch.triu(
                    torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device),
                    diagonal=1
                )
                scores = scores.masked_fill(mask, float('-inf'))

            attn = F.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            out = torch.matmul(attn, v)

        # Reshape and project output
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        out = self.out_proj(out)

        return out, new_cache


class FeedForward(nn.Module):
    """SwiGLU Feed-Forward Network."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.w1 = nn.Linear(config.embedding_dim, config.ff_dim, bias=False)
        self.w2 = nn.Linear(config.ff_dim, config.embedding_dim, bias=False)
        self.w3 = nn.Linear(config.embedding_dim, config.ff_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: (silu(xW1) * xW3) W2
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.ffn = FeedForward(config)
        self.norm1 = RMSNorm(config.embedding_dim)
        self.norm2 = RMSNorm(config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Pre-norm attention
        h, new_cache = self.attention(self.norm1(x), cache, use_cache)
        x = x + self.dropout(h)

        # Pre-norm FFN
        x = x + self.dropout(self.ffn(self.norm2(x)))

        return x, new_cache


class TransformerModel(nn.Module):
    """GPT-style decoder-only transformer."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Token embeddings (no positional - using RoPE)
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])

        # Output
        self.norm = RMSNorm(config.embedding_dim)

        # Weight tying: output projection shares weights with token embedding
        self.output_proj = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        self.output_proj.weight = self.token_embedding.weight

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with scaled normal distribution."""
        std = 0.02
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)

        # Scale output projections for residual connections
        scale = (2 * self.config.num_layers) ** -0.5
        for layer in self.layers:
            layer.attention.out_proj.weight.data *= scale
            layer.ffn.w2.weight.data *= scale

    def forward(
        self,
        input_ids: torch.Tensor,
        cache: Optional[list] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[list]]:
        """Forward pass.

        Args:
            input_ids: (batch, seq_len) token IDs
            cache: Optional list of (k, v) tuples per layer
            use_cache: Whether to return updated cache

        Returns:
            logits: (batch, seq_len, vocab_size)
            cache: Updated cache if use_cache=True
        """
        x = self.dropout(self.token_embedding(input_ids))

        new_cache = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            x, layer_new_cache = layer(x, layer_cache, use_cache)
            if use_cache:
                new_cache.append(layer_new_cache)

        x = self.norm(x)
        logits = self.output_proj(x)

        return logits, new_cache

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(config: ModelConfig, device: str = "cuda", dtype: torch.dtype = torch.bfloat16) -> TransformerModel:
    """Create and initialize model.

    Args:
        config: Model configuration
        device: Device to place model on
        dtype: Model dtype (bfloat16 recommended for H100)

    Returns:
        Initialized model
    """
    model = TransformerModel(config)
    model = model.to(device=device, dtype=dtype)
    return model
