"""
Transformer Model Architecture

A GPT-style decoder-only transformer implementation using MLX.
Implements multi-head causal self-attention with pre-norm architecture.

Optimizations:
- RoPE only (no learned positional embeddings)
- Optimized RoPE computation with interleaved rotation
- Cached causal masks
- FP16-safe RMSNorm epsilon
- Dropout support
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import math

import mlx.core as mx
import mlx.nn as nn


@dataclass
class ModelConfig:
    """Configuration for the Transformer model."""
    vocab_size: int = 32000
    embedding_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    ff_dim: int = 2048
    context_length: int = 1024
    dropout: float = 0.0
    rope_theta: float = 10000.0  # RoPE base frequency

    def __post_init__(self):
        """Validate configuration parameters."""
        assert self.embedding_dim % self.num_heads == 0, \
            f"embedding_dim ({self.embedding_dim}) must be divisible by num_heads ({self.num_heads})"
        self.head_dim = self.embedding_dim // self.num_heads

    @classmethod
    def from_dict(cls, config_dict: dict) -> "ModelConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "context_length": self.context_length,
            "dropout": self.dropout,
            "rope_theta": self.rope_theta,
        }

    def estimate_parameters(self) -> int:
        """Estimate total number of parameters in the model."""
        # Token embeddings (no position embeddings - using RoPE)
        params = self.vocab_size * self.embedding_dim

        # Per transformer block:
        # - Attention: Q, K, V projections + output projection
        attn_params = 4 * self.embedding_dim * self.embedding_dim
        # - FFN: SwiGLU has 3 projections (w1, w2, w3)
        ffn_params = 3 * self.embedding_dim * self.ff_dim
        # - Layer norms (2 per block, only weight param for RMSNorm)
        ln_params = 2 * self.embedding_dim

        params += self.num_layers * (attn_params + ffn_params + ln_params)

        # Final layer norm
        params += self.embedding_dim
        # Output projection (weight tied with embeddings, so not counted)

        return params


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    More efficient than LayerNorm as it doesn't compute mean.
    Used in modern LLMs like LLaMA.
    """

    def __init__(self, dims: int, eps: float = 1e-5):
        """Initialize RMSNorm.

        Args:
            dims: Hidden dimension size
            eps: Epsilon for numerical stability (1e-5 is safer for FP16)
        """
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        # Compute in float32 for stability, then cast back
        dtype = x.dtype
        x = x.astype(mx.float32)
        rms = mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)
        x = (x / rms).astype(dtype)
        return self.weight * x


class RotaryPositionalEmbedding:
    """Rotary Position Embedding (RoPE).

    Encodes position information directly into attention scores.
    Uses optimized interleaved rotation for better performance.
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute rotation matrices
        inv_freq = 1.0 / (base ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
        t = mx.arange(max_seq_len, dtype=mx.float32)
        freqs = mx.outer(t, inv_freq)

        # Cache cos and sin values - shape: (max_seq_len, dim//2)
        self._cos_cached = mx.cos(freqs)
        self._sin_cached = mx.sin(freqs)
        mx.eval(self._cos_cached, self._sin_cached)

    def __call__(self, x: mx.array, offset: int = 0) -> mx.array:
        """Apply rotary embeddings to input tensor.

        Args:
            x: Input tensor of shape (batch, seq_len, num_heads, head_dim)
            offset: Position offset for cached inference

        Returns:
            Tensor with rotary embeddings applied
        """
        seq_len = x.shape[1]

        # Get cached values for this sequence
        cos = self._cos_cached[offset:offset + seq_len]
        sin = self._sin_cached[offset:offset + seq_len]

        # Reshape for broadcasting: (seq_len, dim//2) -> (1, seq_len, 1, dim//2)
        cos = cos.reshape(1, seq_len, 1, -1)
        sin = sin.reshape(1, seq_len, 1, -1)

        # Split into first and second half of head_dim
        x1 = x[..., : self.dim // 2]
        x2 = x[..., self.dim // 2 :]

        # Apply rotation using complex multiplication formula
        # (x1 + i*x2) * (cos + i*sin) = (x1*cos - x2*sin) + i*(x1*sin + x2*cos)
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos

        return mx.concatenate([rotated_x1, rotated_x2], axis=-1)


class MultiHeadAttention(nn.Module):
    """Multi-head causal self-attention with RoPE.

    Implements scaled dot-product attention with causal masking
    to prevent attending to future tokens.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Combined QKV projection for efficiency
        self.qkv_proj = nn.Linear(config.embedding_dim, 3 * config.embedding_dim, bias=False)
        self.out_proj = nn.Linear(config.embedding_dim, config.embedding_dim, bias=False)

        # RoPE
        self.rope = RotaryPositionalEmbedding(
            config.head_dim,
            config.context_length,
            config.rope_theta
        )

        # Dropout
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else None

        # Cache for causal mask
        self._causal_mask_cache: Optional[mx.array] = None
        self._causal_mask_size: int = 0

    def _get_causal_mask(self, q_len: int, kv_len: int) -> mx.array:
        """Get or create causal mask.

        Args:
            q_len: Query sequence length
            kv_len: Key/Value sequence length

        Returns:
            Causal mask of shape (q_len, kv_len)
        """
        # For single token generation, create minimal mask
        if q_len == 1:
            return mx.zeros((1, kv_len))

        # For full sequence, use cached mask if possible
        if self._causal_mask_cache is None or kv_len > self._causal_mask_size:
            # Create new mask (larger than needed for future reuse)
            size = max(kv_len, self.config.context_length)
            mask = mx.triu(
                mx.full((size, size), float("-inf")),
                k=1
            )
            self._causal_mask_cache = mask
            self._causal_mask_size = size
            mx.eval(self._causal_mask_cache)

        # Extract the relevant portion
        return self._causal_mask_cache[:q_len, :kv_len]

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Optional[Tuple[mx.array, mx.array]]]:
        """Forward pass for attention.

        Args:
            x: Input tensor (batch, seq_len, embedding_dim)
            mask: Optional attention mask
            cache: Optional KV cache for inference (k_cache, v_cache)

        Returns:
            Output tensor and updated cache
        """
        batch_size, seq_len, _ = x.shape

        # Compute Q, K, V
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        # Apply rotary embeddings
        offset = 0 if cache is None else cache[0].shape[1]
        q = self.rope(q, offset=offset)
        k = self.rope(k, offset=offset)

        # Update KV cache for inference
        if cache is not None:
            k_cache, v_cache = cache
            k = mx.concatenate([k_cache, k], axis=1)
            v = mx.concatenate([v_cache, v], axis=1)

        new_cache = (k, v)

        # Transpose for attention: (batch, num_heads, seq_len, head_dim)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # Compute attention scores
        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale

        # Apply causal mask
        kv_len = k.shape[2]
        q_len = q.shape[2]

        if mask is None:
            causal_mask = self._get_causal_mask(q_len, kv_len)
            scores = scores + causal_mask
        else:
            scores = scores + mask

        # Softmax and apply to values
        attn_weights = mx.softmax(scores, axis=-1)

        # Apply dropout to attention weights
        if self.dropout is not None:
            attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        output = attn_weights @ v

        # Transpose back and combine heads
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)

        # Output projection
        output = self.out_proj(output)

        return output, new_cache


class FeedForward(nn.Module):
    """Feed-forward network with SwiGLU activation.

    SwiGLU: FFN(x) = (Swish(xW1) * xW3) W2
    Better performance than standard GELU FFN.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        # SwiGLU has 3 projections
        self.w1 = nn.Linear(config.embedding_dim, config.ff_dim, bias=False)
        self.w2 = nn.Linear(config.ff_dim, config.embedding_dim, bias=False)
        self.w3 = nn.Linear(config.embedding_dim, config.ff_dim, bias=False)

        # Dropout
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else None

    def __call__(self, x: mx.array) -> mx.array:
        # SwiGLU: Swish(xW1) * xW3, then project down
        out = self.w2(nn.silu(self.w1(x)) * self.w3(x))
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm architecture.

    Structure:
        x -> RMSNorm -> Attention -> Dropout -> Residual -> RMSNorm -> FFN -> Dropout -> Residual
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention_norm = RMSNorm(config.embedding_dim)
        self.attention = MultiHeadAttention(config)
        self.ffn_norm = RMSNorm(config.embedding_dim)
        self.ffn = FeedForward(config)

        # Residual dropout
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else None

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Optional[Tuple[mx.array, mx.array]]]:
        """Forward pass through transformer block.

        Args:
            x: Input tensor
            mask: Optional attention mask
            cache: Optional KV cache

        Returns:
            Output tensor and updated cache
        """
        # Pre-norm attention with residual
        h = self.attention_norm(x)
        attn_out, new_cache = self.attention(h, mask=mask, cache=cache)
        if self.dropout is not None:
            attn_out = self.dropout(attn_out)
        x = x + attn_out

        # Pre-norm FFN with residual
        h = self.ffn_norm(x)
        ffn_out = self.ffn(h)
        if self.dropout is not None:
            ffn_out = self.dropout(ffn_out)
        x = x + ffn_out

        return x, new_cache


class TransformerModel(nn.Module):
    """GPT-style decoder-only transformer.

    Architecture:
        - Token embeddings (weight tied with output projection)
        - RoPE (applied inside attention, no learned positional embeddings)
        - N transformer blocks
        - Final layer norm
        - Output projection (tied with embeddings)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Token embeddings only (RoPE handles positions)
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)

        # Embedding dropout
        self.embed_dropout = nn.Dropout(config.dropout) if config.dropout > 0 else None

        # Transformer blocks
        self.layers = [TransformerBlock(config) for _ in range(config.num_layers)]

        # Final layer norm
        self.norm = RMSNorm(config.embedding_dim)

        # Output projection (weight tied with token embeddings)
        self.output_proj = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)

        # Tie weights
        self.output_proj.weight = self.token_embedding.weight

    def __call__(
        self,
        input_ids: mx.array,
        cache: Optional[list] = None,
    ) -> Tuple[mx.array, Optional[list]]:
        """Forward pass through the model.

        Args:
            input_ids: Token IDs (batch, seq_len)
            cache: Optional list of KV caches for each layer

        Returns:
            Logits (batch, seq_len, vocab_size) and updated caches
        """
        # Get embeddings (no positional embeddings - RoPE is in attention)
        x = self.token_embedding(input_ids)

        if self.embed_dropout is not None:
            x = self.embed_dropout(x)

        # Process through transformer blocks
        new_cache = []
        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            x, updated_cache = layer(x, cache=layer_cache)
            new_cache.append(updated_cache)

        # Final normalization
        x = self.norm(x)

        # Project to vocabulary
        logits = self.output_proj(x)

        return logits, new_cache

    def generate_step(
        self,
        input_ids: mx.array,
        cache: Optional[list] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
    ) -> Tuple[mx.array, list]:
        """Generate next token given input.

        Args:
            input_ids: Current token IDs
            cache: KV cache from previous steps
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            top_k: Top-k sampling (0 = disabled)

        Returns:
            Next token ID and updated cache
        """
        logits, new_cache = self(input_ids, cache=cache)

        # Get logits for last position
        logits = logits[:, -1, :]

        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature

        # Apply top-k filtering
        if top_k > 0:
            top_k = min(top_k, logits.shape[-1])
            top_k_logits, _ = mx.topk(logits, top_k)
            threshold = top_k_logits[:, -1:]
            logits = mx.where(logits < threshold, float("-inf"), logits)

        # Apply top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_indices = mx.argsort(logits, axis=-1)[:, ::-1]
            sorted_logits = mx.take_along_axis(logits, sorted_indices, axis=-1)
            cumulative_probs = mx.cumsum(mx.softmax(sorted_logits, axis=-1), axis=-1)

            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift to keep first token above threshold
            sorted_indices_to_remove = mx.concatenate([
                mx.zeros((sorted_indices_to_remove.shape[0], 1), dtype=mx.bool_),
                sorted_indices_to_remove[:, :-1]
            ], axis=-1)

            # Scatter back to original indices
            indices_to_remove = mx.zeros_like(logits, dtype=mx.bool_)
            indices_to_remove = mx.put_along_axis(
                indices_to_remove,
                sorted_indices,
                sorted_indices_to_remove,
                axis=-1
            )
            logits = mx.where(indices_to_remove, float("-inf"), logits)

        # Sample from distribution
        probs = mx.softmax(logits, axis=-1)
        next_token = mx.random.categorical(probs)

        return next_token, new_cache

    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        def count_params(params):
            total = 0
            if isinstance(params, dict):
                for v in params.values():
                    total += count_params(v)
            elif isinstance(params, list):
                for v in params:
                    total += count_params(v)
            elif hasattr(params, 'size'):
                total += params.size
            return total
        return count_params(self.parameters())


def create_model(config: ModelConfig, dtype: mx.Dtype = mx.float32) -> TransformerModel:
    """Create and initialize a transformer model.

    Args:
        config: Model configuration
        dtype: Model dtype (mx.float32 or mx.float16)

    Returns:
        Initialized TransformerModel
    """
    model = TransformerModel(config)

    # Scale weights of residual path outputs for better training stability
    # This follows GPT-2 style initialization
    num_layers = config.num_layers
    scale = 1.0 / math.sqrt(2.0 * num_layers)

    for layer in model.layers:
        # Scale attention output projection
        layer.attention.out_proj.weight = layer.attention.out_proj.weight * scale
        # Scale FFN output projection
        layer.ffn.w2.weight = layer.ffn.w2.weight * scale

    mx.eval(model.parameters())

    # Convert to target dtype if needed
    if dtype != mx.float32:
        def convert_dtype(params):
            if isinstance(params, dict):
                return {k: convert_dtype(v) for k, v in params.items()}
            elif isinstance(params, list):
                return [convert_dtype(v) for v in params]
            elif isinstance(params, mx.array):
                return params.astype(dtype)
            return params

        new_params = convert_dtype(model.parameters())
        model.update(new_params)
        mx.eval(model.parameters())

    return model


def load_model(checkpoint_path: str, config: Optional[ModelConfig] = None) -> Tuple[TransformerModel, ModelConfig]:
    """Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        config: Optional config (loaded from checkpoint if not provided)

    Returns:
        Model and config
    """
    import json
    import os

    # Load weights
    weights = mx.load(checkpoint_path)

    # Load config
    config_path = os.path.join(os.path.dirname(checkpoint_path), "config.json")
    if config is None and os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = ModelConfig.from_dict(json.load(f))
    elif config is None:
        raise ValueError("Config not provided and config.json not found")

    # Create model and load weights
    model = TransformerModel(config)
    model.load_weights(list(weights.items()))

    return model, config


def save_model(model: TransformerModel, config: ModelConfig, save_path: str):
    """Save model checkpoint.

    Args:
        model: Model to save
        config: Model configuration
        save_path: Path to save checkpoint
    """
    import json
    import os

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save weights
    mx.save(save_path, dict(model.parameters()))

    # Save config
    config_path = os.path.join(os.path.dirname(save_path), "config.json")
    with open(config_path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)


if __name__ == "__main__":
    # Test model creation
    config = ModelConfig()
    print(f"Creating model with {config.estimate_parameters():,} estimated parameters")

    model = create_model(config)
    print(f"Actual parameters: {model.count_parameters():,}")

    # Test forward pass
    batch_size = 2
    seq_len = 128
    input_ids = mx.random.randint(0, config.vocab_size, (batch_size, seq_len))

    logits, cache = model(input_ids)
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {logits.shape}")

    # Test generation step
    next_token, _ = model.generate_step(input_ids[:, :1], temperature=0.8, top_p=0.95)
    print(f"Generated token: {next_token}")

    # Test FP16 model
    print("\nTesting FP16 model...")
    model_fp16 = create_model(config, dtype=mx.float16)
    logits_fp16, _ = model_fp16(input_ids)
    print(f"FP16 output shape: {logits_fp16.shape}, dtype: {logits_fp16.dtype}")
