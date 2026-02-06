# MLX Transformer Language Model

A complete, production-ready implementation of a GPT-style decoder-only transformer language model using Apple's MLX framework. Optimized for training on Apple Silicon Macs.

## Features

- **GPT-style Architecture**: Decoder-only transformer with causal self-attention
- **Modern Improvements**: RMSNorm, SwiGLU activation, Rotary Position Embeddings (RoPE)
- **Efficient Training**: Gradient accumulation, mixed precision, cosine learning rate schedule
- **Memory Efficient**: Streaming data loading, memory-mapped datasets
- **Full Pipeline**: Tokenizer training, data preparation, training, and inference

## Architecture

The model implements a modern transformer architecture:

```
TransformerModel
├── Token Embeddings (vocab_size × embedding_dim)
├── Position Embeddings (context_length × embedding_dim)
├── Transformer Blocks × num_layers
│   ├── RMSNorm
│   ├── Multi-Head Attention (with RoPE)
│   │   ├── QKV Projection
│   │   ├── Rotary Position Embedding
│   │   ├── Scaled Dot-Product Attention (causal)
│   │   └── Output Projection
│   ├── Residual Connection
│   ├── RMSNorm
│   ├── Feed-Forward (SwiGLU)
│   └── Residual Connection
├── Final RMSNorm
└── Output Projection (weight-tied with embeddings)
```

**Default Configuration (350M parameters):**
- Embedding dimension: 768
- Layers: 16
- Attention heads: 12
- FFN dimension: 3072
- Context length: 2048
- Vocabulary: 32,000 tokens

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data

```bash
# Download and prepare TinyStories dataset
./scripts/prepare_data.sh

# Or with custom settings
DATASET=tinystories VOCAB_SIZE=32000 ./scripts/prepare_data.sh
```

**Manual data preparation:**

```bash
# Train tokenizer
python -m src.tokenizer train --input ./raw_text/*.txt --output data/tokenizer --vocab-size 32000

# Prepare dataset
python -m src.data prepare --input ./raw_text --output data/full.bin --tokenizer data/tokenizer.model

# Split into train/val
python -m src.data split --input data/full.bin --train-output data/train.bin --val-output data/val.bin
```

### 3. Train Model

```bash
# Train with default config (~350M parameters)
python -m src.train --config configs/model_config.yaml

# Or start with smaller model for testing
python -m src.train --config configs/model_config_small.yaml

# Resume from checkpoint
python -m src.train --config configs/model_config.yaml --resume checkpoints/step_10000
```

### 4. Generate Text

```bash
# Interactive chat
python -m src.inference --checkpoint checkpoints/best --tokenizer data/tokenizer.model --interactive

# Single prompt
python -m src.inference --checkpoint checkpoints/best --tokenizer data/tokenizer.model \
    --prompt "Once upon a time" --max-tokens 200 --temperature 0.8

# Batch generation
python -m src.inference --checkpoint checkpoints/best --tokenizer data/tokenizer.model \
    --prompts-file prompts.txt --output results.txt
```

## Configuration

Edit `configs/model_config.yaml` to customize:

```yaml
model:
  vocab_size: 32000
  embedding_dim: 768
  num_layers: 16
  num_heads: 12
  ff_dim: 3072
  context_length: 2048
  dropout: 0.1

training:
  batch_size: 4
  gradient_accumulation_steps: 8  # effective batch = 32
  learning_rate: 3.0e-4
  weight_decay: 0.1
  max_steps: 100000
  warmup_steps: 2000
```

## Memory Requirements

| Model Size | Parameters | Memory (FP16) | Recommended RAM |
|------------|------------|---------------|-----------------|
| Small      | ~25M       | ~50 MB        | 8 GB            |
| Medium     | ~125M      | ~250 MB       | 16 GB           |
| Default    | ~350M      | ~700 MB       | 16 GB           |
| Large      | ~760M      | ~1.5 GB       | 32 GB           |

Actual memory usage includes optimizer states (~2x), gradients (~1x), and activations (depends on batch size and sequence length).

## Training Tips

### For 16GB M4 Mac mini:

1. **Batch Size**: Start with `batch_size: 4` and `gradient_accumulation_steps: 8`
2. **Context Length**: 2048 is feasible; reduce to 1024 if memory constrained
3. **Mixed Precision**: Keep `mixed_precision: true` for memory efficiency
4. **Model Size**: 350M parameters works well; for larger models, reduce batch size

### Hyperparameter Tuning:

- **Learning Rate**: 3e-4 is good for most cases; try 1e-4 for fine-tuning
- **Warmup**: 2000 steps works well; use ~1-2% of total steps
- **Weight Decay**: 0.1 is standard; reduce for smaller datasets

### Monitoring Training:

- Check `logs/` for training metrics (CSV format)
- Sample generations are logged during training
- Validation loss should decrease; if not, check learning rate

## Project Structure

```
├── src/
│   ├── model.py       # Transformer architecture
│   ├── tokenizer.py   # SentencePiece tokenizer
│   ├── data.py        # Data loading and preprocessing
│   ├── train.py       # Training loop
│   └── inference.py   # Text generation
├── configs/
│   ├── model_config.yaml       # Default 350M config
│   └── model_config_small.yaml # Small model for testing
├── scripts/
│   └── prepare_data.sh  # Data download and preparation
├── data/                # Tokenized datasets
├── checkpoints/         # Model checkpoints
└── logs/                # Training logs
```

## Troubleshooting

**Out of Memory:**
- Reduce `batch_size`
- Reduce `context_length`
- Increase `gradient_accumulation_steps`

**Slow Training:**
- Ensure `mixed_precision: true`
- Check that MLX is using Metal backend
- Reduce logging frequency (`log_every`)

**Poor Generation Quality:**
- Train longer (at least until validation loss plateaus)
- Try different temperature (0.7-0.9 works well)
- Use repetition penalty (1.1-1.2)

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer
- [Language Models are Unsupervised Multitask Learners](https://openai.com/research/better-language-models) - GPT-2
- [RoFormer](https://arxiv.org/abs/2104.09864) - Rotary Position Embeddings
- [GLU Variants](https://arxiv.org/abs/2002.05202) - SwiGLU activation
- [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) - RMSNorm

## License

MIT License
