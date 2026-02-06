# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A GPT-style decoder-only transformer language model implementation using Apple's MLX framework, optimized for Apple Silicon (M-series Macs).

## Commands

### Setup
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### Data Preparation
```bash
# Full pipeline (download, tokenize, split)
./scripts/prepare_data.sh

# Manual steps
python -m src.tokenizer train --input raw_data/*.txt --output data/tokenizer --vocab-size 32000
python -m src.data prepare --input raw_data --output data/full.bin --tokenizer data/tokenizer.model
python -m src.data split --input data/full.bin --train-output data/train.bin --val-output data/val.bin
```

### Training
```bash
python -m src.train --config configs/model_config.yaml
python -m src.train --config configs/model_config.yaml --resume checkpoints/step_10000
```

### Inference
```bash
python -m src.inference --checkpoint checkpoints/best --tokenizer data/tokenizer.model --interactive
python -m src.inference --checkpoint checkpoints/best --tokenizer data/tokenizer.model --prompt "Hello" --max-tokens 100
```

## Architecture

```
src/
├── model.py      # Transformer architecture (TransformerModel, MultiHeadAttention, FeedForward)
├── tokenizer.py  # SentencePiece BPE tokenizer wrapper
├── data.py       # Memory-mapped dataset loading (TokenizedDataset, DataLoader)
├── train.py      # Training loop with gradient accumulation, checkpointing
└── inference.py  # Text generation with sampling strategies
```

### Model Components
- **TransformerModel**: Main model class combining all components
- **MultiHeadAttention**: Causal self-attention with Rotary Position Embeddings (RoPE)
- **FeedForward**: SwiGLU activation variant
- **RMSNorm**: Root mean square layer normalization
- **ModelConfig**: Dataclass holding all model hyperparameters

### Key Design Decisions
- Pre-norm architecture (LayerNorm before attention/FFN)
- Weight tying between token embeddings and output projection
- RoPE applied inside attention for better length generalization
- Memory-mapped binary files for efficient data loading
- Gradient accumulation for larger effective batch sizes

## MLX-Specific Notes

- MLX uses lazy evaluation - call `mx.eval()` to force computation
- Arrays are immutable; operations return new arrays
- Use `mx.save_safetensors()` for checkpoints
- Mixed precision via dtype conversion, not autocast context

## Configuration

Model and training configs in `configs/model_config.yaml`. Key parameters:
- `model.context_length`: Max sequence length (default 2048)
- `training.batch_size`: Per-device batch size
- `training.gradient_accumulation_steps`: Multiply with batch_size for effective batch
- `training.mixed_precision`: Use FP16 for memory efficiency
