"""
PyTorch Training Script for Cloud GPUs

Optimized for NVIDIA H100/A100:
- BFloat16 mixed precision (better than FP16 on H100)
- torch.compile for faster training
- Flash Attention 2 (via PyTorch 2.0+)
- Gradient checkpointing option for memory
- Large batch sizes for 80GB VRAM
"""

import argparse
import json
import logging
import math
import os
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import yaml
from tqdm import tqdm

from model_torch import ModelConfig, TransformerModel, create_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Optimization - larger batches for H100
    batch_size: int = 32  # Much larger for 80GB VRAM
    gradient_accumulation_steps: int = 4  # Effective batch = 128
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0

    # Schedule
    max_steps: int = 100000
    warmup_steps: int = 1000
    min_lr_ratio: float = 0.1

    # Checkpointing
    save_every: int = 5000
    eval_every: int = 1000
    eval_batches: int = 50
    log_every: int = 10
    sample_every: int = 5000

    # Data
    train_data_path: str = "data/train.bin"
    val_data_path: str = "data/val.bin"
    tokenizer_path: str = "data/tokenizer.model"

    # Directories
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

    # Hardware
    mixed_precision: bool = True  # Use bfloat16
    compile_model: bool = True  # Use torch.compile
    seed: int = 42

    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001

    @classmethod
    def from_dict(cls, d: dict) -> "TrainingConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


class TokenizedDataset(Dataset):
    """Memory-mapped tokenized dataset."""

    def __init__(self, data_path: str, context_length: int):
        self.context_length = context_length
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        self.num_tokens = len(self.data)
        self.num_samples = (self.num_tokens - 1) // context_length

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.context_length
        end = start + self.context_length + 1

        chunk = self.data[start:end].astype(np.int64)
        input_ids = torch.from_numpy(chunk[:-1])
        target_ids = torch.from_numpy(chunk[1:])

        return input_ids, target_ids


def create_dataloaders(
    train_path: str,
    val_path: str,
    context_length: int,
    batch_size: int,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders."""
    train_dataset = TokenizedDataset(train_path, context_length)
    val_dataset = TokenizedDataset(val_path, context_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    return train_loader, val_loader


class CosineSchedule:
    """Cosine learning rate schedule with warmup."""

    def __init__(
        self,
        learning_rate: float,
        warmup_steps: int,
        max_steps: int,
        min_lr_ratio: float = 0.1,
    ):
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = learning_rate * min_lr_ratio

    def __call__(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.learning_rate * step / max(self.warmup_steps, 1)
        elif step >= self.max_steps:
            return self.min_lr
        else:
            progress = (step - self.warmup_steps) / max(self.max_steps - self.warmup_steps, 1)
            return self.min_lr + 0.5 * (self.learning_rate - self.min_lr) * (1 + math.cos(math.pi * progress))


class MetricsLogger:
    """Log training metrics to CSV."""

    def __init__(self, log_dir: str, run_name: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / f"{run_name}.csv"

        with open(self.log_file, "w") as f:
            f.write("step,timestamp,train_loss,val_loss,learning_rate,tokens_per_sec,grad_norm\n")

    def log(self, step: int, **kwargs):
        values = [
            str(step),
            datetime.now().isoformat(),
            f"{kwargs.get('train_loss', ''):.6f}" if 'train_loss' in kwargs else "",
            f"{kwargs.get('val_loss', ''):.6f}" if 'val_loss' in kwargs else "",
            f"{kwargs.get('learning_rate', ''):.2e}" if 'learning_rate' in kwargs else "",
            f"{kwargs.get('tokens_per_sec', ''):.1f}" if 'tokens_per_sec' in kwargs else "",
            f"{kwargs.get('grad_norm', ''):.4f}" if 'grad_norm' in kwargs else "",
        ]
        with open(self.log_file, "a") as f:
            f.write(",".join(values) + "\n")


def compute_loss(model: TransformerModel, input_ids: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    """Compute cross-entropy loss."""
    logits, _ = model(input_ids)
    logits = logits.view(-1, logits.size(-1))
    targets = target_ids.view(-1)
    return F.cross_entropy(logits, targets)


@torch.no_grad()
def evaluate(model: TransformerModel, val_loader: DataLoader, max_batches: int, device: str) -> float:
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for i, (input_ids, target_ids) in enumerate(val_loader):
        if i >= max_batches:
            break

        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        loss = compute_loss(model, input_ids, target_ids)
        total_loss += loss.item()
        num_batches += 1

    model.train()
    return total_loss / max(num_batches, 1)


def save_checkpoint(
    model: TransformerModel,
    optimizer: torch.optim.Optimizer,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    step: int,
    val_loss: float,
    checkpoint_dir: str,
    is_best: bool = False,
):
    """Save training checkpoint."""
    import shutil

    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    step_dir = checkpoint_path / f"step_{step}"
    step_dir.mkdir(exist_ok=True)

    # Save model weights in a format compatible with MLX conversion
    weights_path = step_dir / "model.pt"
    torch.save(model.state_dict(), weights_path)

    # Save optimizer state
    optimizer_path = step_dir / "optimizer.pt"
    torch.save(optimizer.state_dict(), optimizer_path)

    # Save config
    config_path = step_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump({
            "model": model_config.to_dict(),
            "training": training_config.to_dict(),
            "step": step,
            "val_loss": val_loss,
        }, f, indent=2)

    logger.info(f"Saved checkpoint to {step_dir}")

    if is_best:
        best_dir = checkpoint_path / "best"
        if best_dir.exists():
            shutil.rmtree(best_dir)
        shutil.copytree(step_dir, best_dir)
        logger.info(f"Saved as best model (val_loss={val_loss:.4f})")

    # Clean old checkpoints
    all_checkpoints = sorted(
        [d for d in checkpoint_path.iterdir() if d.is_dir() and d.name.startswith("step_")],
        key=lambda x: int(x.name.split("_")[1])
    )
    if len(all_checkpoints) > 3:
        for old_ckpt in all_checkpoints[:-3]:
            shutil.rmtree(old_ckpt)


def load_checkpoint(
    checkpoint_path: str,
    model: TransformerModel,
    optimizer: torch.optim.Optimizer,
) -> Tuple[int, float]:
    """Load training checkpoint."""
    checkpoint_path = Path(checkpoint_path)

    with open(checkpoint_path / "config.json", "r") as f:
        config = json.load(f)

    model.load_state_dict(torch.load(checkpoint_path / "model.pt"))
    optimizer.load_state_dict(torch.load(checkpoint_path / "optimizer.pt"))

    return config["step"], config["val_loss"]


# Global for signal handling
_interrupted = False

def _signal_handler(signum, frame):
    global _interrupted
    logger.warning(f"\nReceived signal {signum} - will save checkpoint and exit...")
    _interrupted = True


def train(
    model_config: ModelConfig,
    training_config: TrainingConfig,
    resume_from: Optional[str] = None,
):
    """Main training loop."""
    global _interrupted

    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        logger.warning("CUDA not available - training on CPU will be slow!")

    # Print GPU info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    # Set seeds
    torch.manual_seed(training_config.seed)
    np.random.seed(training_config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(training_config.seed)

    # Determine dtype
    dtype = torch.bfloat16 if training_config.mixed_precision else torch.float32
    dtype_name = "bfloat16" if training_config.mixed_precision else "float32"
    logger.info(f"Training with {dtype_name} precision")

    # Create model
    logger.info(f"Creating model with ~{model_config.estimate_parameters():,} parameters")
    model = create_model(model_config, device=device, dtype=dtype)
    actual_params = model.count_parameters()
    logger.info(f"Actual parameters: {actual_params:,}")

    # Compile model for faster training (PyTorch 2.0+)
    if training_config.compile_model and hasattr(torch, 'compile'):
        logger.info("Compiling model with torch.compile...")
        model = torch.compile(model)
        logger.info("Model compiled!")

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
        betas=(0.9, 0.95),
        fused=True if torch.cuda.is_available() else False,  # Fused AdamW is faster
    )

    schedule = CosineSchedule(
        training_config.learning_rate,
        training_config.warmup_steps,
        training_config.max_steps,
        training_config.min_lr_ratio,
    )

    # Create data loaders
    logger.info("Loading data...")
    train_loader, val_loader = create_dataloaders(
        training_config.train_data_path,
        training_config.val_data_path,
        model_config.context_length,
        training_config.batch_size,
        num_workers=4,
    )
    logger.info(f"Train samples: {len(train_loader.dataset):,}")
    logger.info(f"Val samples: {len(val_loader.dataset):,}")

    # Metrics logger
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_logger = MetricsLogger(training_config.log_dir, run_name)

    # Resume from checkpoint
    start_step = 0
    best_val_loss = float("inf")
    patience_counter = 0

    if resume_from:
        start_step, best_val_loss = load_checkpoint(resume_from, model, optimizer)
        start_step += 1
        logger.info(f"Resumed from step {start_step - 1}")

    # Setup signal handler
    original_sigint = signal.signal(signal.SIGINT, _signal_handler)
    original_sigterm = signal.signal(signal.SIGTERM, _signal_handler)

    # Training loop
    logger.info(f"Starting training from step {start_step}")
    effective_batch_size = training_config.batch_size * training_config.gradient_accumulation_steps
    tokens_per_step = effective_batch_size * model_config.context_length
    logger.info(f"Effective batch size: {effective_batch_size}")
    logger.info(f"Tokens per step: {tokens_per_step:,}")

    # Enable gradient scaler for mixed precision
    scaler = torch.amp.GradScaler('cuda', enabled=training_config.mixed_precision)

    model.train()
    train_iter = iter(train_loader)
    accumulated_loss = 0.0
    step_start_time = time.time()
    tokens_processed = 0
    current_step = start_step

    pbar = tqdm(range(start_step, training_config.max_steps), desc="Training")

    try:
        for step in pbar:
            if _interrupted:
                raise KeyboardInterrupt()

            optimizer.zero_grad()

            # Gradient accumulation
            for micro_step in range(training_config.gradient_accumulation_steps):
                try:
                    input_ids, target_ids = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    input_ids, target_ids = next(train_iter)

                input_ids = input_ids.to(device)
                target_ids = target_ids.to(device)

                # Mixed precision forward/backward
                with torch.amp.autocast('cuda', dtype=dtype, enabled=training_config.mixed_precision):
                    loss = compute_loss(model, input_ids, target_ids)
                    loss = loss / training_config.gradient_accumulation_steps

                scaler.scale(loss).backward()

                accumulated_loss += loss.item()
                tokens_processed += input_ids.numel()

            # Unscale gradients and clip
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.max_grad_norm)

            # Update learning rate
            current_lr = schedule(step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            # Optimizer step
            scaler.step(optimizer)
            scaler.update()

            # Calculate metrics
            avg_loss = accumulated_loss
            accumulated_loss = 0.0

            # Update progress bar
            pbar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "lr": f"{current_lr:.2e}",
                "gnorm": f"{grad_norm:.2f}",
            })

            # Logging
            if (step + 1) % training_config.log_every == 0:
                elapsed = time.time() - step_start_time
                tps = tokens_processed / max(elapsed, 1e-6)

                metrics_logger.log(
                    step=step + 1,
                    train_loss=avg_loss,
                    learning_rate=current_lr,
                    tokens_per_sec=tps,
                    grad_norm=float(grad_norm),
                )

                logger.info(
                    f"Step {step + 1}: loss={avg_loss:.4f}, lr={current_lr:.2e}, "
                    f"grad_norm={grad_norm:.2f}, tokens/s={tps:,.0f}"
                )

                step_start_time = time.time()
                tokens_processed = 0

            # Evaluation
            if (step + 1) % training_config.eval_every == 0:
                val_loss = evaluate(model, val_loader, training_config.eval_batches, device)
                metrics_logger.log(step=step + 1, val_loss=val_loss)
                logger.info(f"Validation loss: {val_loss:.4f}")

                if val_loss < best_val_loss - training_config.early_stopping_min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                    is_best = True
                else:
                    patience_counter += 1
                    is_best = False

                if patience_counter >= training_config.early_stopping_patience:
                    logger.info(f"Early stopping at step {step + 1}")
                    save_checkpoint(
                        model, optimizer, model_config, training_config,
                        step + 1, val_loss, training_config.checkpoint_dir, is_best,
                    )
                    break

            # Checkpointing
            if (step + 1) % training_config.save_every == 0:
                val_loss = evaluate(model, val_loader, training_config.eval_batches, device)
                is_best = val_loss < best_val_loss
                if is_best:
                    best_val_loss = val_loss

                save_checkpoint(
                    model, optimizer, model_config, training_config,
                    step + 1, val_loss, training_config.checkpoint_dir, is_best,
                )

            current_step = step + 1

        # Training complete
        logger.info("Training complete!")
        val_loss = evaluate(model, val_loader, training_config.eval_batches, device)
        save_checkpoint(
            model, optimizer, model_config, training_config,
            training_config.max_steps, val_loss, training_config.checkpoint_dir,
            is_best=val_loss < best_val_loss,
        )

    except KeyboardInterrupt:
        logger.warning("\nTraining interrupted - saving emergency checkpoint...")
        emergency_dir = Path(training_config.checkpoint_dir) / "emergency"
        save_checkpoint(
            model, optimizer, model_config, training_config,
            current_step, best_val_loss, str(emergency_dir), is_best=False,
        )
        logger.info(f"Resume with: --resume {emergency_dir}/step_{current_step}")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()

        emergency_dir = Path(training_config.checkpoint_dir) / "emergency"
        try:
            save_checkpoint(
                model, optimizer, model_config, training_config,
                current_step, best_val_loss, str(emergency_dir), is_best=False,
            )
        except Exception as save_error:
            logger.error(f"Failed to save emergency checkpoint: {save_error}")
        raise

    finally:
        signal.signal(signal.SIGINT, original_sigint)
        signal.signal(signal.SIGTERM, original_sigterm)


def load_config(config_path: str) -> Tuple[ModelConfig, TrainingConfig]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_config = ModelConfig.from_dict(config.get("model", {}))
    training_config = TrainingConfig.from_dict(config.get("training", {}))

    return model_config, training_config


def main():
    parser = argparse.ArgumentParser(description="Train transformer (PyTorch/CUDA)")
    parser.add_argument("--config", "-c", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--resume", "-r", type=str, default=None, help="Checkpoint to resume from")

    args = parser.parse_args()

    model_config, training_config = load_config(args.config)

    logger.info("Model configuration:")
    for k, v in model_config.to_dict().items():
        logger.info(f"  {k}: {v}")

    logger.info("Training configuration:")
    for k, v in training_config.to_dict().items():
        logger.info(f"  {k}: {v}")

    train(model_config, training_config, resume_from=args.resume)


if __name__ == "__main__":
    main()
