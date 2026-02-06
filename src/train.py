"""
Training Loop

Complete training pipeline for the transformer language model:
- AdamW optimizer with cosine learning rate schedule
- Gradient accumulation for larger effective batch sizes
- Real mixed precision training (FP16 weights and activations)
- Automatic checkpointing
- Validation and early stopping
- Buffered metrics logging
- Optimized gradient handling

Optimizations:
- Actual FP16 mixed precision (not just input casting)
- Buffered CSV logging to reduce I/O
- Single mx.eval() per gradient accumulation step
- Configurable evaluation batches
"""

import argparse
import json
import logging
import math
import os
import signal
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import yaml
from tqdm import tqdm

try:
    from .model import ModelConfig, TransformerModel, create_model, save_model, load_model
    from .data import TokenizedDataset, DataLoader, create_dataloaders
    from .tokenizer import Tokenizer
except ImportError:
    from model import ModelConfig, TransformerModel, create_model, save_model, load_model
    from data import TokenizedDataset, DataLoader, create_dataloaders
    from tokenizer import Tokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Optimization
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0

    # Schedule
    max_steps: int = 100000
    warmup_steps: int = 1000
    min_lr_ratio: float = 0.1  # Final LR = learning_rate * min_lr_ratio

    # Checkpointing
    save_every: int = 5000
    eval_every: int = 2500
    eval_batches: int = 25  # Number of batches for evaluation
    log_every: int = 50
    sample_every: int = 5000

    # Data
    train_data_path: str = "data/train.bin"
    val_data_path: str = "data/val.bin"
    tokenizer_path: str = "data/tokenizer.model"

    # Directories
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

    # Hardware
    mixed_precision: bool = True
    seed: int = 42

    # Early stopping
    early_stopping_patience: int = 10  # Number of evals without improvement
    early_stopping_min_delta: float = 0.001

    @classmethod
    def from_dict(cls, d: dict) -> "TrainingConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


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
            # Linear warmup
            return self.learning_rate * step / max(self.warmup_steps, 1)
        elif step >= self.max_steps:
            return self.min_lr
        else:
            # Cosine decay
            progress = (step - self.warmup_steps) / max(self.max_steps - self.warmup_steps, 1)
            return self.min_lr + 0.5 * (self.learning_rate - self.min_lr) * (1 + math.cos(math.pi * progress))


class MetricsLogger:
    """Log training metrics to CSV with buffering."""

    def __init__(self, log_dir: str, run_name: str, buffer_size: int = 10):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.log_file = self.log_dir / f"{run_name}.csv"
        self.metrics_history: List[Dict] = []
        self.buffer: List[str] = []
        self.buffer_size = buffer_size

        # Write header
        with open(self.log_file, "w") as f:
            f.write("step,timestamp,train_loss,val_loss,learning_rate,tokens_per_sec,grad_norm\n")

    def log(
        self,
        step: int,
        train_loss: Optional[float] = None,
        val_loss: Optional[float] = None,
        learning_rate: Optional[float] = None,
        tokens_per_sec: Optional[float] = None,
        grad_norm: Optional[float] = None,
    ):
        """Log metrics with buffering."""
        metrics = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": learning_rate,
            "tokens_per_sec": tokens_per_sec,
            "grad_norm": grad_norm,
        }
        self.metrics_history.append(metrics)

        # Format CSV line
        values = [
            str(step),
            datetime.now().isoformat(),
            f"{train_loss:.6f}" if train_loss is not None else "",
            f"{val_loss:.6f}" if val_loss is not None else "",
            f"{learning_rate:.2e}" if learning_rate is not None else "",
            f"{tokens_per_sec:.1f}" if tokens_per_sec is not None else "",
            f"{grad_norm:.4f}" if grad_norm is not None else "",
        ]
        self.buffer.append(",".join(values) + "\n")

        # Flush buffer if full
        if len(self.buffer) >= self.buffer_size:
            self.flush()

    def flush(self):
        """Write buffered metrics to file."""
        if self.buffer:
            with open(self.log_file, "a") as f:
                f.writelines(self.buffer)
            self.buffer = []

    def __del__(self):
        """Flush on cleanup."""
        self.flush()


def compute_loss(model: TransformerModel, input_ids: mx.array, target_ids: mx.array) -> mx.array:
    """Compute cross-entropy loss.

    Args:
        model: Transformer model
        input_ids: Input token IDs (batch, seq_len)
        target_ids: Target token IDs (batch, seq_len)

    Returns:
        Scalar loss value
    """
    logits, _ = model(input_ids)

    # Reshape for cross-entropy: (batch * seq_len, vocab_size)
    batch_size, seq_len, vocab_size = logits.shape
    logits = logits.reshape(-1, vocab_size)
    targets = target_ids.reshape(-1)

    # Cross-entropy loss (computed in float32 for stability)
    logits_f32 = logits.astype(mx.float32)
    loss = nn.losses.cross_entropy(logits_f32, targets, reduction="mean")

    return loss


def train_step(
    model: TransformerModel,
    input_ids: mx.array,
    target_ids: mx.array,
) -> Tuple[mx.array, dict]:
    """Single training step.

    Args:
        model: Transformer model
        input_ids: Input tokens
        target_ids: Target tokens

    Returns:
        Loss value and gradients
    """
    loss_and_grad_fn = nn.value_and_grad(model, lambda m, x, y: compute_loss(m, x, y))
    loss, grads = loss_and_grad_fn(model, input_ids, target_ids)
    return loss, grads


def accumulate_grads(accumulated: Optional[dict], new: dict) -> dict:
    """Recursively accumulate gradients.

    Args:
        accumulated: Previously accumulated gradients (or None for first step)
        new: New gradients to add

    Returns:
        Accumulated gradients
    """
    if accumulated is None:
        return new

    def add_recursive(acc, n):
        if isinstance(acc, dict):
            return {k: add_recursive(acc[k], n[k]) for k in acc}
        elif isinstance(acc, list):
            return [add_recursive(a, b) for a, b in zip(acc, n)]
        else:
            return acc + n

    return add_recursive(accumulated, new)


def scale_grads(grads: dict, scale: float) -> dict:
    """Recursively scale gradients.

    Args:
        grads: Gradient tree
        scale: Scale factor

    Returns:
        Scaled gradients
    """
    def scale_recursive(g):
        if isinstance(g, dict):
            return {k: scale_recursive(v) for k, v in g.items()}
        elif isinstance(g, list):
            return [scale_recursive(v) for v in g]
        else:
            return g * scale

    return scale_recursive(grads)


def clip_gradients(grads: dict, max_norm: float) -> Tuple[dict, float]:
    """Clip gradients by global norm.

    Args:
        grads: Gradient dictionary
        max_norm: Maximum gradient norm

    Returns:
        Clipped gradients and original norm
    """
    # Flatten all gradients recursively
    def flatten_grads(g):
        flat = []
        if isinstance(g, dict):
            for v in g.values():
                flat.extend(flatten_grads(v))
        elif isinstance(g, list):
            for v in g:
                flat.extend(flatten_grads(v))
        else:
            # Ensure float32 for norm computation
            flat.append(g.astype(mx.float32).reshape(-1))
        return flat

    flat_grads = flatten_grads(grads)

    # Compute norm in float32
    total_norm_sq = sum(mx.sum(g * g) for g in flat_grads)
    total_norm = mx.sqrt(total_norm_sq)

    # Clip coefficient
    clip_coef = max_norm / (total_norm + 1e-8)
    clip_coef = mx.minimum(clip_coef, mx.array(1.0))

    def clip_recursive(g):
        if isinstance(g, dict):
            return {k: clip_recursive(v) for k, v in g.items()}
        elif isinstance(g, list):
            return [clip_recursive(v) for v in g]
        else:
            return g * clip_coef

    clipped_grads = clip_recursive(grads)

    # Evaluate to get the norm value and clipped gradients
    mx.eval(total_norm, clipped_grads)

    return clipped_grads, float(total_norm)


def evaluate(
    model: TransformerModel,
    val_loader: DataLoader,
    max_batches: int = 25,
) -> float:
    """Evaluate model on validation set.

    Args:
        model: Transformer model
        val_loader: Validation data loader
        max_batches: Maximum batches to evaluate

    Returns:
        Average validation loss
    """
    total_loss = 0.0
    num_batches = 0

    for i, (input_ids, target_ids) in enumerate(val_loader):
        if i >= max_batches:
            break

        loss = compute_loss(model, input_ids, target_ids)
        mx.eval(loss)
        total_loss += float(loss)
        num_batches += 1

    return total_loss / max(num_batches, 1)


def generate_sample(
    model: TransformerModel,
    tokenizer: Tokenizer,
    prompt: str = "Once upon a time",
    max_tokens: int = 50,
    temperature: float = 0.8,
) -> str:
    """Generate sample text from model.

    Args:
        model: Transformer model
        tokenizer: Tokenizer
        prompt: Starting prompt
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        Generated text
    """
    # Encode prompt
    input_ids = mx.array([tokenizer.encode(prompt)])

    # Generate
    cache = None
    generated = []

    for _ in range(max_tokens):
        if cache is None:
            # First step: process full prompt
            logits, cache = model(input_ids)
            logits = logits[:, -1, :]
        else:
            # Subsequent steps: process only last token
            logits, cache = model(input_ids[:, -1:], cache=cache)
            logits = logits[:, -1, :]

        # Sample (in float32 for stability)
        logits = logits.astype(mx.float32) / temperature
        probs = mx.softmax(logits, axis=-1)
        next_token = mx.random.categorical(probs)
        mx.eval(next_token)

        next_token_id = int(next_token[0])
        generated.append(next_token_id)

        # Check for EOS
        if next_token_id == tokenizer.eos_id:
            break

        # Update input for next iteration
        input_ids = mx.concatenate([input_ids, next_token.reshape(1, 1)], axis=1)

    return prompt + tokenizer.decode(generated)


def save_checkpoint(
    model: TransformerModel,
    optimizer: optim.Optimizer,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    step: int,
    val_loss: float,
    checkpoint_dir: str,
    is_best: bool = False,
):
    """Save training checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state
        model_config: Model configuration
        training_config: Training configuration
        step: Current step
        val_loss: Validation loss
        checkpoint_dir: Directory to save to
        is_best: Whether this is the best model so far
    """
    import shutil
    from mlx.utils import tree_flatten

    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    # Save model weights
    step_dir = checkpoint_path / f"step_{step}"
    step_dir.mkdir(exist_ok=True)

    # Flatten and convert to float32 for saving
    flat_params = tree_flatten(model.parameters())
    weights_f32 = {k: v.astype(mx.float32) for k, v in flat_params}

    weights_path = step_dir / "model.safetensors"
    mx.save_safetensors(str(weights_path), weights_f32)

    # Save optimizer state
    optimizer_path = step_dir / "optimizer.npz"
    opt_state = {}
    for i, (key, state) in enumerate(optimizer.state.items()):
        if isinstance(state, dict):
            for sk, sv in state.items():
                if isinstance(sv, mx.array):
                    opt_state[f"{i}_{sk}"] = sv.astype(mx.float32)
        elif isinstance(state, mx.array):
            opt_state[str(i)] = state.astype(mx.float32)
    if opt_state:
        mx.savez(str(optimizer_path), **opt_state)

    # Save configs
    config_path = step_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump({
            "model": model_config.to_dict(),
            "training": training_config.to_dict(),
            "step": step,
            "val_loss": val_loss,
        }, f, indent=2)

    logger.info(f"Saved checkpoint to {step_dir}")

    # Save as best if applicable
    if is_best:
        best_dir = checkpoint_path / "best"
        if best_dir.exists():
            shutil.rmtree(best_dir)
        shutil.copytree(step_dir, best_dir)
        logger.info(f"Saved as best model (val_loss={val_loss:.4f})")

    # Clean up old checkpoints (keep last 3 + best)
    all_checkpoints = sorted(
        [d for d in checkpoint_path.iterdir() if d.is_dir() and d.name.startswith("step_")],
        key=lambda x: int(x.name.split("_")[1])
    )
    if len(all_checkpoints) > 3:
        for old_ckpt in all_checkpoints[:-3]:
            shutil.rmtree(old_ckpt)
            logger.info(f"Removed old checkpoint: {old_ckpt}")


def load_checkpoint(
    checkpoint_path: str,
    model: TransformerModel,
    optimizer: optim.Optimizer,
    training_dtype: mx.Dtype = mx.float32,
) -> Tuple[int, float]:
    """Load training checkpoint.

    Args:
        checkpoint_path: Path to checkpoint directory
        model: Model to load weights into
        optimizer: Optimizer to load state into
        training_dtype: Dtype to convert weights to

    Returns:
        Tuple of (step, val_loss)
    """
    checkpoint_path = Path(checkpoint_path)

    # Load config
    config_path = checkpoint_path / "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    step = config["step"]
    val_loss = config["val_loss"]

    # Load model weights
    weights_path = checkpoint_path / "model.safetensors"
    weights = mx.load(str(weights_path))

    # Convert to training dtype
    if training_dtype != mx.float32:
        weights = {k: v.astype(training_dtype) if isinstance(v, mx.array) else v
                   for k, v in weights.items()}

    model.load_weights(list(weights.items()))

    logger.info(f"Loaded checkpoint from step {step} (val_loss={val_loss:.4f})")
    return step, val_loss


def preflight_check(
    model: TransformerModel,
    optimizer: optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    schedule: CosineSchedule,
    tokenizer: Tokenizer,
) -> bool:
    """Run pre-flight checks to validate all training components work together.

    This function tests the entire training pipeline before starting the real
    training loop, catching errors early instead of hours into training.

    Args:
        model: The model to test
        optimizer: The optimizer
        train_loader: Training data loader
        val_loader: Validation data loader
        model_config: Model configuration
        training_config: Training configuration
        schedule: Learning rate schedule
        tokenizer: Tokenizer for sample generation

    Returns:
        True if all checks pass, False otherwise
    """
    logger.info("=" * 50)
    logger.info("Running pre-flight checks...")
    logger.info("=" * 50)

    try:
        # 1. Test forward pass
        logger.info("[1/7] Testing forward pass...")
        train_iter = iter(train_loader)
        input_ids, target_ids = next(train_iter)
        logits, _ = model(input_ids)
        mx.eval(logits)
        logger.info(f"      Forward pass OK - output shape: {logits.shape}")

        # 2. Test backward pass (loss + gradients)
        logger.info("[2/7] Testing backward pass...")
        loss, grads = train_step(model, input_ids, target_ids)
        mx.eval(loss, grads)
        logger.info(f"      Backward pass OK - loss: {float(loss):.4f}")

        # 3. Test gradient accumulation
        logger.info("[3/7] Testing gradient accumulation...")
        accumulated_grads = grads
        input_ids2, target_ids2 = next(train_iter)
        loss2, grads2 = train_step(model, input_ids2, target_ids2)
        accumulated_grads = accumulate_grads(accumulated_grads, grads2)
        mx.eval(accumulated_grads)
        logger.info("      Gradient accumulation OK")

        # 4. Test gradient clipping and optimizer update
        logger.info("[4/7] Testing optimizer update...")
        scaled_grads = scale_grads(accumulated_grads, 0.5)
        clipped_grads, grad_norm = clip_gradients(scaled_grads, training_config.max_grad_norm)
        current_lr = schedule(0)
        optimizer.learning_rate = mx.array(current_lr)
        optimizer.update(model, clipped_grads)
        mx.eval(model.parameters())
        logger.info(f"      Optimizer update OK - grad_norm: {grad_norm:.4f}")

        # 5. Test checkpoint save
        logger.info("[5/7] Testing checkpoint save...")
        with tempfile.TemporaryDirectory() as tmpdir:
            save_checkpoint(
                model, optimizer, model_config, training_config,
                step=0, val_loss=float(loss), checkpoint_dir=tmpdir, is_best=False
            )
            # Verify files exist
            ckpt_dir = Path(tmpdir) / "step_0"
            assert (ckpt_dir / "model.safetensors").exists(), "Model weights not saved"
            assert (ckpt_dir / "config.json").exists(), "Config not saved"
            logger.info("      Checkpoint save OK")

            # 6. Test checkpoint load
            logger.info("[6/7] Testing checkpoint load...")
            step, val_loss_loaded = load_checkpoint(
                str(ckpt_dir), model, optimizer,
                training_dtype=mx.float16 if training_config.mixed_precision else mx.float32
            )
            logger.info(f"      Checkpoint load OK - step: {step}")

        # 7. Test evaluation
        logger.info("[7/7] Testing evaluation...")
        val_loss = evaluate(model, val_loader, max_batches=2)
        logger.info(f"      Evaluation OK - val_loss: {val_loss:.4f}")

        logger.info("=" * 50)
        logger.info("All pre-flight checks PASSED!")
        logger.info("=" * 50)
        return True

    except Exception as e:
        logger.error("=" * 50)
        logger.error(f"Pre-flight check FAILED: {e}")
        logger.error("=" * 50)
        import traceback
        traceback.print_exc()
        return False


# Global variable for signal handling
_training_state = {
    "model": None,
    "optimizer": None,
    "model_config": None,
    "training_config": None,
    "step": 0,
    "val_loss": float("inf"),
    "interrupted": False,
}


def _signal_handler(signum, frame):
    """Handle interrupt signals gracefully."""
    signal_name = signal.Signals(signum).name
    logger.warning(f"\nReceived {signal_name} - saving emergency checkpoint...")
    _training_state["interrupted"] = True


def train(
    model_config: ModelConfig,
    training_config: TrainingConfig,
    resume_from: Optional[str] = None,
    skip_preflight: bool = False,
):
    """Main training loop.

    Args:
        model_config: Model configuration
        training_config: Training configuration
        resume_from: Optional checkpoint path to resume from
        skip_preflight: Skip pre-flight checks (use with caution)
    """
    global _training_state

    # Set random seed
    mx.random.seed(training_config.seed)
    np.random.seed(training_config.seed)

    # Determine training dtype
    training_dtype = mx.float16 if training_config.mixed_precision else mx.float32
    dtype_name = "float16" if training_config.mixed_precision else "float32"
    logger.info(f"Training with {dtype_name} precision")

    # Create model
    logger.info(f"Creating model with ~{model_config.estimate_parameters():,} parameters")
    model = create_model(model_config, dtype=training_dtype)
    actual_params = model.count_parameters()
    logger.info(f"Actual parameters: {actual_params:,}")

    # Estimate memory
    bytes_per_param = 2 if training_config.mixed_precision else 4
    param_memory_gb = actual_params * bytes_per_param / (1024**3)
    logger.info(f"Estimated parameter memory: {param_memory_gb:.2f} GB")

    # Create optimizer
    schedule = CosineSchedule(
        training_config.learning_rate,
        training_config.warmup_steps,
        training_config.max_steps,
        training_config.min_lr_ratio,
    )

    # Create optimizer - we'll set learning rate manually each step for FP16 compatibility
    optimizer = optim.AdamW(
        learning_rate=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    # Create data loaders
    logger.info("Loading data...")
    train_loader, val_loader = create_dataloaders(
        training_config.train_data_path,
        training_config.val_data_path,
        model_config.context_length,
        training_config.batch_size,
        seed=training_config.seed,
    )

    # Load tokenizer for sample generation
    tokenizer = Tokenizer(training_config.tokenizer_path)

    # Initialize metrics logger
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_logger = MetricsLogger(training_config.log_dir, run_name)

    # Resume from checkpoint if specified
    start_step = 0
    best_val_loss = float("inf")
    patience_counter = 0

    if resume_from:
        start_step, best_val_loss = load_checkpoint(
            resume_from, model, optimizer, training_dtype
        )
        start_step += 1  # Start from next step

    # Run pre-flight checks (unless resuming or explicitly skipped)
    if not skip_preflight and not resume_from:
        if not preflight_check(
            model, optimizer, train_loader, val_loader,
            model_config, training_config, schedule, tokenizer
        ):
            logger.error("Pre-flight checks failed. Fix the issues above before training.")
            sys.exit(1)

        # Re-create model and optimizer after preflight (they were modified)
        logger.info("Re-initializing model after pre-flight checks...")
        mx.random.seed(training_config.seed)
        model = create_model(model_config, dtype=training_dtype)
        optimizer = optim.AdamW(
            learning_rate=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
        )

    # Store training state for signal handler
    _training_state.update({
        "model": model,
        "optimizer": optimizer,
        "model_config": model_config,
        "training_config": training_config,
        "step": start_step,
        "val_loss": best_val_loss,
        "interrupted": False,
    })

    # Register signal handlers for graceful shutdown
    original_sigint = signal.signal(signal.SIGINT, _signal_handler)
    original_sigterm = signal.signal(signal.SIGTERM, _signal_handler)

    # Training loop
    logger.info(f"Starting training from step {start_step}")

    effective_batch_size = training_config.batch_size * training_config.gradient_accumulation_steps
    tokens_per_step = effective_batch_size * model_config.context_length
    logger.info(f"Effective batch size: {effective_batch_size}")
    logger.info(f"Tokens per step: {tokens_per_step:,}")

    train_iter = iter(train_loader)
    accumulated_loss = 0.0
    accumulated_grads = None
    step_start_time = time.time()
    tokens_processed = 0
    current_step = start_step  # Track for emergency saves

    pbar = tqdm(range(start_step, training_config.max_steps), desc="Training")

    try:
        for step in pbar:
            # Check for interrupt
            if _training_state["interrupted"]:
                logger.info("Interrupt detected - saving checkpoint and exiting...")
                raise KeyboardInterrupt("User interrupt")

            # Gradient accumulation
            for _ in range(training_config.gradient_accumulation_steps):
                try:
                    input_ids, target_ids = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    input_ids, target_ids = next(train_iter)

                loss, grads = train_step(model, input_ids, target_ids)

                accumulated_loss += float(loss)
                tokens_processed += input_ids.shape[0] * input_ids.shape[1]

                # Accumulate gradients
                accumulated_grads = accumulate_grads(accumulated_grads, grads)

            # Evaluate loss and gradients once per optimization step
            mx.eval(accumulated_grads)

            # Average gradients
            scale = 1.0 / training_config.gradient_accumulation_steps
            accumulated_grads = scale_grads(accumulated_grads, scale)

            # Clip gradients
            clipped_grads, grad_norm = clip_gradients(
                accumulated_grads,
                training_config.max_grad_norm,
            )

            # Update learning rate for this step (required for FP16 compatibility)
            current_lr = schedule(step)
            optimizer.learning_rate = mx.array(current_lr)

            # Update parameters
            optimizer.update(model, clipped_grads)
            mx.eval(model.parameters())

            # Calculate metrics
            avg_loss = accumulated_loss / training_config.gradient_accumulation_steps

            # Reset accumulators
            accumulated_loss = 0.0
            accumulated_grads = None

            # Update progress bar
            pbar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "lr": f"{current_lr:.2e}",
                "gnorm": f"{grad_norm:.2f}",
            })

            # Logging
            if (step + 1) % training_config.log_every == 0:
                elapsed = time.time() - step_start_time
                tokens_per_sec = tokens_processed / max(elapsed, 1e-6)

                metrics_logger.log(
                    step=step + 1,
                    train_loss=avg_loss,
                    learning_rate=current_lr,
                    tokens_per_sec=tokens_per_sec,
                    grad_norm=grad_norm,
                )

                logger.info(
                    f"Step {step + 1}: loss={avg_loss:.4f}, lr={current_lr:.2e}, "
                    f"grad_norm={grad_norm:.2f}, tokens/s={tokens_per_sec:.0f}"
                )

                step_start_time = time.time()
                tokens_processed = 0

            # Evaluation
            if (step + 1) % training_config.eval_every == 0:
                logger.info("Running evaluation...")
                val_loss = evaluate(model, val_loader, max_batches=training_config.eval_batches)

                metrics_logger.log(step=step + 1, val_loss=val_loss)
                logger.info(f"Validation loss: {val_loss:.4f}")

                # Early stopping check
                if val_loss < best_val_loss - training_config.early_stopping_min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                    is_best = True
                else:
                    patience_counter += 1
                    is_best = False

                if patience_counter >= training_config.early_stopping_patience:
                    logger.info(f"Early stopping triggered after {step + 1} steps")
                    save_checkpoint(
                        model, optimizer, model_config, training_config,
                        step + 1, val_loss, training_config.checkpoint_dir, is_best=is_best,
                    )
                    break

            # Sample generation
            if (step + 1) % training_config.sample_every == 0:
                logger.info("Generating sample...")
                try:
                    sample = generate_sample(model, tokenizer, temperature=0.8)
                    logger.info(f"Sample: {sample}")
                except Exception as e:
                    logger.warning(f"Sample generation failed: {e}")

            # Checkpointing
            if (step + 1) % training_config.save_every == 0:
                val_loss = evaluate(model, val_loader, max_batches=training_config.eval_batches)
                is_best = val_loss < best_val_loss
                if is_best:
                    best_val_loss = val_loss

                save_checkpoint(
                    model, optimizer, model_config, training_config,
                    step + 1, val_loss, training_config.checkpoint_dir, is_best=is_best,
                )

            # Update current step for emergency saves
            current_step = step + 1

        # Normal completion - final save
        logger.info("Training complete!")
        metrics_logger.flush()
        val_loss = evaluate(model, val_loader, max_batches=training_config.eval_batches)
        save_checkpoint(
            model, optimizer, model_config, training_config,
            training_config.max_steps, val_loss, training_config.checkpoint_dir,
            is_best=val_loss < best_val_loss,
        )

    except KeyboardInterrupt:
        # Graceful shutdown on Ctrl+C
        logger.warning("\n" + "=" * 50)
        logger.warning("Training interrupted by user")
        logger.warning("=" * 50)
        logger.info(f"Saving emergency checkpoint at step {current_step}...")
        try:
            emergency_dir = Path(training_config.checkpoint_dir) / "emergency"
            save_checkpoint(
                model, optimizer, model_config, training_config,
                current_step, best_val_loss, str(emergency_dir), is_best=False,
            )
            logger.info(f"Emergency checkpoint saved to {emergency_dir}")
            logger.info(f"Resume with: --resume {emergency_dir}/step_{current_step}")
        except Exception as save_error:
            logger.error(f"Failed to save emergency checkpoint: {save_error}")

    except Exception as e:
        # Unexpected error - try to save before crashing
        logger.error("\n" + "=" * 50)
        logger.error(f"Training failed with error: {e}")
        logger.error("=" * 50)
        import traceback
        traceback.print_exc()

        logger.info(f"Attempting emergency checkpoint at step {current_step}...")
        try:
            emergency_dir = Path(training_config.checkpoint_dir) / "emergency"
            save_checkpoint(
                model, optimizer, model_config, training_config,
                current_step, best_val_loss, str(emergency_dir), is_best=False,
            )
            logger.info(f"Emergency checkpoint saved to {emergency_dir}")
            logger.info(f"Resume with: --resume {emergency_dir}/step_{current_step}")
        except Exception as save_error:
            logger.error(f"Failed to save emergency checkpoint: {save_error}")

        raise  # Re-raise the original exception

    finally:
        # Restore original signal handlers
        signal.signal(signal.SIGINT, original_sigint)
        signal.signal(signal.SIGTERM, original_sigterm)

        # Ensure metrics are flushed
        metrics_logger.flush()

        logger.info("Training session ended.")


def load_config(config_path: str) -> Tuple[ModelConfig, TrainingConfig]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config YAML

    Returns:
        Tuple of (ModelConfig, TrainingConfig)
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_config = ModelConfig.from_dict(config.get("model", {}))
    training_config = TrainingConfig.from_dict(config.get("training", {}))

    return model_config, training_config


def benchmark_speed(
    model_config: ModelConfig,
    training_config: TrainingConfig,
    num_steps: int = 10,
    warmup_steps: int = 3,
) -> dict:
    """Benchmark training speed to estimate total training time.

    Args:
        model_config: Model configuration
        training_config: Training configuration
        num_steps: Number of steps to benchmark (after warmup)
        warmup_steps: Number of warmup steps (not counted)

    Returns:
        Dictionary with benchmark results
    """
    logger.info("=" * 60)
    logger.info("TRAINING TIME ESTIMATION")
    logger.info("=" * 60)

    # Determine training dtype
    training_dtype = mx.float16 if training_config.mixed_precision else mx.float32
    dtype_name = "float16" if training_config.mixed_precision else "float32"

    # Create model
    logger.info(f"Creating model ({dtype_name})...")
    model = create_model(model_config, dtype=training_dtype)
    actual_params = model.count_parameters()
    logger.info(f"Model parameters: {actual_params:,}")

    # Create data loader
    logger.info("Loading data...")
    train_loader, _ = create_dataloaders(
        training_config.train_data_path,
        training_config.val_data_path,
        model_config.context_length,
        training_config.batch_size,
        seed=training_config.seed,
    )

    # Create optimizer (needed for full step simulation)
    optimizer = optim.AdamW(
        learning_rate=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    train_iter = iter(train_loader)
    tokens_per_step = training_config.batch_size * training_config.gradient_accumulation_steps * model_config.context_length

    # Warmup (not timed - first steps are slower due to compilation)
    logger.info(f"Running {warmup_steps} warmup steps...")
    for _ in range(warmup_steps):
        accumulated_grads = None
        for _ in range(training_config.gradient_accumulation_steps):
            try:
                input_ids, target_ids = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                input_ids, target_ids = next(train_iter)

            loss, grads = train_step(model, input_ids, target_ids)
            accumulated_grads = accumulate_grads(accumulated_grads, grads)

        mx.eval(accumulated_grads)
        scale = 1.0 / training_config.gradient_accumulation_steps
        accumulated_grads = scale_grads(accumulated_grads, scale)
        clipped_grads, _ = clip_gradients(accumulated_grads, training_config.max_grad_norm)
        optimizer.learning_rate = mx.array(training_config.learning_rate)
        optimizer.update(model, clipped_grads)
        mx.eval(model.parameters())

    # Timed benchmark
    logger.info(f"Benchmarking {num_steps} steps...")
    step_times = []

    for step in range(num_steps):
        step_start = time.time()

        accumulated_grads = None
        for _ in range(training_config.gradient_accumulation_steps):
            try:
                input_ids, target_ids = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                input_ids, target_ids = next(train_iter)

            loss, grads = train_step(model, input_ids, target_ids)
            accumulated_grads = accumulate_grads(accumulated_grads, grads)

        mx.eval(accumulated_grads)
        scale = 1.0 / training_config.gradient_accumulation_steps
        accumulated_grads = scale_grads(accumulated_grads, scale)
        clipped_grads, _ = clip_gradients(accumulated_grads, training_config.max_grad_norm)
        optimizer.learning_rate = mx.array(training_config.learning_rate)
        optimizer.update(model, clipped_grads)
        mx.eval(model.parameters())

        step_time = time.time() - step_start
        step_times.append(step_time)
        tokens_per_sec = tokens_per_step / step_time
        logger.info(f"  Step {step + 1}/{num_steps}: {step_time:.3f}s ({tokens_per_sec:,.0f} tokens/sec)")

    # Calculate statistics
    avg_step_time = sum(step_times) / len(step_times)
    min_step_time = min(step_times)
    max_step_time = max(step_times)
    avg_tokens_per_sec = tokens_per_step / avg_step_time

    # Estimate total training time
    total_steps = training_config.max_steps
    total_tokens = total_steps * tokens_per_step

    est_seconds = total_steps * avg_step_time
    est_hours = est_seconds / 3600
    est_days = est_hours / 24

    # Results
    results = {
        "model_params": actual_params,
        "tokens_per_step": tokens_per_step,
        "avg_step_time": avg_step_time,
        "min_step_time": min_step_time,
        "max_step_time": max_step_time,
        "avg_tokens_per_sec": avg_tokens_per_sec,
        "total_steps": total_steps,
        "total_tokens": total_tokens,
        "est_seconds": est_seconds,
        "est_hours": est_hours,
        "est_days": est_days,
    }

    # Print summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("BENCHMARK RESULTS")
    logger.info("=" * 60)
    logger.info(f"Model parameters:     {actual_params:,}")
    logger.info(f"Tokens per step:      {tokens_per_step:,}")
    logger.info(f"Precision:            {dtype_name}")
    logger.info("")
    logger.info(f"Avg step time:        {avg_step_time:.3f}s (range: {min_step_time:.3f}s - {max_step_time:.3f}s)")
    logger.info(f"Throughput:           {avg_tokens_per_sec:,.0f} tokens/sec")
    logger.info("")
    logger.info("-" * 60)
    logger.info("ESTIMATED TRAINING TIME")
    logger.info("-" * 60)
    logger.info(f"Total steps:          {total_steps:,}")
    logger.info(f"Total tokens:         {total_tokens:,}")
    logger.info("")

    if est_days >= 1:
        logger.info(f"Estimated time:       {est_days:.1f} days ({est_hours:.1f} hours)")
    elif est_hours >= 1:
        logger.info(f"Estimated time:       {est_hours:.1f} hours ({est_seconds/60:.0f} minutes)")
    else:
        logger.info(f"Estimated time:       {est_seconds/60:.1f} minutes")

    logger.info("")
    logger.info("=" * 60)

    # Cloud comparison
    logger.info("")
    logger.info("CLOUD COMPARISON (approximate)")
    logger.info("-" * 60)

    cloud_speedups = [
        ("RTX 4090 (Vast.ai)", 15, 0.40),
        ("A100 40GB (Lambda)", 25, 1.10),
        ("A100 80GB (RunPod)", 35, 1.60),
        ("H100 (GCP)", 50, 3.50),
    ]

    for name, speedup, cost_per_hour in cloud_speedups:
        cloud_hours = est_hours / speedup
        cloud_cost = cloud_hours * cost_per_hour
        if cloud_hours >= 1:
            logger.info(f"  {name}: ~{cloud_hours:.1f} hours, ~${cloud_cost:.2f}")
        else:
            logger.info(f"  {name}: ~{cloud_hours*60:.0f} min, ~${cloud_cost:.2f}")

    logger.info("")
    logger.info("=" * 60)

    return results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Train transformer language model")
    parser.add_argument("--config", "-c", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--resume", "-r", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--estimate", "-e", action="store_true", help="Benchmark speed and estimate training time without training")
    parser.add_argument("--skip-preflight", action="store_true", help="Skip pre-flight validation checks")

    args = parser.parse_args()

    # Load config
    model_config, training_config = load_config(args.config)

    logger.info("Model configuration:")
    for k, v in model_config.to_dict().items():
        logger.info(f"  {k}: {v}")

    logger.info("Training configuration:")
    for k, v in training_config.to_dict().items():
        logger.info(f"  {k}: {v}")

    # Estimate mode - benchmark and exit
    if args.estimate:
        benchmark_speed(model_config, training_config)
        return

    # Train
    train(model_config, training_config, resume_from=args.resume, skip_preflight=args.skip_preflight)


if __name__ == "__main__":
    main()
