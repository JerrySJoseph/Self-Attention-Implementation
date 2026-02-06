#!/usr/bin/env python3
"""
Convert PyTorch checkpoint to MLX format.

Usage:
    python scripts/convert_torch_to_mlx.py \
        --checkpoint checkpoints/best/model.pt \
        --output checkpoints/mlx/model.safetensors
"""

import argparse
import json
from pathlib import Path

import numpy as np


def convert_weights(checkpoint_path: str, output_path: str, config_path: str = None):
    """Convert PyTorch weights to MLX safetensors format.

    Args:
        checkpoint_path: Path to PyTorch model.pt file
        output_path: Path for output .safetensors file
        config_path: Optional path to config.json (auto-detected if not provided)
    """
    import torch

    print(f"Loading PyTorch checkpoint from {checkpoint_path}...")
    state_dict = torch.load(checkpoint_path, map_location="cpu")

    # Convert to numpy arrays (float32 for compatibility)
    print("Converting weights...")
    weights = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            # Convert to float32 numpy
            weights[key] = value.float().numpy()

    print(f"Converted {len(weights)} tensors")

    # Save in safetensors format
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Try using MLX's save_safetensors
        import mlx.core as mx

        mlx_weights = {k: mx.array(v) for k, v in weights.items()}
        mx.save_safetensors(str(output_path), mlx_weights)
        print(f"Saved MLX weights to {output_path}")

    except ImportError:
        # Fallback: use safetensors library directly
        try:
            from safetensors.numpy import save_file
            save_file(weights, str(output_path))
            print(f"Saved safetensors to {output_path}")
        except ImportError:
            # Last resort: save as numpy
            np_path = output_path.with_suffix('.npz')
            np.savez(str(np_path), **weights)
            print(f"Saved numpy weights to {np_path}")
            print("Install safetensors or mlx to save in .safetensors format")

    # Copy config if available
    if config_path is None:
        # Try to find config.json in same directory as checkpoint
        checkpoint_dir = Path(checkpoint_path).parent
        config_path = checkpoint_dir / "config.json"

    if Path(config_path).exists():
        import shutil
        output_config = output_path.parent / "config.json"
        shutil.copy(config_path, output_config)
        print(f"Copied config to {output_config}")

    print("\nConversion complete!")
    print(f"\nTo use with MLX inference:")
    print(f"  python -m src.inference --checkpoint {output_path.parent} --interactive")


def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch checkpoint to MLX")
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        required=True,
        help="Path to PyTorch model.pt file"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output path for .safetensors file (default: same dir with mlx suffix)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.json (auto-detected if not provided)"
    )

    args = parser.parse_args()

    # Default output path
    if args.output is None:
        checkpoint_path = Path(args.checkpoint)
        output_dir = checkpoint_path.parent.parent / f"{checkpoint_path.parent.name}_mlx"
        args.output = str(output_dir / "model.safetensors")

    convert_weights(args.checkpoint, args.output, args.config)


if __name__ == "__main__":
    main()
