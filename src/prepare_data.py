#!/usr/bin/env python3
"""
Data Preparation Script (No MLX dependency)

Works on any platform - use this for cloud GPU instances.

Usage:
    python -m src.prepare_data --raw-data raw_data --output-dir data --vocab-size 32000
"""

import argparse
import os
from pathlib import Path
import numpy as np


def train_tokenizer(input_dir: str, output_path: str, vocab_size: int = 32000):
    """Train a SentencePiece BPE tokenizer."""
    import sentencepiece as spm

    input_dir = Path(input_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Find all text files
    text_files = list(input_dir.glob("*.txt"))
    if not text_files:
        raise ValueError(f"No .txt files found in {input_dir}")

    print(f"Found {len(text_files)} text files")

    # Create a combined input file for sentencepiece
    combined_path = output_path.parent / "combined_training_text.txt"
    print(f"Combining text files...")

    with open(combined_path, "w", encoding="utf-8") as out_f:
        for txt_file in text_files:
            print(f"  Adding {txt_file.name}...")
            with open(txt_file, "r", encoding="utf-8") as in_f:
                for line in in_f:
                    line = line.strip()
                    if line:
                        out_f.write(line + "\n")

    # Train tokenizer
    model_prefix = str(output_path).replace(".model", "")

    print(f"Training tokenizer with vocab_size={vocab_size}...")
    spm.SentencePieceTrainer.train(
        input=str(combined_path),
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=0.9995,
        num_threads=os.cpu_count(),
        split_digits=True,
        byte_fallback=True,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
    )

    # Clean up
    combined_path.unlink()

    print(f"Tokenizer saved to {output_path}")
    return str(output_path) + ".model"


def tokenize_data(input_dir: str, output_path: str, tokenizer_path: str):
    """Tokenize all text files and save as binary."""
    import sentencepiece as spm

    input_dir = Path(input_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    sp = spm.SentencePieceProcessor()
    sp.load(tokenizer_path)

    print(f"Loaded tokenizer with vocab_size={sp.get_piece_size()}")

    # Find all text files
    text_files = list(input_dir.glob("*.txt"))

    # Tokenize everything
    all_tokens = []
    total_chars = 0

    for txt_file in text_files:
        print(f"Tokenizing {txt_file.name}...")
        with open(txt_file, "r", encoding="utf-8") as f:
            text = f.read()
            total_chars += len(text)

        tokens = sp.encode(text)
        all_tokens.extend(tokens)
        print(f"  {len(tokens):,} tokens from {len(text):,} chars")

    # Convert to numpy and save
    tokens_array = np.array(all_tokens, dtype=np.uint16)

    print(f"\nTotal: {len(tokens_array):,} tokens from {total_chars:,} chars")
    print(f"Compression ratio: {total_chars / len(tokens_array):.2f} chars/token")

    # Save as memory-mapped file
    tokens_array.tofile(output_path)
    print(f"Saved to {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")

    return str(output_path)


def split_data(input_path: str, train_path: str, val_path: str, val_ratio: float = 0.01):
    """Split tokenized data into train and validation sets."""
    input_path = Path(input_path)
    train_path = Path(train_path)
    val_path = Path(val_path)

    # Load data
    data = np.fromfile(input_path, dtype=np.uint16)
    total_tokens = len(data)

    # Split
    val_size = int(total_tokens * val_ratio)
    train_size = total_tokens - val_size

    print(f"Total tokens: {total_tokens:,}")
    print(f"Train tokens: {train_size:,} ({100*(1-val_ratio):.1f}%)")
    print(f"Val tokens:   {val_size:,} ({100*val_ratio:.1f}%)")

    # Save splits
    train_path.parent.mkdir(parents=True, exist_ok=True)
    val_path.parent.mkdir(parents=True, exist_ok=True)

    data[:train_size].tofile(train_path)
    data[train_size:].tofile(val_path)

    print(f"Saved {train_path} ({train_path.stat().st_size / 1e6:.1f} MB)")
    print(f"Saved {val_path} ({val_path.stat().st_size / 1e6:.1f} MB)")


def prepare_all(raw_data_dir: str, output_dir: str, vocab_size: int = 32000):
    """Full pipeline: train tokenizer, tokenize data, split."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("Step 1: Training tokenizer")
    print("=" * 50)
    tokenizer_path = train_tokenizer(
        raw_data_dir,
        output_dir / "tokenizer",
        vocab_size
    )

    print("\n" + "=" * 50)
    print("Step 2: Tokenizing data")
    print("=" * 50)
    full_path = tokenize_data(
        raw_data_dir,
        output_dir / "full.bin",
        tokenizer_path
    )

    print("\n" + "=" * 50)
    print("Step 3: Splitting into train/val")
    print("=" * 50)
    split_data(
        full_path,
        output_dir / "train.bin",
        output_dir / "val.bin"
    )

    # Clean up full.bin
    Path(full_path).unlink()

    print("\n" + "=" * 50)
    print("Data preparation complete!")
    print("=" * 50)
    print(f"\nFiles in {output_dir}:")
    for f in sorted(output_dir.iterdir()):
        if f.is_file():
            print(f"  {f.name}: {f.stat().st_size / 1e6:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Prepare training data")
    parser.add_argument("--raw-data", type=str, required=True, help="Directory with .txt files")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--vocab-size", type=int, default=32000, help="Vocabulary size")

    args = parser.parse_args()

    prepare_all(args.raw_data, args.output_dir, args.vocab_size)


if __name__ == "__main__":
    main()
