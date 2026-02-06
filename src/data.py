"""
Data Loading and Preprocessing

Implements efficient data loading for language model training:
- Streaming from disk (memory-mapped files)
- Tokenization and caching
- Dynamic batching
- Train/validation splitting
"""

import os
import argparse
import logging
import json
import random
from pathlib import Path
from typing import Generator, Iterator, List, Optional, Tuple, Union
import struct
import mmap

import numpy as np
import mlx.core as mx

try:
    from .tokenizer import Tokenizer, train_tokenizer
except ImportError:
    from tokenizer import Tokenizer, train_tokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class TokenizedDataset:
    """Memory-mapped tokenized dataset for efficient loading.

    Stores tokenized data as uint16 binary files for memory efficiency.
    Supports random access and streaming.
    """

    HEADER_SIZE = 16  # 8 bytes for magic, 8 bytes for length

    def __init__(self, path: str):
        """Load tokenized dataset.

        Args:
            path: Path to .bin file
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset not found: {path}")

        self.path = path
        self.file = open(path, "rb")
        self.mmap = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)

        # Read header
        magic = struct.unpack("Q", self.mmap[:8])[0]
        if magic != 0x544F4B454E53:  # "TOKENS" in hex
            raise ValueError(f"Invalid file format: {path}")

        self.length = struct.unpack("Q", self.mmap[8:16])[0]
        logger.info(f"Loaded dataset with {self.length:,} tokens from {path}")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: Union[int, slice]) -> np.ndarray:
        """Get token(s) at index."""
        if isinstance(idx, int):
            if idx < 0:
                idx = self.length + idx
            if idx < 0 or idx >= self.length:
                raise IndexError(f"Index {idx} out of range for dataset of length {self.length}")
            offset = self.HEADER_SIZE + idx * 2
            return np.frombuffer(self.mmap[offset:offset + 2], dtype=np.uint16)[0]

        elif isinstance(idx, slice):
            start, stop, step = idx.indices(self.length)
            if step != 1:
                # For non-contiguous slices, read one by one
                return np.array([self[i] for i in range(start, stop, step)], dtype=np.uint16)
            offset = self.HEADER_SIZE + start * 2
            length = stop - start
            return np.frombuffer(self.mmap[offset:offset + length * 2], dtype=np.uint16).copy()

        else:
            raise TypeError(f"Invalid index type: {type(idx)}")

    def get_chunk(self, start: int, length: int) -> np.ndarray:
        """Get a contiguous chunk of tokens.

        Args:
            start: Starting index
            length: Number of tokens to get

        Returns:
            Array of token IDs
        """
        if start < 0 or start + length > self.length:
            raise IndexError(f"Chunk ({start}, {start + length}) out of range")
        offset = self.HEADER_SIZE + start * 2
        return np.frombuffer(self.mmap[offset:offset + length * 2], dtype=np.uint16).copy()

    def close(self):
        """Close the memory-mapped file."""
        self.mmap.close()
        self.file.close()

    def __del__(self):
        try:
            self.close()
        except:
            pass


class DataLoader:
    """Efficient data loader for language model training.

    Features:
    - Memory-efficient streaming
    - Configurable context length
    - Shuffled batches
    - Reproducible with seed
    """

    def __init__(
        self,
        dataset: TokenizedDataset,
        context_length: int,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 42,
        drop_last: bool = True,
    ):
        """Initialize data loader.

        Args:
            dataset: Tokenized dataset
            context_length: Sequence length for training
            batch_size: Batch size
            shuffle: Whether to shuffle samples
            seed: Random seed for reproducibility
            drop_last: Drop last incomplete batch
        """
        self.dataset = dataset
        self.context_length = context_length
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last

        # Calculate number of complete sequences
        # Each sequence is context_length + 1 (input + target)
        seq_length = context_length + 1
        self.num_sequences = len(dataset) // seq_length
        self.num_batches = self.num_sequences // batch_size

        if self.drop_last:
            self.total_samples = self.num_batches * batch_size
        else:
            self.total_samples = self.num_sequences

        logger.info(
            f"DataLoader: {self.num_sequences:,} sequences, "
            f"{self.num_batches:,} batches of size {batch_size}"
        )

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[Tuple[mx.array, mx.array]]:
        """Iterate over batches.

        Yields:
            Tuple of (input_ids, target_ids), both shape (batch_size, context_length)
        """
        seq_length = self.context_length + 1

        # Create indices for all sequences
        indices = list(range(self.num_sequences))

        if self.shuffle:
            rng = random.Random(self.seed)
            rng.shuffle(indices)

        # Pre-allocate batch array for efficiency
        batch = np.empty((self.batch_size, seq_length), dtype=np.int32)

        # Yield batches
        for batch_idx in range(self.num_batches):
            batch_indices = indices[batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size]

            # Load sequences directly into pre-allocated array
            for i, idx in enumerate(batch_indices):
                start = idx * seq_length
                seq = self.dataset.get_chunk(start, seq_length)
                batch[i] = seq

            # Split into input and target (views, no copy)
            input_ids = mx.array(batch[:, :-1])
            target_ids = mx.array(batch[:, 1:])

            yield input_ids, target_ids

    def get_batch(self, batch_idx: int) -> Tuple[mx.array, mx.array]:
        """Get a specific batch by index.

        Args:
            batch_idx: Batch index

        Returns:
            Tuple of (input_ids, target_ids)
        """
        seq_length = self.context_length + 1
        sequences = []

        for i in range(self.batch_size):
            idx = batch_idx * self.batch_size + i
            start = idx * seq_length
            seq = self.dataset.get_chunk(start, seq_length)
            sequences.append(seq)

        batch = np.stack(sequences, axis=0)
        input_ids = mx.array(batch[:, :-1].astype(np.int32))
        target_ids = mx.array(batch[:, 1:].astype(np.int32))

        return input_ids, target_ids


def prepare_dataset(
    input_path: str,
    output_path: str,
    tokenizer: Tokenizer,
    add_eos: bool = True,
    max_files: Optional[int] = None,
    chunk_size: int = 10000,
) -> int:
    """Tokenize text files and save as binary dataset using streaming.

    Args:
        input_path: Path to input text file or directory
        output_path: Path to output .bin file
        tokenizer: Trained tokenizer
        add_eos: Add EOS token after each document
        max_files: Maximum number of files to process
        chunk_size: Number of documents to process before writing to disk

    Returns:
        Total number of tokens
    """
    input_path = Path(input_path)
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    # Collect input files
    if input_path.is_file():
        files = [input_path]
    else:
        files = sorted(input_path.glob("**/*.txt"))
        if max_files:
            files = files[:max_files]

    logger.info(f"Processing {len(files)} files")

    # Temporary file for streaming writes
    temp_path = output_path + ".tmp"
    total_tokens = 0

    with open(temp_path, "wb") as out_f:
        # Write placeholder header (will update later)
        out_f.write(struct.pack("Q", 0x544F4B454E53))  # "TOKENS" magic
        out_f.write(struct.pack("Q", 0))  # Placeholder for length

        token_buffer = []
        docs_processed = 0

        for file_path in files:
            logger.info(f"Tokenizing: {file_path}")

            with open(file_path, "r", encoding="utf-8", errors="ignore") as in_f:
                current_doc = []

                for line in in_f:
                    # Documents are separated by blank lines
                    if line.strip() == "":
                        if current_doc:
                            # Process completed document
                            doc_text = "".join(current_doc)
                            tokens = tokenizer.encode(doc_text, add_eos=add_eos)
                            token_buffer.extend(tokens)
                            current_doc = []
                            docs_processed += 1

                            # Write buffer to disk periodically
                            if docs_processed % chunk_size == 0:
                                if token_buffer:
                                    arr = np.array(token_buffer, dtype=np.uint16)
                                    out_f.write(arr.tobytes())
                                    total_tokens += len(token_buffer)
                                    token_buffer = []
                                    logger.info(f"  Processed {docs_processed:,} docs, {total_tokens:,} tokens")
                    else:
                        current_doc.append(line)

                # Process last document in file
                if current_doc:
                    doc_text = "".join(current_doc)
                    tokens = tokenizer.encode(doc_text, add_eos=add_eos)
                    token_buffer.extend(tokens)
                    docs_processed += 1

        # Write remaining buffer
        if token_buffer:
            arr = np.array(token_buffer, dtype=np.uint16)
            out_f.write(arr.tobytes())
            total_tokens += len(token_buffer)

    logger.info(f"Total: {docs_processed:,} docs, {total_tokens:,} tokens")

    # Update header with actual token count
    with open(temp_path, "r+b") as f:
        f.seek(8)  # Skip magic number
        f.write(struct.pack("Q", total_tokens))

    # Rename temp to final
    os.replace(temp_path, output_path)

    logger.info(f"Saved to: {output_path}")
    return total_tokens


def split_dataset(
    input_path: str,
    train_path: str,
    val_path: str,
    val_ratio: float = 0.05,
    seed: int = 42,
    chunk_size: int = 1000000,
):
    """Split a tokenized dataset into train and validation sets using streaming.

    Args:
        input_path: Path to input .bin file
        train_path: Path to output train .bin file
        val_path: Path to output validation .bin file
        val_ratio: Ratio of data for validation
        seed: Random seed
        chunk_size: Number of tokens to process at a time
    """
    dataset = TokenizedDataset(input_path)
    total_tokens = len(dataset)

    # Calculate split point
    val_size = int(total_tokens * val_ratio)
    train_size = total_tokens - val_size

    logger.info(f"Splitting: {train_size:,} train, {val_size:,} val")

    # Write train set with streaming
    with open(train_path, "wb") as f:
        f.write(struct.pack("Q", 0x544F4B454E53))
        f.write(struct.pack("Q", train_size))

        written = 0
        while written < train_size:
            chunk_len = min(chunk_size, train_size - written)
            chunk = dataset.get_chunk(written, chunk_len)
            f.write(chunk.astype(np.uint16).tobytes())
            written += chunk_len
            if written % (chunk_size * 10) == 0:
                logger.info(f"  Train: {written:,} / {train_size:,} tokens")

    logger.info(f"Saved train to: {train_path}")

    # Write val set with streaming
    with open(val_path, "wb") as f:
        f.write(struct.pack("Q", 0x544F4B454E53))
        f.write(struct.pack("Q", val_size))

        written = 0
        while written < val_size:
            chunk_len = min(chunk_size, val_size - written)
            chunk = dataset.get_chunk(train_size + written, chunk_len)
            f.write(chunk.astype(np.uint16).tobytes())
            written += chunk_len

    logger.info(f"Saved val to: {val_path}")

    dataset.close()


def create_dataloaders(
    train_path: str,
    val_path: str,
    context_length: int,
    batch_size: int,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation data loaders.

    Args:
        train_path: Path to train .bin file
        val_path: Path to val .bin file
        context_length: Sequence length
        batch_size: Batch size
        seed: Random seed

    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = TokenizedDataset(train_path)
    val_dataset = TokenizedDataset(val_path)

    train_loader = DataLoader(
        train_dataset,
        context_length=context_length,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )

    val_loader = DataLoader(
        val_dataset,
        context_length=context_length,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
    )

    return train_loader, val_loader


def estimate_dataset_stats(path: str, context_length: int) -> dict:
    """Estimate dataset statistics.

    Args:
        path: Path to .bin file
        context_length: Training context length

    Returns:
        Dictionary of statistics
    """
    dataset = TokenizedDataset(path)
    total_tokens = len(dataset)
    seq_length = context_length + 1
    num_sequences = total_tokens // seq_length

    stats = {
        "total_tokens": total_tokens,
        "num_sequences": num_sequences,
        "file_size_mb": os.path.getsize(path) / (1024 * 1024),
        "estimated_epochs_per_step": {
            "batch_1": 1 / num_sequences if num_sequences > 0 else 0,
            "batch_4": 4 / num_sequences if num_sequences > 0 else 0,
            "batch_8": 8 / num_sequences if num_sequences > 0 else 0,
        },
    }

    dataset.close()
    return stats


def main():
    """CLI for data preparation."""
    parser = argparse.ArgumentParser(description="Prepare data for language model training")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Prepare command
    prep_parser = subparsers.add_parser("prepare", help="Tokenize and prepare dataset")
    prep_parser.add_argument("--input", "-i", type=str, required=True, help="Input text file or directory")
    prep_parser.add_argument("--output", "-o", type=str, required=True, help="Output .bin file")
    prep_parser.add_argument("--tokenizer", "-t", type=str, required=True, help="Path to tokenizer model")
    prep_parser.add_argument("--max-files", type=int, default=None, help="Max files to process")

    # Split command
    split_parser = subparsers.add_parser("split", help="Split dataset into train/val")
    split_parser.add_argument("--input", "-i", type=str, required=True, help="Input .bin file")
    split_parser.add_argument("--train-output", type=str, required=True, help="Output train .bin file")
    split_parser.add_argument("--val-output", type=str, required=True, help="Output val .bin file")
    split_parser.add_argument("--val-ratio", type=float, default=0.05, help="Validation ratio")

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show dataset statistics")
    stats_parser.add_argument("--input", "-i", type=str, required=True, help="Input .bin file")
    stats_parser.add_argument("--context-length", type=int, default=2048, help="Context length")

    # All-in-one command
    all_parser = subparsers.add_parser("all", help="Prepare, tokenize, and split dataset")
    all_parser.add_argument("--raw-data", type=str, required=True, help="Raw text directory")
    all_parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    all_parser.add_argument("--vocab-size", type=int, default=32000, help="Tokenizer vocab size")
    all_parser.add_argument("--val-ratio", type=float, default=0.05, help="Validation ratio")

    args = parser.parse_args()

    if args.command == "prepare":
        tokenizer = Tokenizer(args.tokenizer)
        prepare_dataset(args.input, args.output, tokenizer, max_files=args.max_files)

    elif args.command == "split":
        split_dataset(args.input, args.train_output, args.val_output, args.val_ratio)

    elif args.command == "stats":
        stats = estimate_dataset_stats(args.input, args.context_length)
        print(json.dumps(stats, indent=2))

    elif args.command == "all":
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Train tokenizer
        logger.info("Training tokenizer...")
        raw_files = list(Path(args.raw_data).glob("**/*.txt"))
        if not raw_files:
            raise ValueError(f"No .txt files found in {args.raw_data}")

        tokenizer_path = train_tokenizer(
            input_files=[str(f) for f in raw_files],
            output_path=str(output_dir / "tokenizer"),
            vocab_size=args.vocab_size,
        )
        tokenizer = Tokenizer(tokenizer_path)

        # Prepare dataset
        logger.info("Preparing dataset...")
        full_data_path = str(output_dir / "full.bin")
        prepare_dataset(args.raw_data, full_data_path, tokenizer)

        # Split dataset
        logger.info("Splitting dataset...")
        split_dataset(
            full_data_path,
            str(output_dir / "train.bin"),
            str(output_dir / "val.bin"),
            args.val_ratio,
        )

        # Show stats
        logger.info("Dataset statistics:")
        for name, path in [("train", output_dir / "train.bin"), ("val", output_dir / "val.bin")]:
            stats = estimate_dataset_stats(str(path), 2048)
            logger.info(f"{name}: {stats['total_tokens']:,} tokens, {stats['file_size_mb']:.1f} MB")

        # Clean up full dataset
        os.remove(full_data_path)
        logger.info("Done!")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
