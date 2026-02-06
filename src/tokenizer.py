"""
Tokenizer Training and Loading

Implements SentencePiece BPE tokenizer training and loading
with special token support for language model training.
"""

import os
import argparse
import logging
from pathlib import Path
from typing import List, Optional, Union
import tempfile

import sentencepiece as spm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Special tokens for the language model
SPECIAL_TOKENS = {
    "pad": "<|pad|>",
    "unk": "<|unk|>",
    "bos": "<|startoftext|>",
    "eos": "<|endoftext|>",
    "user": "<|user|>",
    "assistant": "<|assistant|>",
    "system": "<|system|>",
}


class Tokenizer:
    """SentencePiece-based tokenizer with special token support."""

    def __init__(self, model_path: str):
        """Load tokenizer from trained model.

        Args:
            model_path: Path to .model file
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Tokenizer model not found: {model_path}")

        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)
        self.model_path = model_path

        # Cache special token IDs
        self.pad_id = self.sp.PieceToId(SPECIAL_TOKENS["pad"])
        self.unk_id = self.sp.PieceToId(SPECIAL_TOKENS["unk"])
        self.bos_id = self.sp.PieceToId(SPECIAL_TOKENS["bos"])
        self.eos_id = self.sp.PieceToId(SPECIAL_TOKENS["eos"])

        # Chat tokens
        self.user_id = self.sp.PieceToId(SPECIAL_TOKENS["user"])
        self.assistant_id = self.sp.PieceToId(SPECIAL_TOKENS["assistant"])
        self.system_id = self.sp.PieceToId(SPECIAL_TOKENS["system"])

        # Cache special token IDs set for efficient filtering
        self._special_ids = {
            self.pad_id, self.bos_id, self.eos_id,
            self.user_id, self.assistant_id, self.system_id
        }

        logger.info(f"Loaded tokenizer with vocab size: {self.vocab_size}")

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.sp.GetPieceSize()

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[int]:
        """Encode text to token IDs.

        Args:
            text: Input text
            add_bos: Add beginning of sequence token
            add_eos: Add end of sequence token

        Returns:
            List of token IDs
        """
        ids = self.sp.EncodeAsIds(text)

        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]

        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text.

        Args:
            ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens

        Returns:
            Decoded text
        """
        if skip_special_tokens:
            # Filter out special token IDs using cached set
            ids = [i for i in ids if i not in self._special_ids]

        return self.sp.DecodeIds(ids)

    def encode_batch(
        self,
        texts: List[str],
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[List[int]]:
        """Encode multiple texts.

        Args:
            texts: List of input texts
            add_bos: Add beginning of sequence token
            add_eos: Add end of sequence token

        Returns:
            List of token ID lists
        """
        return [self.encode(t, add_bos=add_bos, add_eos=add_eos) for t in texts]

    def decode_batch(self, ids_batch: List[List[int]], skip_special_tokens: bool = True) -> List[str]:
        """Decode multiple token ID lists.

        Args:
            ids_batch: List of token ID lists
            skip_special_tokens: Whether to skip special tokens

        Returns:
            List of decoded texts
        """
        return [self.decode(ids, skip_special_tokens=skip_special_tokens) for ids in ids_batch]

    def encode_chat(
        self,
        messages: List[dict],
        add_generation_prompt: bool = True,
    ) -> List[int]:
        """Encode chat messages into token IDs.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            add_generation_prompt: Add assistant prompt at end

        Returns:
            List of token IDs
        """
        ids = [self.bos_id]

        for message in messages:
            role = message["role"]
            content = message["content"]

            if role == "system":
                ids.append(self.system_id)
            elif role == "user":
                ids.append(self.user_id)
            elif role == "assistant":
                ids.append(self.assistant_id)

            ids.extend(self.sp.EncodeAsIds(content))

        if add_generation_prompt:
            ids.append(self.assistant_id)

        return ids

    def get_token(self, id: int) -> str:
        """Get token string for ID."""
        return self.sp.IdToPiece(id)

    def get_id(self, token: str) -> int:
        """Get ID for token string."""
        return self.sp.PieceToId(token)


def train_tokenizer(
    input_files: Union[str, List[str]],
    output_path: str,
    vocab_size: int = 32000,
    model_type: str = "bpe",
    character_coverage: float = 0.9995,
    num_threads: int = 4,
    max_sentence_length: int = 16384,
) -> str:
    """Train a SentencePiece tokenizer.

    Args:
        input_files: Input text file(s) for training
        output_path: Output path for the model (without extension)
        vocab_size: Vocabulary size
        model_type: Model type (bpe, unigram, char, word)
        character_coverage: Character coverage for training
        num_threads: Number of threads for training
        max_sentence_length: Maximum sentence length

    Returns:
        Path to trained model
    """
    if isinstance(input_files, str):
        input_files = [input_files]

    # Verify files exist
    for f in input_files:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Training file not found: {f}")

    # Create output directory
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    # Build special tokens string
    user_defined_symbols = ",".join([
        SPECIAL_TOKENS["user"],
        SPECIAL_TOKENS["assistant"],
        SPECIAL_TOKENS["system"],
    ])

    logger.info(f"Training tokenizer with vocab_size={vocab_size}")
    logger.info(f"Input files: {input_files}")

    # Train the model
    spm.SentencePieceTrainer.Train(
        input=",".join(input_files),
        model_prefix=output_path,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=character_coverage,
        num_threads=num_threads,
        max_sentence_length=max_sentence_length,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece=SPECIAL_TOKENS["pad"],
        unk_piece=SPECIAL_TOKENS["unk"],
        bos_piece=SPECIAL_TOKENS["bos"],
        eos_piece=SPECIAL_TOKENS["eos"],
        user_defined_symbols=user_defined_symbols,
        byte_fallback=True,  # Handle unknown characters
        split_by_whitespace=True,
        split_digits=True,
        normalization_rule_name="identity",  # Don't normalize text
    )

    model_path = f"{output_path}.model"
    logger.info(f"Tokenizer saved to: {model_path}")

    return model_path


def train_from_text(
    text: str,
    output_path: str,
    vocab_size: int = 32000,
    **kwargs,
) -> str:
    """Train tokenizer from raw text string.

    Args:
        text: Raw text for training
        output_path: Output path for the model
        vocab_size: Vocabulary size
        **kwargs: Additional arguments for train_tokenizer

    Returns:
        Path to trained model
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(text)
        temp_path = f.name

    try:
        return train_tokenizer(temp_path, output_path, vocab_size=vocab_size, **kwargs)
    finally:
        os.unlink(temp_path)


def load_tokenizer(model_path: str) -> Tokenizer:
    """Load a trained tokenizer.

    Args:
        model_path: Path to .model file

    Returns:
        Loaded Tokenizer
    """
    return Tokenizer(model_path)


def main():
    """CLI for tokenizer training."""
    parser = argparse.ArgumentParser(description="Train or use SentencePiece tokenizer")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a new tokenizer")
    train_parser.add_argument("--input", "-i", type=str, required=True, help="Input text file(s), comma-separated")
    train_parser.add_argument("--output", "-o", type=str, default="tokenizer", help="Output path (without extension)")
    train_parser.add_argument("--vocab-size", type=int, default=32000, help="Vocabulary size")
    train_parser.add_argument("--model-type", type=str, default="bpe", choices=["bpe", "unigram"], help="Model type")
    train_parser.add_argument("--threads", type=int, default=4, help="Number of threads")

    # Encode command
    encode_parser = subparsers.add_parser("encode", help="Encode text")
    encode_parser.add_argument("--model", "-m", type=str, required=True, help="Path to tokenizer model")
    encode_parser.add_argument("--text", "-t", type=str, required=True, help="Text to encode")
    encode_parser.add_argument("--add-special", action="store_true", help="Add BOS/EOS tokens")

    # Decode command
    decode_parser = subparsers.add_parser("decode", help="Decode token IDs")
    decode_parser.add_argument("--model", "-m", type=str, required=True, help="Path to tokenizer model")
    decode_parser.add_argument("--ids", type=str, required=True, help="Comma-separated token IDs")

    # Info command
    info_parser = subparsers.add_parser("info", help="Show tokenizer info")
    info_parser.add_argument("--model", "-m", type=str, required=True, help="Path to tokenizer model")

    args = parser.parse_args()

    if args.command == "train":
        input_files = args.input.split(",")
        model_path = train_tokenizer(
            input_files=input_files,
            output_path=args.output,
            vocab_size=args.vocab_size,
            model_type=args.model_type,
            num_threads=args.threads,
        )
        print(f"Trained tokenizer saved to: {model_path}")

    elif args.command == "encode":
        tokenizer = load_tokenizer(args.model)
        ids = tokenizer.encode(args.text, add_bos=args.add_special, add_eos=args.add_special)
        print(f"Token IDs: {ids}")
        print(f"Tokens: {[tokenizer.get_token(i) for i in ids]}")
        print(f"Length: {len(ids)}")

    elif args.command == "decode":
        tokenizer = load_tokenizer(args.model)
        ids = [int(x.strip()) for x in args.ids.split(",")]
        text = tokenizer.decode(ids)
        print(f"Decoded text: {text}")

    elif args.command == "info":
        tokenizer = load_tokenizer(args.model)
        print(f"Vocab size: {tokenizer.vocab_size}")
        print(f"PAD ID: {tokenizer.pad_id}")
        print(f"UNK ID: {tokenizer.unk_id}")
        print(f"BOS ID: {tokenizer.bos_id}")
        print(f"EOS ID: {tokenizer.eos_id}")
        print(f"User ID: {tokenizer.user_id}")
        print(f"Assistant ID: {tokenizer.assistant_id}")
        print(f"System ID: {tokenizer.system_id}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
