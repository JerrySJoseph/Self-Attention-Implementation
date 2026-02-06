"""
Text Generation Interface

Provides text generation capabilities with various sampling strategies:
- Temperature scaling
- Top-p (nucleus) sampling
- Top-k sampling
- Repetition penalty (vectorized)
- Interactive chat mode
- Batch generation
- Streaming output
- Rolling KV cache for long sequences

Optimizations:
- Vectorized repetition penalty
- Rolling KV cache instead of cache discard
- FP32 sampling for numerical stability
- Efficient top-k/top-p filtering
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Callable, Generator as GenType, List, Optional, Tuple, Set

import mlx.core as mx
import mlx.nn as nn

try:
    from .model import ModelConfig, TransformerModel
    from .tokenizer import Tokenizer
except ImportError:
    from model import ModelConfig, TransformerModel
    from tokenizer import Tokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def apply_repetition_penalty(
    logits: mx.array,
    generated_ids: List[int],
    penalty: float = 1.1,
) -> mx.array:
    """Apply repetition penalty to logits (vectorized).

    Args:
        logits: Logits array of shape (batch, vocab_size)
        generated_ids: List of previously generated token IDs
        penalty: Penalty factor (>1 discourages repetition)

    Returns:
        Modified logits
    """
    if penalty == 1.0 or not generated_ids:
        return logits

    # Get unique token IDs
    unique_ids = list(set(generated_ids))

    # Create penalty mask
    penalty_mask = mx.zeros(logits.shape[-1])
    for token_id in unique_ids:
        penalty_mask = mx.where(
            mx.arange(logits.shape[-1]) == token_id,
            mx.array(penalty),
            penalty_mask
        )

    # Apply penalty: divide positive logits, multiply negative logits
    penalty_mask = mx.where(penalty_mask == 0, mx.array(1.0), penalty_mask)
    logits = mx.where(logits > 0, logits / penalty_mask, logits * penalty_mask)

    return logits


def sample_token(
    logits: mx.array,
    temperature: float = 0.8,
    top_p: float = 0.95,
    top_k: int = 0,
) -> mx.array:
    """Sample next token from logits with various strategies.

    Args:
        logits: Logits array of shape (batch, vocab_size)
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold
        top_k: Top-k sampling (0 = disabled)

    Returns:
        Sampled token ID
    """
    # Convert to float32 for numerical stability
    logits = logits.astype(mx.float32)

    # Apply temperature
    if temperature > 0 and temperature != 1.0:
        logits = logits / temperature
    elif temperature <= 0:
        # Greedy decoding
        return mx.argmax(logits, axis=-1)

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

        # Create mask for tokens to remove (keep at least one)
        sorted_mask = cumulative_probs > top_p
        sorted_mask = mx.concatenate([
            mx.zeros((sorted_mask.shape[0], 1), dtype=mx.bool_),
            sorted_mask[:, :-1]
        ], axis=-1)

        # Scatter mask back to original order
        mask = mx.zeros_like(logits, dtype=mx.bool_)
        mask = mx.put_along_axis(mask, sorted_indices, sorted_mask, axis=-1)
        logits = mx.where(mask, float("-inf"), logits)

    # Sample from distribution
    probs = mx.softmax(logits, axis=-1)
    return mx.random.categorical(probs)


def roll_kv_cache(cache: List[Tuple[mx.array, mx.array]], keep_length: int) -> List[Tuple[mx.array, mx.array]]:
    """Roll KV cache to keep only the most recent tokens.

    Args:
        cache: List of (k, v) tuples for each layer
        keep_length: Number of tokens to keep

    Returns:
        Truncated cache
    """
    new_cache = []
    for k, v in cache:
        # k and v have shape (batch, seq_len, num_heads, head_dim)
        new_cache.append((k[:, -keep_length:], v[:, -keep_length:]))
    return new_cache


class TextGenerator:
    """Text generation interface for the transformer model."""

    def __init__(
        self,
        model: TransformerModel,
        tokenizer: Tokenizer,
        config: ModelConfig,
    ):
        """Initialize generator.

        Args:
            model: Trained transformer model
            tokenizer: Tokenizer
            config: Model configuration
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, tokenizer_path: str) -> "TextGenerator":
        """Load generator from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint directory
            tokenizer_path: Path to tokenizer model

        Returns:
            Initialized TextGenerator
        """
        checkpoint_path = Path(checkpoint_path)

        # Load config
        config_path = checkpoint_path / "config.json"
        with open(config_path, "r") as f:
            config_dict = json.load(f)

        model_config = ModelConfig.from_dict(config_dict["model"])

        # Load model
        model = TransformerModel(model_config)
        weights_path = checkpoint_path / "model.safetensors"
        weights = mx.load(str(weights_path))
        model.load_weights(list(weights.items()))
        mx.eval(model.parameters())

        # Load tokenizer
        tokenizer = Tokenizer(tokenizer_path)

        logger.info(f"Loaded model from {checkpoint_path}")
        logger.info(f"Model parameters: {model.count_parameters():,}")

        return cls(model, tokenizer, model_config)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
        stop_tokens: Optional[List[str]] = None,
    ) -> str:
        """Generate text from prompt.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling probability threshold
            top_k: Top-k sampling (0 = disabled)
            repetition_penalty: Penalty for repeating tokens (>1 = discourage)
            stop_tokens: List of tokens to stop generation at

        Returns:
            Generated text (including prompt)
        """
        generated_tokens = list(self.generate_stream(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            stop_tokens=stop_tokens,
        ))

        return prompt + "".join(generated_tokens)

    def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
        stop_tokens: Optional[List[str]] = None,
    ) -> GenType[str, None, None]:
        """Generate text with streaming output.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            top_k: Top-k sampling (0 = disabled)
            repetition_penalty: Penalty for repeating tokens
            stop_tokens: List of tokens to stop at

        Yields:
            Generated tokens one at a time
        """
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt)

        # Truncate if necessary
        max_prompt_len = self.config.context_length - 1
        if len(input_ids) > max_prompt_len:
            input_ids = input_ids[-max_prompt_len:]
            logger.warning(f"Prompt truncated to {len(input_ids)} tokens")

        input_ids = mx.array([input_ids])
        generated_ids: List[int] = []

        # Convert stop tokens to IDs
        stop_ids: Set[int] = {self.tokenizer.eos_id}
        if stop_tokens:
            for token in stop_tokens:
                token_id = self.tokenizer.get_id(token)
                if token_id is not None:
                    stop_ids.add(token_id)

        # Generate with KV cache
        cache = None
        current_pos = 0

        for _ in range(max_tokens):
            # Forward pass
            if cache is None:
                logits, cache = self.model(input_ids)
                current_pos = input_ids.shape[1]
            else:
                # Only process the last token
                last_token = input_ids[:, -1:]
                logits, cache = self.model(last_token, cache=cache)
                current_pos += 1

            # Get logits for last position
            logits = logits[:, -1, :]

            # Apply repetition penalty
            if repetition_penalty != 1.0 and generated_ids:
                logits = apply_repetition_penalty(logits, generated_ids, repetition_penalty)

            # Sample next token
            next_token = sample_token(logits, temperature, top_p, top_k)
            mx.eval(next_token)

            next_token_id = int(next_token[0])

            # Check for stop tokens
            if next_token_id in stop_ids:
                break

            # Decode and yield
            token_str = self.tokenizer.decode([next_token_id], skip_special_tokens=False)
            yield token_str

            # Update for next iteration
            generated_ids.append(next_token_id)
            input_ids = mx.concatenate([input_ids, next_token.reshape(1, 1)], axis=1)

            # Roll cache if context is getting too long
            if current_pos >= self.config.context_length - 1:
                # Keep most recent tokens, roll cache instead of discarding
                keep_len = self.config.context_length - self.config.context_length // 4
                cache = roll_kv_cache(cache, keep_len)
                input_ids = input_ids[:, -keep_len:]
                current_pos = keep_len
                logger.debug(f"Rolled cache to {keep_len} tokens")

    def generate_batch(
        self,
        prompts: List[str],
        max_tokens: int = 100,
        temperature: float = 0.8,
        top_p: float = 0.95,
        **kwargs,
    ) -> List[str]:
        """Generate text for multiple prompts.

        Note: Currently processes sequentially. For true batch generation,
        prompts would need to be padded to same length.

        Args:
            prompts: List of input prompts
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            **kwargs: Additional generation arguments

        Returns:
            List of generated texts
        """
        results = []
        for i, prompt in enumerate(prompts):
            logger.info(f"Generating {i+1}/{len(prompts)}...")
            result = self.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                **kwargs,
            )
            results.append(result)
        return results

    def chat(
        self,
        messages: List[dict],
        max_tokens: int = 500,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stream: bool = False,
        **kwargs,
    ):
        """Generate response for chat messages.

        Args:
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            stream: Whether to stream output
            **kwargs: Additional generation arguments

        Returns:
            Generated response (string or generator if streaming)
        """
        # Encode chat using tokenizer's chat template
        try:
            input_ids = self.tokenizer.encode_chat(messages, add_generation_prompt=True)
            prompt = self.tokenizer.decode(input_ids, skip_special_tokens=False)
        except Exception:
            # Fallback: simple concatenation
            prompt = ""
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                prompt += f"<|{role}|>\n{content}\n"
            prompt += "<|assistant|>\n"

        if stream:
            return self.generate_stream(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop_tokens=["<|user|>", "<|system|>", "<|end|>"],
                **kwargs,
            )
        else:
            full_response = self.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop_tokens=["<|user|>", "<|system|>", "<|end|>"],
                **kwargs,
            )
            # Extract just the assistant response
            return full_response[len(prompt):]


def interactive_chat(generator: TextGenerator):
    """Run interactive chat session.

    Args:
        generator: Initialized TextGenerator
    """
    print("\n" + "=" * 50)
    print("Interactive Chat Mode")
    print("Type 'quit' or 'exit' to end the session")
    print("Type 'clear' to start a new conversation")
    print("=" * 50 + "\n")

    messages = []

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["quit", "exit"]:
                print("Goodbye!")
                break

            if user_input.lower() == "clear":
                messages = []
                print("\n[Conversation cleared]\n")
                continue

            # Add user message
            messages.append({"role": "user", "content": user_input})

            # Generate response with streaming
            print("Assistant: ", end="", flush=True)

            response_tokens = []
            for token in generator.chat(messages, stream=True):
                print(token, end="", flush=True)
                response_tokens.append(token)

            response = "".join(response_tokens)
            print()  # Newline after response

            # Add assistant message to history
            messages.append({"role": "assistant", "content": response})

        except KeyboardInterrupt:
            print("\n\nInterrupted. Type 'quit' to exit.")
        except Exception as e:
            logger.error(f"Error: {e}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Generate text with trained model")
    parser.add_argument("--checkpoint", "-c", type=str, required=True, help="Path to checkpoint directory")
    parser.add_argument("--tokenizer", "-t", type=str, required=True, help="Path to tokenizer model")

    # Generation mode
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--interactive", action="store_true", help="Interactive chat mode")
    mode_group.add_argument("--prompt", "-p", type=str, help="Single prompt to generate from")
    mode_group.add_argument("--prompts-file", type=str, help="File with prompts (one per line)")

    # Generation parameters
    parser.add_argument("--max-tokens", type=int, default=200, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.95, help="Nucleus sampling probability")
    parser.add_argument("--top-k", type=int, default=0, help="Top-k sampling (0 = disabled)")
    parser.add_argument("--repetition-penalty", type=float, default=1.1, help="Repetition penalty")

    # Output
    parser.add_argument("--output", "-o", type=str, help="Output file for batch generation")
    parser.add_argument("--stream", action="store_true", help="Stream output for single prompt")

    args = parser.parse_args()

    # Load model
    logger.info("Loading model...")
    generator = TextGenerator.from_checkpoint(args.checkpoint, args.tokenizer)

    if args.interactive:
        interactive_chat(generator)

    elif args.prompt:
        if args.stream:
            print(args.prompt, end="", flush=True)
            for token in generator.generate_stream(
                prompt=args.prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
            ):
                print(token, end="", flush=True)
            print()
        else:
            start_time = time.time()
            result = generator.generate(
                prompt=args.prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
            )
            elapsed = time.time() - start_time

            print(result)
            logger.info(f"Generated in {elapsed:.2f}s")

    elif args.prompts_file:
        with open(args.prompts_file, "r") as f:
            prompts = [line.strip() for line in f if line.strip()]

        logger.info(f"Generating for {len(prompts)} prompts...")

        results = generator.generate_batch(
            prompts=prompts,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
        )

        if args.output:
            with open(args.output, "w") as f:
                for result in results:
                    f.write(result + "\n\n---\n\n")
            logger.info(f"Results saved to {args.output}")
        else:
            for i, result in enumerate(results):
                print(f"\n--- Prompt {i + 1} ---")
                print(result)


if __name__ == "__main__":
    main()
