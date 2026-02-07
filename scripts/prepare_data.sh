#!/bin/bash
# Data Preparation Script
# Downloads and prepares training data for the language model

set -e

# Configuration
DATA_DIR="${DATA_DIR:-./raw_data}"
OUTPUT_DIR="${OUTPUT_DIR:-./data}"
VOCAB_SIZE="${VOCAB_SIZE:-32000}"
DATASET="${DATASET:-tinystories}"  # Options: tinystories, openwebtext_sample

mkdir -p "$DATA_DIR"
mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "Language Model Data Preparation"
echo "=============================================="
echo "Data directory: $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Vocabulary size: $VOCAB_SIZE"
echo "Dataset: $DATASET"
echo "=============================================="

# Function to download TinyStories dataset
download_tinystories() {
    echo "Downloading TinyStories dataset..."

    # TinyStories is a collection of simple stories good for training small LMs
    # Available on HuggingFace
    python3 << 'EOF'
import os
from datasets import load_dataset

data_dir = os.environ.get("DATA_DIR", "./raw_data")
os.makedirs(data_dir, exist_ok=True)

print("Loading TinyStories dataset from HuggingFace...")
dataset = load_dataset("roneneldan/TinyStories", split="train")

# Save to text file
output_path = os.path.join(data_dir, "tinystories.txt")
print(f"Saving to {output_path}...")

with open(output_path, "w", encoding="utf-8") as f:
    for i, example in enumerate(dataset):
        f.write(example["text"] + "\n\n")
        if (i + 1) % 100000 == 0:
            print(f"  Processed {i + 1} examples...")

print(f"Saved {len(dataset)} stories")
EOF
}

# Function to download a sample of OpenWebText
download_openwebtext_sample() {
    echo "Downloading OpenWebText sample..."

    python3 << 'EOF'
import os
from datasets import load_dataset

data_dir = os.environ.get("DATA_DIR", "./raw_data")
os.makedirs(data_dir, exist_ok=True)

print("Loading OpenWebText dataset from HuggingFace...")
# Load a subset to avoid downloading the full dataset
dataset = load_dataset("openwebtext", split="train[:100000]")

# Save to text file
output_path = os.path.join(data_dir, "openwebtext.txt")
print(f"Saving to {output_path}...")

with open(output_path, "w", encoding="utf-8") as f:
    for i, example in enumerate(dataset):
        f.write(example["text"] + "\n\n")
        if (i + 1) % 10000 == 0:
            print(f"  Processed {i + 1} examples...")

print(f"Saved {len(dataset)} documents")
EOF
}

# Download dataset
case "$DATASET" in
    "tinystories")
        download_tinystories
        ;;
    "openwebtext_sample")
        download_openwebtext_sample
        ;;
    *)
        echo "Unknown dataset: $DATASET"
        echo "Available: tinystories, openwebtext_sample"
        exit 1
        ;;
esac

echo ""
echo "Training tokenizer and preparing data..."

# Use the platform-independent data preparation script
cd "$(dirname "$0")/.."
python3 -m src.prepare_data \
    --raw-data "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --vocab-size "$VOCAB_SIZE"

echo ""
echo "=============================================="
echo "Data preparation complete!"
echo "=============================================="
echo ""
echo "Files created:"
ls -lh "$OUTPUT_DIR"
echo ""
echo "To start training, run:"
echo "  python src/train.py --config configs/model_config.yaml"
