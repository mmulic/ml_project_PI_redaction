"""
Data Preparation for BERT Training
This script converts the CSV files with character-level PII annotations
into token-level BIO (Begin-Inside-Outside) labels for BERT training.

BIO format: 
  - B-PER = Beginning of a PERSON entity
  - I-PER = Inside a PERSON entity
  - O = Outside any entity (no PII)
"""

import pandas as pd
import ast
import json
from pathlib import Path
from transformers import AutoTokenizer
import time

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "distilbert-base-multilingual-cased",
    use_fast=True
)
print(type(tokenizer))

def parse_span_labels(x):
    if pd.isna(x):
        return []
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception:
            return []
    return x



def tokenize_and_align_labels(text, spans):
    """
    Tokenize text using DistilBERT tokenizer and align BIO labels.

    Args:
        text: raw string
        spans: list of (start_char, end_char, label)

    Returns:
        tokens: list of tokens
        bio_labels: list of BIO labels
    """

    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        padding=False
    )

    offsets = encoding["offset_mapping"]
    input_ids = encoding["input_ids"]

    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    bio_labels = ["O"] * len(tokens)

    for start_char, end_char, label in spans:
        first = True

        for i, (start, end) in enumerate(offsets):
            # Skip special tokens like [CLS], [SEP]
            if start == end == 0:
                continue

            # Check overlap between token and entity span
            if start < end_char and end > start_char:
                if first:
                    bio_labels[i] = f"B-{label}"
                    first = False
                else:
                    bio_labels[i] = f"I-{label}"

    return tokens, bio_labels


def prepare_dataset(csv_file, output_file):
    print(f"Loading {csv_file}...")
    df = pd.read_csv(csv_file)

    if "text" not in df.columns and "source_text" in df.columns:
        df["text"] = df["source_text"]

    df["span_labels_parsed"] = df["span_labels"].apply(parse_span_labels)

    prepared_data = []

    for _, row in df.iterrows():
        text = row.get("text", "")
        spans = row.get("span_labels_parsed", [])

        if not text:
            continue

        tokens, labels = tokenize_and_align_labels(text, spans)

        prepared_data.append({
            "tokens": tokens,
            "ner_tags": labels,
            "id": row.get("id", ""),
            "language": row.get("language", "")
        })

    print(f"Saving {len(prepared_data)} examples to {output_file}...")

    with open(output_file, "w", encoding="utf-8") as f:
        for item in prepared_data:
            f.write(json.dumps(item) + "\n")

    print("Done!")
    return prepared_data


if __name__ == "__main__":
    start_time = time.time()
    data_dir = Path("")

    print("=" * 60)
    print("PREPARING TRAINING DATA")
    print("=" * 60)
    prepare_dataset(
        data_dir / "group_training.csv",
        data_dir / "train_bio.jsonl"
    )

    print("\n" + "=" * 60)
    print("PREPARING VALIDATION DATA")
    print("=" * 60)
    prepare_dataset(
        data_dir / "group_validation.csv",
        data_dir / "val_bio.jsonl"
    )

    print("\n" + "=" * 60)
    print("PREPARING TEST DATA")
    print("=" * 60)
    prepare_dataset(
        data_dir / "group_testing.csv",
        data_dir / "test_bio.jsonl"
    )

    print("\nALL DATA PREPARED!")
    end_time = time.time()
    print(end_time - start_time)
