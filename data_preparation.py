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

def parse_span_labels(x):
    """
    Convert the span_labels string into a list of tuples.
    Each tuple is (start_char, end_char, entity_type)
    """
    if pd.isna(x):
        return []
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception:
            return []
    return x


def character_spans_to_bio_labels(text, spans):
    """
    Convert character-level spans to token-level BIO labels.
    
    Args:
        text: The raw text
        spans: List of (start_char, end_char, entity_type) tuples
    
    Returns:
        tokens: List of tokens
        bio_labels: List of BIO labels (same length as tokens)
    """
    
    # Split text into words (simple tokenization)
    # In a real project, you'd use a proper tokenizer like BERT's
    tokens = text.split()
    bio_labels = ["O"] * len(tokens)
    
    # Map each character to its token index
    char_to_token = {}
    char_pos = 0
    for token_idx, token in enumerate(tokens):
        # Account for the space between tokens
        while char_pos < len(text) and text[char_pos] == ' ':
            char_to_token[char_pos] = -1  # Space character
            char_pos += 1
        
        # Map characters of this token
        for i in range(len(token)):
            char_to_token[char_pos] = token_idx
            char_pos += 1
    
    # For each span, mark the corresponding tokens
    for start_char, end_char, entity_type in spans:
        first_token = True
        for char_idx in range(start_char, end_char):
            if char_idx in char_to_token:
                token_idx = char_to_token[char_idx]
                if token_idx != -1:  # Not a space
                    if first_token:
                        bio_labels[token_idx] = f"B-{entity_type}"
                        first_token = False
                    else:
                        bio_labels[token_idx] = f"I-{entity_type}"
    
    return tokens, bio_labels


def prepare_dataset(csv_file, output_file):
    """
    Convert a CSV file into BERT-ready format with BIO labels.
    
    Args:
        csv_file: Path to input CSV (e.g., group_training.csv)
        output_file: Path to save the prepared data as JSONL
    """
    
    print(f"Loading {csv_file}...")
    df = pd.read_csv(csv_file)

    # The raw dataset uses source_text, so we normalize it to text here.
    if "text" not in df.columns and "source_text" in df.columns:
        df["text"] = df["source_text"]
    
    # Parse the span labels
    df["span_labels_parsed"] = df["span_labels"].apply(parse_span_labels)
    
    prepared_data = []
    
    for idx, row in df.iterrows():
        text = row.get("text", row.get("source_text", ""))
        spans = row.get("span_labels_parsed", [])
        
        if not text or len(text) == 0:
            continue
        
        # Convert to BIO format
        tokens, bio_labels = character_spans_to_bio_labels(text, spans)
        
        # Only keep examples with at least some tokens
        if len(tokens) > 0:
            prepared_data.append({
                "tokens": tokens,
                "ner_tags": bio_labels,
                "id": row.get("id", ""),
                "language": row.get("language", "")
            })
    
    # Save to JSONL format (one JSON object per line)
    print(f"Saving {len(prepared_data)} examples to {output_file}...")
    with open(output_file, 'w') as f:
        for item in prepared_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"Done! Saved to {output_file}")
    return prepared_data


if __name__ == "__main__":
    # Convert each split
    data_dir = Path("/Users/evwu/Documents/Repositories/ml_project_PI_redaction")
    
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
    
    print("\n" + "=" * 60)
    print("ALL DATA PREPARED!")
    print("=" * 60)
