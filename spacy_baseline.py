"""
SpaCy Baseline for PII Detection
This script runs a pretrained SpaCy model (en_core_web_sm) on test data
and evaluates how well it detects PII entities.

SpaCy finds standard entities like PERSON, ORG, GPE, LOC, DATE, etc.
This gives us a baseline to compare against our fine-tuned BERT model.
"""

import spacy
import pandas as pd
import ast
import json
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

def parse_span_labels(x):
    """Parse the span_labels string into a list of tuples."""
    if pd.isna(x):
        return []
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception:
            return []
    return x


def character_spans_to_bio_labels(text, spans):
    """Convert character-level spans to token-level BIO labels (same as in data_preparation.py)."""
    tokens = text.split()
    bio_labels = ["O"] * len(tokens)
    
    char_to_token = {}
    char_pos = 0
    for token_idx, token in enumerate(tokens):
        while char_pos < len(text) and text[char_pos] == ' ':
            char_to_token[char_pos] = -1
            char_pos += 1
        
        for i in range(len(token)):
            char_to_token[char_pos] = token_idx
            char_pos += 1
    
    for start_char, end_char, entity_type in spans:
        first_token = True
        for char_idx in range(start_char, end_char):
            if char_idx in char_to_token:
                token_idx = char_to_token[char_idx]
                if token_idx != -1:
                    if first_token:
                        bio_labels[token_idx] = f"B-{entity_type}"
                        first_token = False
                    else:
                        bio_labels[token_idx] = f"I-{entity_type}"
    
    return tokens, bio_labels


def extract_spacy_bio_labels(doc, tokens):
    """
    Extract BIO labels from a SpaCy doc object.
    SpaCy gives us entities, so we convert them to BIO format.
    
    Args:
        doc: SpaCy doc object with entities already extracted
        tokens: List of tokens (for length alignment)
    
    Returns:
        bio_labels: List of BIO labels
    """
    bio_labels = ["O"] * len(tokens)
    
    # Build character-to-token mapping
    char_to_token = {}
    char_pos = 0
    for token_idx, token in enumerate(tokens):
        while char_pos < len(doc.text) and doc.text[char_pos] == ' ':
            char_pos += 1
        
        for i in range(len(token)):
            if char_pos < len(doc.text):
                char_to_token[char_pos] = token_idx
                char_pos += 1
    
    # Convert SpaCy entities to BIO labels
    for ent in doc.ents:
        # Map the entity span to token indices
        first_token = True
        for char_idx in range(ent.start_char, ent.end_char):
            if char_idx in char_to_token:
                token_idx = char_to_token[char_idx]
                if first_token:
                    bio_labels[token_idx] = f"B-{ent.label_}"
                    first_token = False
                else:
                    bio_labels[token_idx] = f"I-{ent.label_}"
    
    return bio_labels


def evaluate_spacy_model(nlp, csv_file, output_file=None):
    """
    Run SpaCy on test data and evaluate against ground truth.
    
    Args:
        nlp: SpaCy model
        csv_file: Path to test CSV
        output_file: Path to save predictions (optional)
    
    Returns:
        metrics: Dictionary with precision, recall, F1, accuracy
        predictions: List of predictions for each example
    """
    
    print(f"Loading test data from {csv_file}...")
    df = pd.read_csv(csv_file)
    if "text" not in df.columns and "source_text" in df.columns:
        df["text"] = df["source_text"]
    df["span_labels_parsed"] = df["span_labels"].apply(parse_span_labels)
    
    all_true_labels = []
    all_pred_labels = []
    predictions = []
    
    print("Running SpaCy on test set...")
    for idx, row in df.iterrows():
        text = row.get("text", row.get("source_text", ""))
        true_spans = row.get("span_labels_parsed", [])
        
        if not text or len(text) == 0:
            continue
        
        # Get ground truth labels
        _, true_bio = character_spans_to_bio_labels(text, true_spans)
        
        # Get SpaCy predictions
        doc = nlp(text)
        tokens = text.split()
        
        # Make sure tokens match
        if len(tokens) != len(true_bio):
            continue
        
        pred_bio = extract_spacy_bio_labels(doc, tokens)
        
        # Collect for metrics
        all_true_labels.extend(true_bio)
        all_pred_labels.extend(pred_bio)
        
        predictions.append({
            "id": row.get("id", ""),
            "text": text,
            "tokens": tokens,
            "true_labels": true_bio,
            "pred_labels": pred_bio,
            "true_entities": true_spans,
            "pred_entities": [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
        })
        
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1} examples...")
    
    # Calculate metrics
    # Convert labels to numeric for sklearn
    label_set = set(all_true_labels + all_pred_labels)
    label_to_id = {label: i for i, label in enumerate(sorted(label_set))}
    
    true_numeric = [label_to_id[label] for label in all_true_labels]
    pred_numeric = [label_to_id[label] for label in all_pred_labels]
    
    metrics = {
        "accuracy": accuracy_score(true_numeric, pred_numeric),
        "precision": precision_score(true_numeric, pred_numeric, average='weighted', zero_division=0),
        "recall": recall_score(true_numeric, pred_numeric, average='weighted', zero_division=0),
        "f1": f1_score(true_numeric, pred_numeric, average='weighted', zero_division=0),
        "num_examples": len(predictions),
        "num_tokens": len(all_true_labels)
    }
    
    # Save predictions if requested
    if output_file:
        print(f"Saving predictions to {output_file}...")
        with open(output_file, 'w') as f:
            for pred in predictions:
                f.write(json.dumps(pred) + '\n')
    
    return metrics, predictions


if __name__ == "__main__":
    print("=" * 60)
    print("SPACY BASELINE PII DETECTION")
    print("=" * 60)
    
    # Load the pretrained SpaCy model
    print("\nLoading SpaCy model (en_core_web_sm)...")
    print("If this fails, run: python -m spacy download en_core_web_sm")
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("ERROR: SpaCy model not found. Install it with:")
        print("  python -m spacy download en_core_web_sm")
        exit(1)
    
    # Run evaluation on test set
    data_dir = Path("/Users/evwu/Documents/Repositories/ml_project_PI_redaction")
    
    print("\n" + "=" * 60)
    print("EVALUATING ON TEST SET")
    print("=" * 60)
    
    metrics, predictions = evaluate_spacy_model(
        nlp,
        data_dir / "group_testing.csv",
        data_dir / "spacy_predictions.jsonl"
    )
    
    print("\nSPACY RESULTS:")
    print("-" * 60)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"Examples:  {metrics['num_examples']}")
    print(f"Tokens:    {metrics['num_tokens']}")
    print("-" * 60)
    
    # Save metrics to file
    with open(data_dir / "spacy_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nMetrics saved to spacy_metrics.json")
    print(f"Predictions saved to spacy_predictions.jsonl")
