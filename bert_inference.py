"""
BERT Inference and Comparison with SpaCy
This script runs the fine-tuned BERT model on test data and compares it
side-by-side with the SpaCy baseline.

The comparison shows which model is better for your PII detection task.
"""

import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def load_jsonl_data(jsonl_file):
    """Load data from JSONL file."""
    data = []
    with open(jsonl_file, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def run_bert_inference(model_dir, test_jsonl, test_csv, output_file):
    """
    Run the fine-tuned BERT model on test data and evaluate.
    
    Args:
        model_dir: Directory containing the saved BERT model
        test_jsonl: Path to prepared test data (BIO format)
        test_csv: Path to original test CSV (for reference)
        output_file: Where to save predictions
    
    Returns:
        metrics: Dictionary with precision, recall, F1, accuracy
        predictions: List of predictions for each example
    """
    
    print("Loading BERT model...")
    model = AutoModelForTokenClassification.from_pretrained(model_dir).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    # Load label mappings
    with open(model_dir / "label2id.json", 'r') as f:
        label2id = json.load(f)
    with open(model_dir / "id2label.json", 'r') as f:
        id2label = json.load(f)
    
    id2label = {int(k): v for k, v in id2label.items()}  # Convert string keys to int
    
    # Load test data
    test_data = load_jsonl_data(test_jsonl)
    test_df = pd.read_csv(test_csv)
    if "text" not in test_df.columns and "source_text" in test_df.columns:
        test_df["text"] = test_df["source_text"]
    
    # Create NER pipeline with BERT
    ner_pipeline = pipeline(
        "token-classification",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
        aggregation_strategy="simple"  # Keep tokens separate
    )
    
    all_true_labels = []
    all_pred_labels = []
    predictions = []
    
    print(f"Running BERT inference on {len(test_data)} examples...")
    
    model.eval()
    with torch.no_grad():
        for idx, item in enumerate(test_data):
            tokens = item["tokens"]
            true_ner_tags = item["ner_tags"]
            text = " ".join(tokens)
            
            # Skip if empty
            if not tokens:
                continue
            
            # Run BERT
            bert_output = ner_pipeline(text)
            
            # Convert BERT output to BIO labels
            # BERT output is at token level, we need to align with our tokens
            pred_ner_tags = ["O"] * len(tokens)
            
            for bert_entity in bert_output:
                word = bert_entity["word"]
                label = bert_entity["entity"]
                # entity format from pipeline: "B-PERSON", "I-PERSON", etc.
                
                # Find which token this corresponds to
                # This is a simplified alignment
                for token_idx, token in enumerate(tokens):
                    if token.lower() in word.lower() or word.lower() in token.lower():
                        if pred_ner_tags[token_idx] == "O":
                            pred_ner_tags[token_idx] = label
                        break
            
            # Collect metrics
            all_true_labels.extend(true_ner_tags)
            all_pred_labels.extend(pred_ner_tags)
            
            predictions.append({
                "tokens": tokens,
                "true_labels": true_ner_tags,
                "pred_labels": pred_ner_tags,
                "text": text,
            })
            
            if (idx + 1) % 50 == 0:
                print(f"  Processed {idx + 1}/{len(test_data)} examples...")
    
    # Calculate metrics
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
    
    # Save predictions
    print(f"Saving predictions to {output_file}...")
    with open(output_file, 'w') as f:
        for pred in predictions:
            f.write(json.dumps(pred) + '\n')
    
    return metrics, predictions


def compare_models(data_dir):
    """
    Load results from both SpaCy and BERT and create a comparison.
    """
    
    print("\nLoading model results...")
    
    # Load metrics
    with open(data_dir / "spacy_metrics.json", 'r') as f:
        spacy_metrics = json.load(f)
    with open(data_dir / "bert_metrics.json", 'r') as f:
        bert_metrics = json.load(f)
    
    # Create comparison table
    print("\n" + "=" * 80)
    print("MODEL COMPARISON: SPACY vs BERT")
    print("=" * 80)
    
    metrics_to_compare = ["accuracy", "precision", "recall", "f1"]
    
    print(f"\n{'Metric':<15} {'SpaCy':<20} {'BERT':<20} {'Winner':<15}")
    print("-" * 80)
    
    for metric in metrics_to_compare:
        spacy_val = spacy_metrics.get(metric, 0)
        bert_val = bert_metrics.get(metric, 0)
        
        if bert_val > spacy_val:
            winner = "BERT ✓"
        elif spacy_val > bert_val:
            winner = "SpaCy ✓"
        else:
            winner = "Tie"
        
        print(f"{metric:<15} {spacy_val:<20.4f} {bert_val:<20.4f} {winner:<15}")
    
    print("-" * 80)
    print(f"{'Examples':<15} {spacy_metrics.get('num_examples', 0):<20} {bert_metrics.get('num_examples', 0):<20}")
    print(f"{'Tokens':<15} {spacy_metrics.get('num_tokens', 0):<20} {bert_metrics.get('num_tokens', 0):<20}")
    print("=" * 80)
    
    # Summary
    print("\nSUMMARY:")
    print("-" * 80)
    
    bert_wins = sum(1 for metric in metrics_to_compare if bert_metrics.get(metric, 0) > spacy_metrics.get(metric, 0))
    spacy_wins = sum(1 for metric in metrics_to_compare if spacy_metrics.get(metric, 0) > bert_metrics.get(metric, 0))
    
    if bert_wins > spacy_wins:
        print(f"🎯 BERT is the winner ({bert_wins} out of {len(metrics_to_compare)} metrics)")
    elif spacy_wins > bert_wins:
        print(f"🎯 SpaCy is the winner ({spacy_wins} out of {len(metrics_to_compare)} metrics)")
    else:
        print("🎯 It's a tie! Both models perform similarly")
    
    print("-" * 80)
    
    # Recommendations
    print("\nRECOMMENDATIONS:")
    if bert_metrics["f1"] > spacy_metrics["f1"]:
        print("✓ Use BERT for production PII detection")
    else:
        print("✓ Use SpaCy (faster and simpler)")
    print("✓ Consider ensemble: run both and flag when they disagree")
    
    return {
        "spacy": spacy_metrics,
        "bert": bert_metrics,
        "winner": "bert" if bert_wins > spacy_wins else "spacy" if spacy_wins > bert_wins else "tie"
    }


if __name__ == "__main__":
    print("=" * 80)
    print("BERT INFERENCE AND MODEL COMPARISON")
    print("=" * 80)
    
    data_dir = Path("/Users/evwu/Documents/Repositories/ml_project_PI_redaction")
    model_dir = data_dir / "bert_model"
    
    # Check if BERT model exists
    if not model_dir.exists():
        print(f"ERROR: BERT model not found at {model_dir}")
        print("Please run bert_training.py first to train the model.")
        exit(1)
    
    # Run BERT inference
    print("\nStep 1: Running BERT inference...")
    print("-" * 80)
    
    bert_metrics, bert_predictions = run_bert_inference(
        model_dir,
        data_dir / "test_bio.jsonl",
        data_dir / "group_testing.csv",
        data_dir / "bert_predictions.jsonl"
    )
    
    print("\nBERT RESULTS:")
    print("-" * 80)
    print(f"Accuracy:  {bert_metrics['accuracy']:.4f}")
    print(f"Precision: {bert_metrics['precision']:.4f}")
    print(f"Recall:    {bert_metrics['recall']:.4f}")
    print(f"F1 Score:  {bert_metrics['f1']:.4f}")
    print(f"Examples:  {bert_metrics['num_examples']}")
    print(f"Tokens:    {bert_metrics['num_tokens']}")
    print("-" * 80)
    
    # Save BERT metrics
    with open(data_dir / "bert_metrics.json", 'w') as f:
        json.dump(bert_metrics, f, indent=2)
    
    # Compare models
    print("\nStep 2: Comparing SpaCy and BERT...")
    print("-" * 80)
    
    if (data_dir / "spacy_metrics.json").exists():
        comparison = compare_models(data_dir)
        
        # Save comparison
        with open(data_dir / "model_comparison.json", 'w') as f:
            json.dump(comparison, f, indent=2)
        
        print("\nComparison saved to model_comparison.json")
    else:
        print("SpaCy results not found. Run spacy_baseline.py first for comparison.")
    
    print("\n" + "=" * 80)
    print("INFERENCE COMPLETE!")
    print("=" * 80)
    print(f"BERT predictions: bert_predictions.jsonl")
    print(f"BERT metrics: bert_metrics.json")
    if (data_dir / "model_comparison.json").exists():
        print(f"Comparison: model_comparison.json")
