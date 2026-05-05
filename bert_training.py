"""
BERT Training for PII Detection
This script fine-tunes a pretrained BERT model on your PII data.

BERT is a powerful language model that understands context, so it should
outperform SpaCy on detecting PII in your specific domain.

This script:
1. Loads the prepared data (BIO labels)
2. Tokenizes text using BERT's tokenizer
3. Fine-tunes BERT for token classification
4. Saves the model for later use
"""

import json
import torch
import numpy as np
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Device setup (prefer CUDA, then Apple Silicon MPS, then CPU)
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

if device.type == "cpu":
    print("\n" + "=" * 60)
    print("WARNING: Training on CPU!")
    print("This will be extremely slow. On a MacBook Air with Apple Silicon,")
    print("PyTorch should use the MPS backend if it is installed correctly.")
    print("=" * 60 + "\n")

def load_jsonl_data(jsonl_file):
    """Load data from JSONL file."""
    data = []
    with open(jsonl_file, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def create_label_mapping(train_file, val_file, test_file):
    """
    Extract all unique labels from the data and create a mapping.
    This is important because BERT needs numeric labels, not strings.
    """
    all_labels = set()
    all_labels.add("O")  # Always include the "outside" label
    
    for jsonl_file in [train_file, val_file, test_file]:
        data = load_jsonl_data(jsonl_file)
        for item in data:
            all_labels.update(item["ner_tags"])
    
    # Create mapping
    label_list = sorted(list(all_labels))
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for label, i in label2id.items()}
    
    print(f"Found {len(label_list)} unique labels: {label_list}")
    return label2id, id2label, label_list

# changed max length to 256 instead of 512
def tokenize_and_align_labels(examples, tokenizer, label2id, max_length=256):
    """
    Tokenize text with BERT's tokenizer and align labels.
    
    Challenge: BERT's tokenizer splits words into subword tokens.
    Example: "John" might become ["John"], but "computer" might become ["com", "##put", "##er"]
    
    Solution: When a word is split into subwords, only the first subword gets the real label,
    and the rest get -100 (which we ignore in training).
    """
    
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=max_length,
        padding='max_length',
        return_tensors=None
    )
    
    labels = []
    for i, label_list in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None
        
        for word_idx in word_ids:
            if word_idx is None:
                # Special tokens like [CLS], [SEP], [PAD] get -100
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                # First subword of a word gets the real label
                label_ids.append(label2id[label_list[word_idx]])
            else:
                # Subsequent subwords get -100 (ignored in loss calculation)
                label_ids.append(-100)
            previous_word_idx = word_idx
        
        labels.append(label_ids)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def compute_metrics(p):
    """
    Compute metrics for BERT evaluation.
    p contains predictions and label_ids
    """
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    
    # Remove ignored index (special tokens)
    true_predictions = [
        predictions[i][j] for i in range(len(predictions)) for j in range(len(predictions[i]))
        if labels[i][j] != -100
    ]
    true_labels = [
        labels[i][j] for i in range(len(labels)) for j in range(len(labels[i]))
        if labels[i][j] != -100
    ]
    
    # Calculate metrics
    results = {
        "accuracy": accuracy_score(true_labels, true_predictions),
        "precision": precision_score(true_labels, true_predictions, average='weighted', zero_division=0),
        "recall": recall_score(true_labels, true_predictions, average='weighted', zero_division=0),
        "f1": f1_score(true_labels, true_predictions, average='weighted', zero_division=0),
    }
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("BERT TRAINING FOR PII DETECTION")
    print("=" * 60)
    
    data_dir = Path("/Users/evwu/Documents/Repositories/ml_project_PI_redaction")
    model_output_dir = data_dir / "bert_model"
    
    # Step 1: Load data and create label mapping
    print("\nStep 1: Loading data and creating label mapping...")
    print("-" * 60)
    
    label2id, id2label, label_list = create_label_mapping(
        data_dir / "train_bio.jsonl",
        data_dir / "val_bio.jsonl",
        data_dir / "test_bio.jsonl"
    )
    
    train_data = load_jsonl_data(data_dir / "train_bio.jsonl")
    val_data = load_jsonl_data(data_dir / "val_bio.jsonl")
    test_data = load_jsonl_data(data_dir / "test_bio.jsonl")
    
    print(f"Training examples: {len(train_data)}")
    print(f"Validation examples: {len(val_data)}")
    print(f"Test examples: {len(test_data)}")
    
    # Step 2: Load tokenizer and model
    print("\nStep 2: Loading BERT tokenizer and model...")
    print("-" * 60)
    
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id
    ).to(device)
    
    print(f"Model: {model_name}")
    print(f"Number of parameters: {model.num_parameters():,}")
    
    # Step 3: Tokenize and prepare datasets
    print("\nStep 3: Tokenizing data...")
    print("-" * 60)
    
    train_dataset = Dataset.from_dict({
        "tokens": [item["tokens"] for item in train_data],
        "ner_tags": [item["ner_tags"] for item in train_data]
    })
    
    val_dataset = Dataset.from_dict({
        "tokens": [item["tokens"] for item in val_data],
        "ner_tags": [item["ner_tags"] for item in val_data]
    })
    
    test_dataset = Dataset.from_dict({
        "tokens": [item["tokens"] for item in test_data],
        "ner_tags": [item["ner_tags"] for item in test_data]
    })
    
    # Tokenize all datasets
    tokenize_fn = lambda examples: tokenize_and_align_labels(examples, tokenizer, label2id)
    
    train_tokenized = train_dataset.map(tokenize_fn, batched=True, remove_columns=train_dataset.column_names)
    val_tokenized = val_dataset.map(tokenize_fn, batched=True, remove_columns=val_dataset.column_names)
    test_tokenized = test_dataset.map(tokenize_fn, batched=True, remove_columns=test_dataset.column_names)
    
    print(f"Train examples: {len(train_tokenized)}")
    print(f"Validation examples: {len(val_tokenized)}")
    print(f"Test examples: {len(test_tokenized)}")
    
    # Step 4: Training setup
    print("\nStep 4: Setting up training...")
    print("-" * 60)
    
    training_args = TrainingArguments(
        output_dir=str(model_output_dir),
        num_train_epochs=3,
        per_device_train_batch_size=32 if device.type == "cuda" else 8,  # Smaller for CPU
        per_device_eval_batch_size=32 if device.type == "cuda" else 8,
        gradient_accumulation_steps=2 if device.type == "cuda" else 4,  # Simulate larger batch
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=50,  # Log more frequently to monitor progress
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=device.type == "cuda",  # Mixed precision for CUDA only
        optim="adamw_torch",  # More efficient optimizer
        max_grad_norm=1.0,
        warmup_steps=100,
    )
    
    data_collator = DataCollatorForTokenClassification(tokenizer)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Step 5: Train the model
    print("\nStep 5: Training BERT...")
    print("-" * 60)
    print("This may take a few minutes depending on your GPU/CPU...\n")
    
    trainer.train()
    
    # Step 6: Save the model
    print("\nStep 6: Saving model...")
    print("-" * 60)
    
    model.save_pretrained(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)
    
    with open(model_output_dir / "label2id.json", 'w') as f:
        json.dump(label2id, f)
    with open(model_output_dir / "id2label.json", 'w') as f:
        json.dump(id2label, f)
    
    print(f"Model saved to {model_output_dir}")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("Next step: Run bert_inference.py to evaluate on test set")
    print("=" * 60)
