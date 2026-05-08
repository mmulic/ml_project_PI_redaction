import json
import psutil
import time
import torch
import numpy as np
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# DEVICE SETUP
# ============================================================

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
    print("=" * 60 + "\n")


# ============================================================
# DATA LOADING
# ============================================================

def load_jsonl_data(jsonl_file):
    """Load data from JSONL file."""
    data = []
    with open(jsonl_file, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def create_label_mapping(train_file, val_file, test_file):
    """
    Extract all unique labels and create mappings.
    """

    all_labels = set()
    all_labels.add("O")

    for jsonl_file in [train_file, val_file, test_file]:
        data = load_jsonl_data(jsonl_file)

        for item in data:
            all_labels.update(item["ner_tags"])

    label_list = sorted(list(all_labels))

    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for label, i in label2id.items()}

    print(f"Found {len(label_list)} labels")
    print(label_list)

    return label2id, id2label, label_list


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize_and_align_labels(examples, tokenizer, label2id, max_length=50):
    """
    Tokenize and align labels for token classification.
    """

    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,
    )

    labels = []

    for i, label_list in enumerate(examples["ner_tags"]):

        word_ids = tokenized_inputs.word_ids(batch_index=i)

        label_ids = []
        previous_word_idx = None

        for word_idx in word_ids:

            if word_idx is None:
                label_ids.append(-100)

            elif word_idx != previous_word_idx:
                label_ids.append(label2id[label_list[word_idx]])

            else:
                label_ids.append(-100)

            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels

    return tokenized_inputs


# ============================================================
# METRICS
# ============================================================

def compute_metrics(p):
    """
    Compute evaluation metrics.
    """

    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        predictions[i][j]
        for i in range(len(predictions))
        for j in range(len(predictions[i]))
        if labels[i][j] != -100
    ]

    true_labels = [
        labels[i][j]
        for i in range(len(labels))
        for j in range(len(labels[i]))
        if labels[i][j] != -100
    ]

    results = {
        "accuracy": accuracy_score(true_labels, true_predictions),
        "precision": precision_score(
            true_labels,
            true_predictions,
            average="weighted",
            zero_division=0,
        ),
        "recall": recall_score(
            true_labels,
            true_predictions,
            average="weighted",
            zero_division=0,
        ),
        "f1": f1_score(
            true_labels,
            true_predictions,
            average="weighted",
            zero_division=0,
        ),
    }

    return results


if __name__ == "__main__":

    print("=" * 60)
    print("BERT TRAINING FOR PII DETECTION")
    print("=" * 60)

    data_dir = Path("")
    model_output_dir = data_dir / "bert_model"

    # ========================================================
    # STEP 1 - LOAD DATA
    # ========================================================

    print("\nStep 1: Loading data...")
    print("-" * 60)

    label2id, id2label, label_list = create_label_mapping(
        data_dir / "train_bio.jsonl",
        data_dir / "val_bio.jsonl",
        data_dir / "test_bio.jsonl",
    )

    train_data = load_jsonl_data(data_dir / "train_bio.jsonl")
    val_data = load_jsonl_data(data_dir / "val_bio.jsonl")
    test_data = load_jsonl_data(data_dir / "test_bio.jsonl")

    print(f"Training examples:   {len(train_data)}")
    print(f"Validation examples: {len(val_data)}")
    print(f"Test examples:       {len(test_data)}")

    # ========================================================
    # STEP 2 - LOAD MODEL
    # ========================================================

    print("\nStep 2: Loading model...")
    print("-" * 60)

    model_name = "distilbert-base-multilingual-cased"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    ).to(device)

    # Freeze base model
    for param in model.distilbert.parameters():
        param.requires_grad = False

    print(f"Model: {model_name}")
    print(f"Parameters: {model.num_parameters():,}")

    # ========================================================
    # STEP 3 - DATASETS
    # ========================================================

    print("\nStep 3: Tokenizing datasets...")
    print("-" * 60)

    train_dataset = Dataset.from_dict({
        "tokens": [item["tokens"] for item in train_data],
        "ner_tags": [item["ner_tags"] for item in train_data],
    })

    val_dataset = Dataset.from_dict({
        "tokens": [item["tokens"] for item in val_data],
        "ner_tags": [item["ner_tags"] for item in val_data],
    })

    test_dataset = Dataset.from_dict({
        "tokens": [item["tokens"] for item in test_data],
        "ner_tags": [item["ner_tags"] for item in test_data],
    })

    tokenize_fn = lambda examples: tokenize_and_align_labels(
        examples,
        tokenizer,
        label2id,
    )

    train_tokenized = train_dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=train_dataset.column_names,
    )

    val_tokenized = val_dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=val_dataset.column_names,
    )

    test_tokenized = test_dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=test_dataset.column_names,
    )

    # ========================================================
    # STEP 4 - TRAINING SETUP
    # ========================================================

    print("\nStep 4: Configuring training...")
    print("-" * 60)

    training_args = TrainingArguments(
        output_dir=str(model_output_dir),
        num_train_epochs=5,
        per_device_train_batch_size=32 if device.type == "cuda" else 4,
        per_device_eval_batch_size=32 if device.type == "cuda" else 4,
        gradient_accumulation_steps=2,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=500,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=5,
        fp16=device.type == "cuda",
        optim="adamw_torch",
        max_grad_norm=1.0,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=1,
        early_stopping_threshold=0.0001,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping],
    )
    print("\nStep 5: Training model...")
    print("-" * 60)

    # Record start time
    start_time = time.time()

    # Optional: clear CUDA cache before training
    if device.type == "cuda":
        torch.cuda.empty_cache()

    trainer.train()

    # Record end time
    end_time = time.time()
    total_training_time = end_time - start_time

    # Track peak GPU memory usage if on CUDA
    peak_gpu_memory = None
    if device.type == "cuda":
        peak_gpu_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # in MB

    # Track CPU memory usage
    process = psutil.Process()
    peak_cpu_memory = process.memory_info().rss / (1024 ** 2)  # in MB

    print(f"\nTraining completed in {total_training_time:.2f} seconds ({total_training_time/60:.2f} minutes).")
    if peak_gpu_memory:
        print(f"Peak GPU memory usage: {peak_gpu_memory:.2f} MB")
    print(f"CPU memory usage at the end of training: {peak_cpu_memory:.2f} MB")


    print("\nStep 6: Evaluating BEST model on validation set...")
    print("-" * 60)
    val_results = trainer.evaluate(val_tokenized)
    print("\nValidation Results:")
    for key, value in val_results.items():
        print(f"{key}: {value}")

    # ========================================================
    # TEST EVALUATION
    # ========================================================

    print("\nStep 7: Evaluating BEST model on test set...")
    print("-" * 60)

    test_results = trainer.evaluate(test_tokenized)

    print("\nTest Results:")
    for key, value in test_results.items():
        print(f"{key}: {value}")

    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    print("\nStep 8: Saving BEST model...")
    print("-" * 60)

    trainer.save_model(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)

    with open(model_output_dir / "label2id.json", "w") as f:
        json.dump(label2id, f)

    with open(model_output_dir / "id2label.json", "w") as f:
        json.dump(id2label, f)

    print(f"\nBest model saved to: {model_output_dir}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
