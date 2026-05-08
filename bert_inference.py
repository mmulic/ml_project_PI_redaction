import json
import numpy as np
import torch
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    DataCollatorForTokenClassification,
)
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# ============================================================
# DATA LOADING
# ============================================================

def load_jsonl(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data

# ============================================================
# TOKENIZATION
# ============================================================

def tokenize_and_align_labels(examples, tokenizer, label2id, max_length=200):
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
# HELPERS
# ============================================================

def collapse_bio(label):
    """B-NAME / I-NAME -> NAME, O -> O"""
    if label == "O":
        return "O"
    return label.split("-", 1)[-1]

def compute_flat_metrics(true_labels, pred_labels):
    return {
        "accuracy": accuracy_score(true_labels, pred_labels),
        "precision": precision_score(true_labels, pred_labels, average="weighted", zero_division=0),
        "recall": recall_score(true_labels, pred_labels, average="weighted", zero_division=0),
        "f1": f1_score(true_labels, pred_labels, average="weighted", zero_division=0),
    }

def per_class_accuracy(true_labels, pred_labels):
    """Compute accuracy per class."""
    classes = sorted(set(true_labels + pred_labels))
    class_acc = {}
    for cls in classes:
        idx = [i for i, label in enumerate(true_labels) if label == cls]
        if len(idx) == 0:
            class_acc[cls] = float('nan')
            continue
        correct = sum([1 for i in idx if true_labels[i] == pred_labels[i]])
        class_acc[cls] = correct / len(idx)
    return class_acc

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("BERT TEST SET EVALUATION (PII DETECTION)")
    print("=" * 60)

    data_dir = Path("")
    model_dir = data_dir / "bert_model"

    # ========================================================
    # LOAD LABEL MAPS
    # ========================================================

    label2id = json.load(open(model_dir / "label2id.json"))
    id2label = {int(k): v for k, v in json.load(open(model_dir / "id2label.json")).items()}

    # ========================================================
    # LOAD MODEL
    # ========================================================

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir).to(device)
    model.eval()

    # ========================================================
    # LOAD TEST DATA
    # ========================================================

    test_data = load_jsonl(data_dir / "test_bio.jsonl")

    test_dataset = Dataset.from_dict({
        "tokens": [x["tokens"] for x in test_data],
        "ner_tags": [x["ner_tags"] for x in test_data],
    })

    tokenize_fn = lambda x: tokenize_and_align_labels(x, tokenizer, label2id)

    test_tokenized = test_dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=test_dataset.column_names,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
    )

    # ========================================================
    # PREDICT
    # ========================================================

    print("\nRunning inference on test set...")
    preds_output = trainer.predict(test_tokenized)

    logits = preds_output.predictions
    labels = preds_output.label_ids

    predictions = np.argmax(logits, axis=2)

    # ========================================================
    # FLATTEN
    # ========================================================

    true_preds = []
    true_labels = []

    for i in range(len(predictions)):
        for j in range(len(predictions[i])):
            if labels[i][j] != -100:
                true_preds.append(predictions[i][j])
                true_labels.append(labels[i][j])

    true_preds_str = [id2label[p] for p in true_preds]
    true_labels_str = [id2label[l] for l in true_labels]

    # ========================================================
    # OVERALL METRICS
    # ========================================================

    print("\nOVERALL TEST RESULTS")
    print("-" * 60)

    overall_metrics = compute_flat_metrics(true_labels_str, true_preds_str)
    for k, v in overall_metrics.items():
        print(f"{k}: {v:.4f}")

    # ========================================================
    # CONFUSION MATRIX - OVERALL
    # ========================================================
    print("\nOVERALL CONFUSION MATRIX (B- and I- collapsed)")
    print("-" * 60)

    # Collapse BIO tags
    collapsed_true_labels = [collapse_bio(l) for l in true_labels_str]
    collapsed_pred_labels = [collapse_bio(p) for p in true_preds_str]

    # Get sorted class list
    classes_sorted = sorted(set(collapsed_true_labels + collapsed_pred_labels))

    # Compute confusion matrix
    overall_cm = confusion_matrix(collapsed_true_labels, collapsed_pred_labels, labels=classes_sorted)
    overall_cm_df = pd.DataFrame(overall_cm, index=classes_sorted, columns=classes_sorted)
    print(overall_cm_df)

    # ========================================================
    # PER-CLASS ACCURACY - OVERALL
    # ========================================================
    print("\nPER-CLASS ACCURACY (OVERALL)")
    print("-" * 60)
    overall_class_acc = per_class_accuracy(true_labels_str, true_preds_str)
    for cls, acc in overall_class_acc.items():
        print(f"{cls:15s}: {acc:.4f}")

    # ========================================================
    # PER-LANGUAGE METRICS
    # ========================================================

    for lang in ["English", "Spanish"]:

        lang_true_f = []
        lang_pred_f = []

        for i, item in enumerate(test_data):
            if item.get("language") != lang:
                continue
            for j in range(len(predictions[i])):
                if labels[i][j] != -100:
                    lang_true_f.append(id2label[labels[i][j]])
                    lang_pred_f.append(id2label[predictions[i][j]])

        if len(lang_true_f) == 0:
            print(f"{lang}: No samples found")
            continue

        lang_true_c = [collapse_bio(x) for x in lang_true_f]
        lang_pred_c = [collapse_bio(x) for x in lang_pred_f]

        print(f"\nLanguage: {lang}")
        print(classification_report(lang_true_c, lang_pred_c, zero_division=0))

    # ========================================================
    # PER-LANGUAGE CONFUSION MATRIX
    # ========================================================
    for lang in ["English", "Spanish"]:
        lang_true_f = []
        lang_pred_f = []

        for i, item in enumerate(test_data):
            if item.get("language") != lang:
                continue
            for j in range(len(predictions[i])):
                if labels[i][j] != -100:
                    lang_true_f.append(id2label[labels[i][j]])
                    lang_pred_f.append(id2label[predictions[i][j]])

        if len(lang_true_f) == 0:
            print(f"{lang}: No samples found")
            continue

        lang_true_c = [collapse_bio(x) for x in lang_true_f]
        lang_pred_c = [collapse_bio(x) for x in lang_pred_f]

        print(f"\nLanguage: {lang}")
        print(classification_report(lang_true_c, lang_pred_c, zero_division=0))

        # CONFUSION MATRIX (nicely formatted)
        print(f"\nCONFUSION MATRIX ({lang})")
        print("-" * 60)
        lang_classes = sorted(set(lang_true_c + lang_pred_c))
        lang_cm = confusion_matrix(lang_true_c, lang_pred_c, labels=lang_classes)
        lang_cm_df = pd.DataFrame(lang_cm, index=lang_classes, columns=lang_classes)
        print(lang_cm_df)

        # ====================================================
        # PER-CLASS ACCURACY - LANGUAGE
        # ====================================================
        print(f"\nPER-CLASS ACCURACY ({lang})")
        print("-" * 60)
        lang_class_acc = per_class_accuracy(lang_true_c, lang_pred_c)
        for cls, acc in lang_class_acc.items():
            print(f"{cls:15s}: {acc:.4f}")

    # ========================================================
    # OVERALL MACRO SCORES
    # ========================================================

    print("\nOVERALL MACRO SCORES")
    print("-" * 60)

    collapsed_true = [collapse_bio(x) for x in true_labels_str]
    collapsed_pred = [collapse_bio(x) for x in true_preds_str]

    macro_precision = precision_score(collapsed_true, collapsed_pred, average="macro", zero_division=0)
    macro_recall = recall_score(collapsed_true, collapsed_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(collapsed_true, collapsed_pred, average="macro", zero_division=0)
    macro_accuracy = accuracy_score(collapsed_true, collapsed_pred)

    print(f"Macro Accuracy   : {macro_accuracy:.4f}")
    print(f"Macro Precision  : {macro_precision:.4f}")
    print(f"Macro Recall     : {macro_recall:.4f}")
    print(f"Macro F1-Score   : {macro_f1:.4f}")

    print("\nDone.")
