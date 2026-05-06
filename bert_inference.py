import json
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForTokenClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def load_jsonl_data(jsonl_file):
    data = []
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def align_predictions(word_ids, pred_ids, id2label):
    """
    Convert subword-level predictions -> word-level labels
    """
    preds = []
    prev_word_id = None

    for idx, word_id in enumerate(word_ids):
        if word_id is None:
            continue

        # take first subword prediction per token
        if word_id != prev_word_id:
            preds.append(id2label[pred_ids[idx]])
            prev_word_id = word_id

    return preds


def run_bert_inference(model_dir, test_jsonl, output_file):
    print("Loading model...")

    model = AutoModelForTokenClassification.from_pretrained(model_dir).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)

    id2label = model.config.id2label

    test_data = load_jsonl_data(test_jsonl)

    all_true = []
    all_pred = []
    predictions = []

    # =========================
    # NEW: tracking containers
    # =========================
    class_stats = {}   # label-level metrics
    lang_stats = {}    # language-level metrics

    print(f"Running inference on {len(test_data)} examples...")

    for idx, item in enumerate(test_data):
        tokens = item["tokens"]
        true_labels = item["ner_tags"]
        language = item.get("language", "unknown")

        if not tokens:
            continue

        encodings = tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True
        ).to(device)

        with torch.no_grad():
            outputs = model(**encodings)

        logits = outputs.logits
        pred_ids = torch.argmax(logits, dim=-1).squeeze().cpu().numpy()

        word_ids = encodings.word_ids()
        pred_labels = align_predictions(word_ids, pred_ids, id2label)

        min_len = min(len(true_labels), len(pred_labels))
        true_labels = true_labels[:min_len]
        pred_labels = pred_labels[:min_len]

        all_true.extend(true_labels)
        all_pred.extend(pred_labels)

        predictions.append({
            "tokens": tokens,
            "true_labels": true_labels,
            "pred_labels": pred_labels,
            "language": language
        })

        # =========================
        # NEW: per-language tracking
        # =========================
        if language not in lang_stats:
            lang_stats[language] = {"true": [], "pred": []}

        lang_stats[language]["true"].extend(true_labels)
        lang_stats[language]["pred"].extend(pred_labels)

        # =========================
        # NEW: per-class tracking
        # =========================
        for t, p in zip(true_labels, pred_labels):
            if t not in class_stats:
                class_stats[t] = {"true": [], "pred": []}
            class_stats[t]["true"].append(t)
            class_stats[t]["pred"].append(p)

        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{len(test_data)}")

    # =========================
    # GLOBAL METRICS
    # =========================
    labels = list(id2label.values())
    label_to_id = {l: i for i, l in enumerate(labels)}

    y_true = [label_to_id.get(l, 0) for l in all_true]
    y_pred = [label_to_id.get(l, 0) for l in all_pred]

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "num_examples": len(predictions),
        "num_tokens": len(all_true)
    }

    print("\n" + "=" * 60)
    print("GLOBAL METRICS")
    print("=" * 60)
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1 Score : {metrics['f1']:.4f}")

    # =========================
    # PER-CLASS METRICS
    # =========================
    print("\n" + "=" * 60)
    print("PER-CLASS METRICS")
    print("=" * 60)

    per_class_metrics = {}

    for label, data in class_stats.items():
        yt = [1 if x == label else 0 for x in data["true"]]
        yp = [1 if x == label else 0 for x in data["pred"]]

        per_class_metrics[label] = {
            "accuracy": accuracy_score(yt, yp),
            "precision": precision_score(yt, yp, zero_division=0),
            "recall": recall_score(yt, yp, zero_division=0),
            "f1": f1_score(yt, yp, zero_division=0),
            "support": len(yt)
        }

        print(f"\n{label}")
        print(f"  Precision: {per_class_metrics[label]['precision']:.4f}")
        print(f"  Recall   : {per_class_metrics[label]['recall']:.4f}")
        print(f"  F1       : {per_class_metrics[label]['f1']:.4f}")
        print(f"  Support  : {per_class_metrics[label]['support']}")

    # =========================
    # PER-LANGUAGE METRICS
    # =========================
    print("\n" + "=" * 60)
    print("PER-LANGUAGE METRICS")
    print("=" * 60)

    per_language_metrics = {}

    for lang, data in lang_stats.items():
        yt = [label_to_id.get(l, 0) for l in data["true"]]
        yp = [label_to_id.get(l, 0) for l in data["pred"]]

        per_language_metrics[lang] = {
            "accuracy": accuracy_score(yt, yp),
            "precision": precision_score(yt, yp, average="weighted", zero_division=0),
            "recall": recall_score(yt, yp, average="weighted", zero_division=0),
            "f1": f1_score(yt, yp, average="weighted", zero_division=0),
            "support": len(yt)
        }

        print(f"\n{lang}")
        print(f"  Accuracy : {per_language_metrics[lang]['accuracy']:.4f}")
        print(f"  Precision: {per_language_metrics[lang]['precision']:.4f}")
        print(f"  Recall   : {per_language_metrics[lang]['recall']:.4f}")
        print(f"  F1       : {per_language_metrics[lang]['f1']:.4f}")
        print(f"  Support  : {per_language_metrics[lang]['support']}")

    # =========================
    # SAVE RESULTS
    # =========================
    with open(output_file, "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")

    return metrics, per_class_metrics, per_language_metrics, predictions


if __name__ == "__main__":
    print("=" * 60)
    print("BERT TOKEN-LEVEL INFERENCE (FIXED)")
    print("=" * 60)

    data_dir = Path("")
    model_dir = data_dir / "bert_model"

    if not model_dir.exists():
        print("ERROR: model not found")
        exit(1)

    bert_metrics, _ = run_bert_inference(
        model_dir,
        data_dir / "test_bio.jsonl",
        data_dir / "bert_predictions.jsonl"
    )

    with open(data_dir / "bert_metrics.json", "w") as f:
        json.dump(bert_metrics, f, indent=2)

    print("\nSaved: bert_metrics.json")
    print("Saved: bert_predictions.jsonl")
