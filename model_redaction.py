import ast
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import pipeline

MODEL_ID   = "Isotonic/distilbert_finetuned_ai4privacy_v2"
BATCH_SIZE = 64


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def build_pipeline():
    if torch.cuda.is_available():
        device = 0
        device_name = "CUDA GPU"
    elif torch.backends.mps.is_available():
        device = "mps"
        device_name = "Apple Silicon GPU (MPS)"
    else:
        device = -1
        device_name = "CPU"

    print(f"Running on: {device_name}")

    return pipeline(
        "token-classification",
        model=MODEL_ID,
        aggregation_strategy="simple",
        batch_size=BATCH_SIZE,
        device=device,
    )


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------

def redact_text(text: str, entities: list) -> str:
    for ent in sorted(entities, key=lambda e: e["start"], reverse=True):
        label = ent["entity_group"]
        text  = text[:ent["start"]] + f"[{label}]" + text[ent["end"]:]
    return text


def run_redaction(df: pd.DataFrame, ner, text_col: str = "source_text"):
    texts   = df[text_col].fillna("").tolist()
    results = []

    # Process in chunks so tqdm can show per-batch progress and ETA
    chunks = [texts[i:i + BATCH_SIZE] for i in range(0, len(texts), BATCH_SIZE)]
    for chunk in tqdm(chunks, desc="  Batches", unit="batch"):
        results.extend(ner(chunk))

    redacted = [redact_text(t, ents) for t, ents in zip(texts, results)]
    df = df.copy()
    df[f"{text_col}_redacted"] = redacted
    return df, results


# ---------------------------------------------------------------------------
# Evaluation from pii redaction
# ---------------------------------------------------------------------------

def parse_span_labels(raw) -> list:
    if pd.isna(raw):
        return []
    if isinstance(raw, str):
        try:
            raw = ast.literal_eval(raw)
        except Exception:
            return []
    return [(int(s[0]), int(s[1])) for s in raw if len(s) >= 3]


def spans_to_mask(spans: list, text_len: int) -> np.ndarray:
    mask = np.zeros(text_len, dtype=bool)
    for start, end in spans:
        mask[start:end] = True
    return mask


def word_level_metrics(text: str, gt_spans: list, pred_entities: list) -> dict:
    import re
    pred_spans = [(e["start"], e["end"]) for e in pred_entities]

    gt_mask   = spans_to_mask(gt_spans,   len(text))
    pred_mask = spans_to_mask(pred_spans, len(text))

    tp = fp = fn = tn = 0
    for m in re.finditer(r'\S+', text):
        is_gt   = gt_mask[m.start():m.end()].any()
        is_pred = pred_mask[m.start():m.end()].any()
        if     is_gt and     is_pred: tp += 1
        elif   is_gt and not is_pred: fn += 1
        elif not is_gt and  is_pred:  fp += 1
        else:                         tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    accuracy  = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall,
            "f1": f1, "accuracy": accuracy}


def evaluate(df: pd.DataFrame, all_entities: list,
             text_col: str = "source_text") -> dict:
    totals = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}

    for (_, row), entities in zip(df.iterrows(), all_entities):
        text = row.get(text_col, "")
        if not isinstance(text, str) or not text:
            continue
        gt_spans = parse_span_labels(row.get("span_labels"))
        m = word_level_metrics(text, gt_spans, entities)
        for k in totals:
            totals[k] += m[k]

    tp, fp, fn, tn = totals["tp"], totals["fp"], totals["fn"], totals["tn"]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    accuracy  = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    return {**totals, "precision": precision, "recall": recall,
            "f1": f1, "accuracy": accuracy}


def print_metrics(split_name: str, m: dict) -> None:
    print(f"\n{'='*52}")
    print(f"  {split_name} — DistilBERT Word level")
    print(f"{'='*52}")
    print(f"  Accuracy  : {m['accuracy']:.4f}")
    print(f"  Precision : {m['precision']:.4f}")
    print(f"  Recall    : {m['recall']:.4f}")
    print(f"  F1        : {m['f1']:.4f}")
    print(f"  TP={m['tp']:,}  FP={m['fp']:,}  FN={m['fn']:,}  TN={m['tn']:,}")


# ---------------------------------------------------------------------------
# Per-split pipeline
# ---------------------------------------------------------------------------

def process_split(input_csv: str, output_csv: str, ner,
                  text_col: str = "source_text") -> None:
    df = pd.read_csv(input_csv)

    if text_col not in df.columns:
        raise ValueError(
            f"Column '{text_col}' not found. Available: {df.columns.tolist()}"
        )

    print(f"\n[{input_csv}] {len(df):,} rows — running DistilBERT...")
    df, all_entities = run_redaction(df, ner, text_col)

    modified = (df[f"{text_col}_redacted"] != df[text_col]).sum()
    print(f"  {modified:,} rows modified → saved to '{output_csv}'")
    df.to_csv(output_csv, index=False)

    if "span_labels" in df.columns:
        split_name = input_csv.replace(".csv", "").replace("group_", "").upper()
        m = evaluate(df, all_entities, text_col)
        print_metrics(split_name, m)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ner = build_pipeline()
    process_split("group_testing.csv", "group_testing_model_redacted.csv", ner)
    print("\n\nDone.")
