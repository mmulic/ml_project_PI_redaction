import re
import ast
import numpy as np
import pandas as pd

# Labels match the dataset's actual classes. Only classes with a feasible
# regex pattern are included — free-text classes (GIVENNAME, LASTNAME, CITY,
# STATE, COUNTRY, STREET, SEX, TITLE) cannot be reliably caught by regex.
PII_PATTERNS = {
    # --- Contact ---
    "EMAIL":         r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b',
    "TEL":           r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b',

    # --- Network ---
    # Covers both IPv4 and IPv6 under the dataset's single IP label
    "IP":            r'\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b'
                     r'|\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b',

    # --- Date / time ---
    "DATE":          r'\b(?:\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2}|\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})\b',
    "BOD":           r'\b(?:\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2}|\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})\b',
    "TIME":          r'\b\d{1,2}:\d{2}(?::\d{2})?(?:\s?[APap][Mm])?\b',

    # --- Location ---
    # POSTCODE covers US ZIP (12345 / 12345-6789) and UK postcodes (SW1A 1AA)
    "POSTCODE":      r'\b[A-Z]{1,2}\d{1,2}[A-Z]?\s?\d[A-Z]{2}\b|\b\d{5}(?:-\d{4})?\b',
    "GEOCOORD":      r'-?\d{1,3}\.\d{3,},\s*-?\d{1,3}\.\d{3,}',
    "BUILDING":      r'\b\d{1,5}[A-Za-z]?\b',
    "SECADDRESS":    r'(?i)\b(?:apt|apartment|suite|ste|unit|floor|fl|room|rm)\.?\s*[\w\-]+\b',

    # --- IDs & documents ---
    "SOCIALNUMBER":  r'\b\d{3}[-\s]\d{2}[-\s]\d{4}\b',
    "PASSPORT":      r'\b[A-Z]{1,2}\d{6,9}\b',
    "DRIVERLICENSE": r'\b[A-Z]{1,2}\d{6,8}[A-Z]?\b',
    "IDCARD":        r'\b[A-Z]{0,2}\d{6,9}\b',

    # --- Credentials ---
    "USERNAME":      r'(?i)\busername\s*:?\s*\S+',
    "PASS":          r'(?i)\bpassword\s*:?\s*\S+',
}


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------

def get_predicted_spans(text: str) -> list[tuple[int, int, str]]:
    """Return (start, end, label) for every regex match in text."""
    spans = []
    for label, pattern in PII_PATTERNS.items():
        for m in re.finditer(pattern, text):
            spans.append((m.start(), m.end(), label))
    return spans


def redact_pii(text: str) -> str:
    if not isinstance(text, str):
        return text
    for label, pattern in PII_PATTERNS.items():
        text = re.sub(pattern, f'[{label}]', text)
    return text


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def parse_span_labels(raw) -> list[tuple[int, int, str]]:
    """Parse the span_labels column into (start, end, label) tuples."""
    if pd.isna(raw):
        return []
    if isinstance(raw, str):
        try:
            raw = ast.literal_eval(raw)
        except Exception:
            return []
    result = []
    for span in raw:
        if len(span) >= 3:
            result.append((int(span[0]), int(span[1]), str(span[2])))
    return result


def spans_to_mask(spans: list[tuple[int, int, str]], text_len: int) -> np.ndarray:
    """Boolean char-level mask: True where a character belongs to a PII span."""
    mask = np.zeros(text_len, dtype=bool)
    for start, end, _ in spans:
        mask[start:end] = True
    return mask


# ---------------------------------------------------------------------------
# Word-level evaluation
# ---------------------------------------------------------------------------

def word_level_metrics(text: str, gt_spans: list, pred_spans: list) -> dict:
    """
    Tokenise text by whitespace. Each word is labelled PII if any of its
    characters overlap with a GT or predicted span. This gives a natural TN:
    words that are not PII and were not flagged.
    """
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


def evaluate_word_level(df: pd.DataFrame, text_col: str = "source_text") -> dict:
    totals = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}

    for _, row in df.iterrows():
        text = row.get(text_col, "")
        if not isinstance(text, str) or not text:
            continue

        gt_spans   = parse_span_labels(row.get("span_labels"))
        pred_spans = get_predicted_spans(text)

        m = word_level_metrics(text, gt_spans, pred_spans)
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


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_metrics(split_name: str, word_m: dict) -> None:
    print(f"\n{'='*52}")
    print(f"  {split_name} — Word level")
    print(f"{'='*52}")
    print(f"  Accuracy  : {word_m['accuracy']:.4f}")
    print(f"  Precision : {word_m['precision']:.4f}")
    print(f"  Recall    : {word_m['recall']:.4f}")
    print(f"  F1        : {word_m['f1']:.4f}")
    print(f"  TP={word_m['tp']:,}  FP={word_m['fp']:,}  "
          f"FN={word_m['fn']:,}  TN={word_m['tn']:,}")


# ---------------------------------------------------------------------------
# Per-split pipeline
# ---------------------------------------------------------------------------

def process_split(input_csv: str, output_csv: str,
                  text_col: str = "source_text") -> None:
    df = pd.read_csv(input_csv)

    if text_col not in df.columns:
        raise ValueError(
            f"Column '{text_col}' not found. Available: {df.columns.tolist()}"
        )

    # Redact
    df[f"{text_col}_redacted"] = df[text_col].apply(redact_pii)
    modified = (df[f"{text_col}_redacted"] != df[text_col]).sum()
    print(f"\n[{input_csv}] {len(df):,} rows | {modified:,} rows modified → '{output_csv}'")
    df.to_csv(output_csv, index=False)

    # Evaluate against ground truth if span_labels exists
    if "span_labels" in df.columns:
        split_name = input_csv.replace(".csv", "").replace("group_", "").upper()
        word_m = evaluate_word_level(df, text_col)
        print_metrics(split_name, word_m)
    else:
        print("  (skipping metrics — 'span_labels' column not found)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    splits = [
        ("group_training.csv",   "group_training_redacted.csv"),
        ("group_validation.csv", "group_validation_redacted.csv"),
        ("group_testing.csv",    "group_testing_redacted.csv"),
    ]

    for input_path, output_path in splits:
        process_split(input_path, output_path)

    print("\n\nDone. Redacted CSVs written for all three splits.")
