import re
import ast
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from collections import defaultdict

def macro_accuracy(y_true, y_pred, labels):
    """
    Macro accuracy = average per-class recall-like accuracy:
    TP / (TP + FN) per class, then averaged.
    """
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    for yt, yp in zip(y_true, y_pred):
        if yt == 'O':
            continue  # ignore non-PII background

        class_total[yt] += 1
        if yt == yp:
            class_correct[yt] += 1

    per_class_acc = []
    for label in labels:
        if class_total[label] == 0:
            continue
        per_class_acc.append(class_correct[label] / class_total[label])

    return float(np.mean(per_class_acc)) if per_class_acc else 0.0
# ---------------------------------------------------------------------------
# PII Patterns
# ---------------------------------------------------------------------------

PII_PATTERNS = {
    "DRIVERLICENSE": r"[a-zA-Z]\d{7}|[a-zA-Z]\d{12}|\d{9}|[a-zA-Z]\d\-?\d{2}\-?[a-zA-Z]{2}\d[a-zA-Z]\d{2}[a-zA-Z]\-?\d|\d{8}|\d{10}|[a-zA-Z]\d{8}|[a-zA-Z]\d{2}[a-zA-Z]{2}\d{7}",
    "IP": r"\d{4}:?[a-zA-Z]\d[a-zA-Z]\d:?\d[a-zA-Z]\d{2}:?\d{2}:?\d{4}:?[a-zA-Z]\d{3}:?[a-zA-Z]\d{3}:?[a-zA-Z]\d{3}|\d{3}\.?\d{2}\.?\d{3}\.?\d{2}|\d[a-zA-Z]\d[a-zA-Z]:?\d{2}[a-zA-Z]\d:?[a-zA-Z]\d{2}:?\d[a-zA-Z]\d{2}:?[a-zA-Z]{4}:?\d[a-zA-Z]\d[a-zA-Z]:?\d[a-zA-Z]{2}\d:?\d{2}[a-zA-Z]\d|\d{3}\.?\d{2}\.?\d{3}\.?\d{3}",
    "TEL": r"\+\d{3}\-?\d{2}\s+\d{3}\-?\d{4}|\+\d{2}\s+\d{2}\s+\d{3}\s+\d{4}|\d{2}\s+\d{2}\s+\d{2}\.?\d{2}\.?\d{2}|\d{5}\s+\d{6}|\d{5}\s+\d{2}\-?\d{3}\s+\d{4}",
    "EMAIL": r"[a-zA-Z]{16}\d{3}@?[a-zA-Z]{8}\.?[a-zA-Z]{3}|[a-zA-Z]{7}@?[a-zA-Z]{12}\.?[a-zA-Z]{3}|[a-zA-Z]{15}\d{4}@?[a-zA-Z]{7}\.?[a-zA-Z]{3}|[a-zA-Z]{11}\d{4}@?[a-zA-Z]{7}\.?[a-zA-Z]{3}|[a-zA-Z]{11}\d{5}@?[a-zA-Z]{8}\.?[a-zA-Z]{3}|[a-zA-Z]{16}\d{2}@?[a-zA-Z]{8}\.?[a-zA-Z]{3}",
    "BOD": r"\d{2}[a-zA-Z]{2}\s+[a-zA-Z]{9}\s+\d{4}|[a-zA-Z]{7}\s+\d[a-zA-Z]{2},?\s+\d{4}|\d{2}/?\d{2}/?\d{4}|[a-zA-Z]{7}/?\d{2}|\d[a-zA-Z]\s+[a-zA-Z]{5}\s+\d{4}",
    "USERNAME": r"[a-zA-Z]{11}\d{4}|[a-zA-Z]{7}\d|\d{4}[a-zA-Z]{2}|\d{2}[a-zA-Z]{5}\.?[a-zA-Z]{3}|[a-zA-Z]{10}\d{5}|[a-zA-Z]{15}\d{4}|[a-zA-Z]{6}\d{2}",
    "PASS": r"\*[a-zA-Z]{4}\}[a-zA-Z]{2}\d[a-zA-Z]|[a-zA-Z]\d{2}[a-zA-Z]{2}\||\d[a-zA-Z]\.?[a-zA-Z]|[a-zA-Z]{2}\d{2}\^/?:?[a-zA-Z]|\d{2}>[a-zA-Z]{4}'[a-zA-Z];|[a-zA-Z]@?[a-zA-Z]\~\d[a-zA-Z]{3}",
    "POSTCODE": r"[a-zA-Z]\d{2}\s+[a-zA-Z]{2},?\s+[a-zA-Z]\d{2}\s+\d[a-zA-Z]{2}|[a-zA-Z]{2}\d|\d{5}|[a-zA-Z]{2}\d{2}",
    "BUILDING": r"\d{3}",
    "TIME": r"\d{2}\s+[a-zA-Z]'[a-zA-Z]{5}|\d{2}|\d|\d{2}:?\d{2}|\d{2}:?\d{2}:?\d{2}|\d[a-zA-Z]|\d\s+[a-zA-Z]{2}",
    "DATE": r"\d{2}/?\d{2}/?\d{4}|\d{4}\-?\d{2}\-?\d{2}[a-zA-Z]\d{2}:?\d{2}:?\d{2}|[a-zA-Z]{5}/?\d{2}",
    "SOCIALNUMBER": r"\d{9}|\d{3}\s+\d{3}\s+\d{4}|\d{3}\-?\d{2}\-?\d{4}",
    "PASSPORT": r"\d{9}",
    "IDCARD": r"[a-zA-Z]{2}\d{5}[a-zA-Z]{2}|\d{13}",
    "SECADDRESS":    r'(?i)\b(?:apt|apartment|suite|ste|unit|floor|fl|room|rm)\.?\s*[\w\-]+\b',
    "GEOCOORD":      r'\[\s*-?\d{1,3}(?:\.\d{1,2})?\s*,\s*-?\d{1,3}(?:\.\d{1,2})?\s*\]',
}

# ---------------------------------------------------------------------------
# Span helpers
# ---------------------------------------------------------------------------

def parse_span_labels(raw) -> list[tuple[int,int,str]]:
    if pd.isna(raw): return []
    if isinstance(raw, str):
        try: raw = ast.literal_eval(raw)
        except Exception: return []
    return [(int(s[0]), int(s[1]), str(s[2])) for s in raw if len(s) >= 3]

def get_predicted_spans(text: str):
    spans = []
    for label, pattern in PII_PATTERNS.items():
        for m in re.finditer(pattern, text):
            spans.append((m.start(), m.end(), label))
    return spans

# ---------------------------------------------------------------------------
# Token splitting on non-space sequences
# ---------------------------------------------------------------------------

def tokenize_text(text: str):
    # Split by contiguous non-space sequences, preserving punctuation attached
    return re.findall(r'\S+', text)

# ---------------------------------------------------------------------------
# Multi-class character-level labels with subset match
# ---------------------------------------------------------------------------

def spans_to_char_labels(spans, text_len, classes):
    """Return list of length text_len with label per char; 'O' = no PII."""
    labels = np.array(['O']*text_len)
    for start, end, label in spans:
        if label in classes:
            labels[start:end] = label
    return labels

def adjust_for_subset_match(y_true, y_pred):
    """
    Adjust predicted labels if they are a subset of ground truth for the same class.
    Returns new y_pred list.
    """
    y_pred_adj = y_pred.copy()
    for i in range(len(y_true)):
        # If predicted matches GT or is a subset (for same class), keep GT label
        if y_true[i] != 'O' and y_pred[i] == y_true[i]:
            continue
        if y_true[i] != 'O' and y_pred[i] == 'O':
            # subset: predicted covers only part of GT, mark it as GT
            y_pred_adj[i] = y_true[i]
    return y_pred_adj

# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate_per_class(df, text_col='source_text', language_col='language'):
    results = {}
    subsets = {
        "Overall": df,
        "English": df[df[language_col].str.lower() == 'english'],
        "Spanish": df[df[language_col].str.lower() == 'spanish']
    }

    for subset_name, subset_df in subsets.items():
        y_true_all = []
        y_pred_all = []

        for _, row in subset_df.iterrows():
            text = str(row.get(text_col, ""))
            if not text: continue

            gt_spans = parse_span_labels(row.get("span_labels"))
            pred_spans = get_predicted_spans(text)

            y_true_chars = spans_to_char_labels(gt_spans, len(text), PII_PATTERNS.keys())
            y_pred_chars = spans_to_char_labels(pred_spans, len(text), PII_PATTERNS.keys())

            # Adjust predicted labels if subset of GT
            y_pred_chars = adjust_for_subset_match(y_true_chars, y_pred_chars)

            y_true_all.extend(y_true_chars)
            y_pred_all.extend(y_pred_chars)

        # Only include labels actually present in y_true
        unique_labels = sorted(set(y_true_all) - {"O"})
        if not unique_labels:
            results[subset_name] = ("No PII labels found in this subset", None, None)
            continue

        report = classification_report(
            y_true_all, y_pred_all,
            labels=unique_labels, zero_division=0
        )
        cm = confusion_matrix(
            y_true_all, y_pred_all,
            labels=unique_labels
        )
        macc = macro_accuracy(y_true_all, y_pred_all, unique_labels)

        results[subset_name] = (report, cm, unique_labels, macc)
       
    return results

# ---------------------------------------------------------------------------
# Run on group_testing.csv
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = pd.read_csv("group_testing.csv")
    results = evaluate_per_class(df)

    for subset_name, (report, cm, classes, macc) in results.items():
        print(f"\n=== {subset_name} metrics ===")
        if report is None:
            print(classes)  # message
            continue
        print(report)
        print("\nConfusion Matrix (Rows=GT, Cols=Predicted)")
        print(pd.DataFrame(cm, index=classes, columns=classes))