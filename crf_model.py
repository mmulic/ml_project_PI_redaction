import pandas as pd
import numpy as np
import ast
import time
import psutil
import os
import joblib

from transformers import AutoTokenizer
from sklearn_crfsuite import CRF
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_recall_fscore_support,
    f1_score as sk_f1_score
)

from collections import defaultdict

from seqeval.metrics import (
    classification_report as seq_classification_report,
    f1_score,
    precision_score,
    recall_score
)

process = psutil.Process(os.getpid())


def log_time_and_cpu(start_time, start_cpu, message):

    elapsed = time.time() - start_time

    end_cpu = process.cpu_times()

    cpu_used = (
        (end_cpu.user - start_cpu.user)
        + (end_cpu.system - start_cpu.system)
    )

    cpu_percent = (
        (cpu_used / elapsed) * 100
        if elapsed > 0 else 0
    )

    print(
        f"[TIME] {message}: "
        f"{elapsed:.2f} sec | "
        f"CPU time: {cpu_used:.2f} sec | "
        f"Approx CPU usage: {cpu_percent:.2f}%"
    )

    return time.time(), process.cpu_times()


global_start = time.time()
global_cpu = process.cpu_times()


print("Loading tokenizer...")

start = time.time()
start_cpu = process.cpu_times()

tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-multilingual-cased"
)

start, start_cpu = log_time_and_cpu(
    start,
    start_cpu,
    "Tokenizer loaded"
)

def flatten(y):
    return [label for seq in y for label in seq]

# =========================================================
# HELPER: CONVERT SPANS -> BIO LABELS
# =========================================================
def create_bio_labels(text, tokens, offsets, span_labels):

    labels = ["O"] * len(tokens)

    for start_span, end_span, label in span_labels:

        inside = False

        for i, (tok_start, tok_end) in enumerate(offsets):

            if tok_start >= end_span or tok_end <= start_span:
                continue

            if tok_start >= start_span and tok_end <= end_span:

                if not inside:
                    labels[i] = f"B-{label}"
                    inside = True
                else:
                    labels[i] = f"I-{label}"

    return labels

# =========================================================
# FEATURE EXTRACTION
# =========================================================
def token2features(tokens, i):

    token = tokens[i]

    features = {
        'bias': 1.0,
        'word.lower()': token.lower(),
        'word.isupper()': token.isupper(),
        'word.istitle()': token.istitle(),
        'word.isdigit()': token.isdigit(),
        'prefix-1': token[:1],
        'prefix-2': token[:2],
        'suffix-1': token[-1:],
        'suffix-2': token[-2:],
    }

    if i > 0:
        prev = tokens[i - 1]

        features.update({
            '-1:word.lower()': prev.lower(),
        })

    else:
        features['BOS'] = True

    if i < len(tokens) - 1:

        nxt = tokens[i + 1]

        features.update({
            '+1:word.lower()': nxt.lower(),
        })

    else:
        features['EOS'] = True

    return features


def sent2features(tokens):
    return [token2features(tokens, i)
            for i in range(len(tokens))]

# =========================================================
# PREPROCESS DATASET
# =========================================================
def preprocess(df, name="dataset"):

    print(f"\nPreprocessing {name}...")

    start = time.time()
    start_cpu = process.cpu_times()

    X, y, languages = [], [], []

    for idx, row in df.iterrows():

        if idx % 100 == 0:
            print(f"  Processed {idx} rows...")

        text = row["source_text"]

        span_labels = ast.literal_eval(
            row["span_labels"]
        )

        encoding = tokenizer(
            text,
            return_offsets_mapping=True,
            add_special_tokens=False,
            truncation=True
        )

        tokens = tokenizer.convert_ids_to_tokens(
            encoding["input_ids"]
        )

        offsets = encoding["offset_mapping"]

        labels = create_bio_labels(
            text,
            tokens,
            offsets,
            span_labels
        )

        X.append(sent2features(tokens))
        y.append(labels)
        languages.append(row["language"])

    log_time_and_cpu(
        start,
        start_cpu,
        f"{name} preprocessing done"
    )

    return X, y, languages

def strip_bio(label):
    if label.startswith("B-") or label.startswith("I-"):
        return label[2:]
    return label

print("\nLoading CSV files...")

start = time.time()
start_cpu = process.cpu_times()

train_df = pd.read_csv("group_training.csv")
val_df = pd.read_csv("group_validation.csv")
test_df = pd.read_csv("group_testing.csv")

start, start_cpu = log_time_and_cpu(
    start,
    start_cpu,
    "CSV loading done"
)


X_train, y_train, lang_train = preprocess(
    train_df,
    "train"
)

X_val, y_val, lang_val = preprocess(
    val_df,
    "validation"
)

X_test, y_test, lang_test = preprocess(
    test_df,
    "test"
)

# =========================================================
# TRAINING (SINGLE RUN, ITERATIONS VARY)
# =========================================================
print("\nTraining CRF model (single run, 30 iterations)...")

start = time.time()
start_cpu = process.cpu_times()

crf = CRF(
    algorithm='lbfgs',
    max_iterations=40,
    epsilon=1e-5,
    c1=0.3,
    c2=0.5,
    all_possible_transitions=True,
    verbose=True
)

crf.fit(X_train, y_train)

joblib.dump(crf, "v2_best_crf_model.pkl")

log_time_and_cpu(
    start,
    start_cpu,
    "CRF training complete"
)


crf = joblib.load("best_crf_model.pkl")

print("\nRunning predictions...")

y_train_pred = crf.predict(X_train)
y_val_pred = crf.predict(X_val)
y_test_pred = crf.predict(X_test)

# =========================================================
# METRICS HELPERS
# =========================================================
def print_metrics(y_true, y_pred, title="Dataset"):

    print("\n========================================")
    print(f"METRICS: {title}")
    print("========================================")

    # flatten
    y_true_flat = flatten(y_true)
    y_pred_flat = flatten(y_pred)

    # 🔥 merge B- and I- labels
    y_true_flat = [strip_bio(l) for l in y_true_flat]
    y_pred_flat = [strip_bio(l) for l in y_pred_flat]

    print("\nAccuracy:")
    print(accuracy_score(y_true_flat, y_pred_flat))

    print("\nToken-level Classification Report:")
    print(classification_report(
        y_true_flat,
        y_pred_flat,
        zero_division=0
    ))

    labels = sorted(list(set(y_true_flat) | set(y_pred_flat)))

    precision, recall, f1_vals, support = precision_recall_fscore_support(
        y_true_flat,
        y_pred_flat,
        labels=labels,
        zero_division=0
    )

    print("\nPer-class metrics:")
    for l, p, r, f, s in zip(labels, precision, recall, f1_vals, support):
        print(f"{l:15} | P={p:.4f} | R={r:.4f} | F1={f:.4f} | Support={s}")

    filtered_f1 = [f for l, f in zip(labels, f1_vals) if l != "O"]
    print(f"\nMacro F1 (no O): {np.mean(filtered_f1):.4f}")


def evaluate_by_language(y_true, y_pred, languages, title="Dataset"):

    print("\n========================================")
    print(f"LANGUAGE-WISE METRICS: {title}")
    print("========================================")

    from collections import defaultdict

    lang_map = defaultdict(lambda: {"true": [], "pred": [], "count": 0})

    for yt, yp, lang in zip(y_true, y_pred, languages):

        if lang is None:
            lang = "unknown"
        else:
            lang = str(lang).strip().lower()

        if lang in ["en", "eng", "english"]:
            lang = "english"
        elif lang in ["es", "spa", "spanish"]:
            lang = "spanish"

        lang_map[lang]["true"].append(yt)
        lang_map[lang]["pred"].append(yp)
        lang_map[lang]["count"] += 1

    print("\nDetected languages:")
    for lang, data in lang_map.items():
        print(f"  - {lang}: {data['count']} samples")

    print("\n========================================")

    for lang, data in sorted(lang_map.items()):

        print(f"\n----- Language: {lang} -----")

        print_metrics(
            data["true"],
            data["pred"],
            title=f"{title} | {lang}"
        )

# =========================================================
# EVALUATION: 
# =========================================================
print_metrics(y_train, y_train_pred, "TRAIN")
print_metrics(y_val, y_val_pred, "VALIDATION")
print_metrics(y_test, y_test_pred, "TEST")

evaluate_by_language(y_train, y_train_pred, lang_train, "TRAIN")
evaluate_by_language(y_val, y_val_pred, lang_val, "VALIDATION")
evaluate_by_language(y_test, y_test_pred, lang_test, "TEST")

log_time_and_cpu(
    global_start,
    global_cpu,
    "TOTAL PIPELINE TIME"
)


