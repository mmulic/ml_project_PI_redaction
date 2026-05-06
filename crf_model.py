import pandas as pd
import numpy as np
import ast
import time
from transformers import AutoTokenizer
from sklearn_crfsuite import CRF
from sklearn.metrics import classification_report, accuracy_score
from collections import defaultdict
from seqeval.metrics import classification_report as seq_classification_report
from seqeval.metrics import f1_score, precision_score, recall_score

# -----------------------------
# Global timer helper
# -----------------------------
def log_time(start, message):
    elapsed = time.time() - start
    print(f"[TIME] {message}: {elapsed:.2f} sec")
    return time.time()


global_start = time.time()

# -----------------------------
# Load tokenizer
# -----------------------------
print("Loading tokenizer...")
start = time.time()

tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

start = log_time(start, "Tokenizer loaded")


# -----------------------------
# Helper: Convert spans → BIO labels
# -----------------------------
def create_bio_labels(text, tokens, offsets, span_labels):
    labels = ["O"] * len(tokens)

    for start_span, end_span, label in span_labels:
        for i, (tok_start, tok_end) in enumerate(offsets):
            if tok_start >= end_span or tok_end <= start_span:
                continue

            if tok_start >= start_span and tok_end <= end_span:
                if labels[i] == "O":
                    labels[i] = f"B-{label}"
                else:
                    labels[i] = f"I-{label}"

    return labels


# -----------------------------
# Feature extraction
# -----------------------------
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
        prev = tokens[i-1]
        features.update({
            '-1:word.lower()': prev.lower(),
        })
    else:
        features['BOS'] = True

    if i < len(tokens)-1:
        nxt = tokens[i+1]
        features.update({
            '+1:word.lower()': nxt.lower(),
        })
    else:
        features['EOS'] = True

    return features


def sent2features(tokens):
    return [token2features(tokens, i) for i in range(len(tokens))]


# -----------------------------
# Preprocess dataset
# -----------------------------
def preprocess(df, name="dataset"):
    print(f"\nPreprocessing {name}...")
    start = time.time()

    X, y, languages = [], [], []

    for idx, row in df.iterrows():
        if idx % 100 == 0:
            print(f"  Processed {idx} rows...")

        text = row["source_text"]
        span_labels = ast.literal_eval(row["span_labels"])

        encoding = tokenizer(
            text,
            return_offsets_mapping=True,
            add_special_tokens=False,
            truncation=True
        )

        tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])
        offsets = encoding["offset_mapping"]

        labels = create_bio_labels(text, tokens, offsets, span_labels)

        X.append(sent2features(tokens))
        y.append(labels)
        languages.append(row["language"])

    log_time(start, f"{name} preprocessing done")
    return X, y, languages


# -----------------------------
# Load data
# -----------------------------
print("\nLoading CSV files...")
start = time.time()

train_df = pd.read_csv("group_training.csv")
val_df = pd.read_csv("group_validation.csv")
test_df = pd.read_csv("group_testing.csv")

start = log_time(start, "CSV loading done")


# -----------------------------
# Preprocessing
# -----------------------------
X_train, y_train, lang_train = preprocess(train_df, "train")
X_val, y_val, lang_val = preprocess(val_df, "validation")
X_test, y_test, lang_test = preprocess(test_df, "test")


# -----------------------------
# Train CRF model
# -----------------------------
print("\nTraining CRF model...")
start = time.time()

crf = CRF(
    algorithm='lbfgs',
    max_iterations=75, #75 or 125       
    c1=0.3,                      # L1 regularisation
    c2=0.5,                      # L2 regularisation
    all_possible_transitions=True,
    verbose=True
)

crf.fit(X_train, y_train)

start = log_time(start, "CRF training complete")


# -----------------------------
# Prediction
# -----------------------------
print("\nRunning predictions...")
start = time.time()

y_pred = crf.predict(X_test)

start = log_time(start, "Prediction complete")


# -----------------------------
# Flatten for metrics
# -----------------------------
def flatten(y):
    return [label for seq in y for label in seq]

y_test_flat = flatten(y_test)
y_pred_flat = flatten(y_pred)


# -----------------------------
# Metrics
# -----------------------------
print("\nEvaluating metrics...")
start = time.time()

print("\n=== Overall Metrics ===")
print("Accuracy:", accuracy_score(y_test_flat, y_pred_flat))
print(classification_report(y_test_flat, y_pred_flat))

print("\n=== SeqEval Metrics ===")
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))

print(seq_classification_report(y_test, y_pred))

start = log_time(start, "Evaluation complete")


# -----------------------------
# Language-wise metrics
# -----------------------------
def evaluate_by_language(y_true, y_pred, languages):
    print("\nEvaluating by language...")
    start = time.time()

    lang_map = defaultdict(lambda: {"true": [], "pred": []})

    for yt, yp, lang in zip(y_true, y_pred, languages):
        lang_map[lang]["true"].append(yt)
        lang_map[lang]["pred"].append(yp)

    for lang in lang_map:
        print(f"\n=== Language: {lang} ===")
        yt = lang_map[lang]["true"]
        yp = lang_map[lang]["pred"]

        print("Precision:", precision_score(yt, yp))
        print("Recall:", recall_score(yt, yp))
        print("F1:", f1_score(yt, yp))
        print(seq_classification_report(yt, yp))

    log_time(start, "Language evaluation complete")


evaluate_by_language(y_test, y_pred, lang_test)

# -----------------------------
# Total runtime
# -----------------------------
log_time(global_start, "TOTAL PIPELINE TIME")
