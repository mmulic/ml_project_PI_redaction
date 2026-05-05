# PII Redaction: 3-Layer Pipeline with SpaCy and BERT

This project compares two approaches for detecting and redacting personally identifiable information (PII):
1. **SpaCy**: A pre-trained named entity recognition model
2. **BERT**: A fine-tuned transformer model trained on your specific PII data

## Overview

The pipeline has 4 main scripts that you need to run in order:

### 1. `data_preparation.py`
Converts your raw CSV data into a format that BERT can understand.

**What it does:**
- Takes the `group_training.csv`, `group_validation.csv`, and `group_testing.csv` files
- Parses the `span_labels` (which contain PII annotations)
- Converts character-level annotations into token-level BIO labels
- Saves as JSONL files (one example per line)

**BIO Format Explanation:**
- `B-PERSON` = Beginning of a person's name
- `I-PERSON` = Inside a person's name (continuation)
- `O` = Outside any PII entity (normal text)

**Run:**
```bash
python data_preparation.py
```

**Output:**
- `train_bio.jsonl` (training data with BIO labels)
- `val_bio.jsonl` (validation data)
- `test_bio.jsonl` (test data)

---

### 2. `spacy_baseline.py`
Runs a pre-trained SpaCy model to detect PII.

**What it does:**
- Loads the `en_core_web_sm` model (pre-trained on general English text)
- Runs it on your test set
- Detects standard entities: PERSON, ORG, GPE, LOC, DATE, etc.
- Compares predictions against your ground truth labels
- Calculates accuracy, precision, recall, and F1 score

**Why this first?**
SpaCy is the baseline—it's fast and requires no training. If BERT is better, we know the effort was worth it.

**Run:**
```bash
python spacy_baseline.py
```

**If this fails with "model not found":**
```bash
python -m spacy download en_core_web_sm
```

**Output:**
- `spacy_predictions.jsonl` (predictions for each example)
- `spacy_metrics.json` (performance metrics)

---

### 3. `bert_training.py`
Fine-tunes a BERT model on your PII data.

**What it does:**
- Loads the prepared data (BIO format)
- Uses BERT's tokenizer to split text into subwords
- Aligns the BIO labels with BERT's subword tokens
- Fine-tunes BERT for 3 epochs
- Saves the best model

**Why BERT?**
BERT understands context and meaning, so it should catch more nuanced PII than SpaCy.

**Important concepts:**
- **Tokenization**: BERT splits words into smaller pieces. Example: "unbelievable" → ["un", "##believable"]
- **Subword alignment**: Only the first subword of a word gets the real label; the rest are ignored in training
- **Epochs**: Training passes over the data. 3 passes is usually enough to learn patterns without overfitting

**Run:**
```bash
python bert_training.py
```

**This will take 5-15 minutes depending on your computer/GPU.**

**What to watch for:**
- It should print "CUDA" in the device line if you have a GPU (much faster)
- Look for validation metrics improving over epochs
- The script saves only the best model to avoid wasting space

**Output:**
- `bert_model/` directory containing:
  - `pytorch_model.bin` (the trained model weights)
  - `config.json` (model configuration)
  - `tokenizer_config.json` (tokenizer settings)
  - `label2id.json` and `id2label.json` (label mappings)

---

### 4. `bert_inference.py`
Runs the fine-tuned BERT model and compares it with SpaCy.

**What it does:**
- Loads your trained BERT model
- Runs it on the test set
- Calculates metrics (accuracy, precision, recall, F1)
- **Compares side-by-side with SpaCy results**
- Shows which model is better and makes recommendations

**Run:**
```bash
python bert_inference.py
```

**Output:**
- `bert_predictions.jsonl` (predictions for each example)
- `bert_metrics.json` (performance metrics)
- `model_comparison.json` (head-to-head comparison)
- **Nice comparison table printed to console**

---

## Running the Full Pipeline

```bash
# Step 1: Prepare data
python data_preparation.py

# Step 2: Get SpaCy baseline
python spacy_baseline.py

# Step 3: Train BERT (takes 5-15 minutes)
python bert_training.py

# Step 4: Compare models
python bert_inference.py
```

---

## Expected Results

After running all scripts, you'll see output like:

```
================================================================================
MODEL COMPARISON: SPACY vs BERT
================================================================================

Metric          SpaCy                BERT                 Winner
--------------------------------------------------------------------------------
accuracy        0.8234               0.9102               BERT ✓
precision       0.7891               0.8956               BERT ✓
recall          0.7234               0.8801               BERT ✓
f1              0.7554               0.8877               BERT ✓
================================================================================

SUMMARY:
🎯 BERT is the winner (4 out of 4 metrics)

RECOMMENDATIONS:
✓ Use BERT for production PII detection
✓ Consider ensemble: run both and flag when they disagree
```

---

## Understanding the Architecture

### Why 3 Layers?

1. **Regex** (not implemented yet, but could be)
   - Fastest but most limited
   - Best for: exact patterns like emails, phone numbers, SSNs

2. **SpaCy NER** (Layer 1)
   - Pre-trained on general text
   - Detects: PERSON, ORG, LOC, DATE
   - Good baseline, no training required

3. **BERT** (Layer 2)
   - Fine-tuned on your specific data
   - Understands context and nuance
   - Best for: your domain-specific PII patterns

### Why BERT Usually Wins

SpaCy is trained on general English text from Wikipedia and news. Your PII data has different patterns. BERT adapts to learn YOUR specific PII patterns, so it should perform better.

---

## Interpreting Metrics

- **Accuracy**: % of tokens correctly labeled (can be misleading if PII is rare)
- **Precision**: Of all PII we predicted, how many were actually PII?
- **Recall**: Of all actual PII, how many did we find?
- **F1**: Balanced score between precision and recall (usually what you care about most)

For PII detection, **recall** is often more important than precision—it's better to over-report (flag false positives) than to miss real PII.

---

## Common Issues

### "CUDA not available"
Your computer doesn't have an NVIDIA GPU. Training will be slower but still works. Typical times:
- With GPU: 5-10 minutes
- With CPU: 30-60 minutes

### "Model not found" for SpaCy
Run: `python -m spacy download en_core_web_sm`

### Low metrics for BERT
- Not enough training data (your dataset is quite small)
- Too few epochs (try increasing to 5 or 10 in bert_training.py)
- Data quality issues (labels might be inconsistent)

### BERT overfitting
If train F1 is 0.95 but val F1 is 0.75, you're overfitting. Add more regularization:
- Increase `weight_decay` in bert_training.py
- Add dropout
- Use fewer epochs

---

## Next Steps After Comparison

1. **If BERT is better**: Use it in production. Integrate bert_inference.py into your app.
2. **If SpaCy is better**: Stick with SpaCy (it's faster and simpler).
3. **If both are comparable**: Use SpaCy (simpler, faster, no GPU needed).
4. **Fine-tune further**: Add domain-specific regex patterns before running models.
5. **Ensemble**: Run both models and combine their predictions for highest recall.

---

## Code Walkthrough by Layer

### data_preparation.py
- **Key function**: `character_spans_to_bio_labels()` - Converts character positions to token labels
- **Why this matters**: BERT works with tokens, not characters. We have to align them carefully.

### spacy_baseline.py
- **Key function**: `extract_spacy_bio_labels()` - Extracts entities from SpaCy doc
- **Why this matters**: SpaCy gives us entity objects, we convert to BIO format for fair comparison

### bert_training.py
- **Key function**: `tokenize_and_align_labels()` - Aligns BIO labels with BERT's subword tokens
- **Why this matters**: BERT's tokenizer splits words differently than we expect
- **Tricky part**: When a word becomes multiple subwords, only the first gets the label, rest get -100

### bert_inference.py
- **Key function**: `run_bert_inference()` - Runs inference and calculates metrics
- **Why this matters**: We need to measure model performance consistently
- **Comparison**: Side-by-side with SpaCy to see which is actually better

---

## Questions You Might Have

**Q: Why do I need to prepare data?**
A: BERT's tokenizer works differently than simple word splitting. We align labels to handle this.

**Q: Why does BERT training take time?**
A: It's learning to recognize patterns in your data. More data = more time but better performance.

**Q: Can I use a GPU to speed this up?**
A: Yes! The scripts auto-detect GPU (CUDA). If you have an NVIDIA GPU, it'll be 3-5x faster.

**Q: What if my results are bad?**
A: Common fixes:
- More training data
- Fix label inconsistencies in your data
- Try a larger BERT model (bert-base-cased, bert-large-uncased)
- Add data augmentation

**Q: How do I use this in production?**
A: You'd load the BERT model and wrap it in an API. See bert_inference.py for the pattern.

---

Good luck! 🚀
