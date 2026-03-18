# Sentiment Analysis Tutorial
### Python Machine Learning – Tutorial Series 
**Author:** Dr. Priya Lakshmi Narayanan

---

## Overview

`demo_sentiment_analysis.py` is a self-contained tutorial script that walks through the full pipeline for building a text sentiment classifier in Python — from raw text cleaning through to a persisted, reloadable model. It is written as a teaching resource and runs immediately with no external data download required.

The script uses a built-in corpus of 16 IMDB-style movie reviews and demonstrates exactly how to scale up to the full 50,000-review IMDB dataset using out-of-core (memory-efficient) learning.

---

## Learning Objectives

By working through this script you will learn how to:

1. Build a text cleaning and preprocessing pipeline (HTML removal, emoticon handling, lowercasing)
2. Convert raw text into numeric features using Bag-of-Words (BoW)
3. Improve representations with TF-IDF weighting
4. Train a sentiment classifier using Logistic Regression inside a `sklearn` Pipeline
5. Implement out-of-core (batch-wise) learning with `HashingVectorizer` + `SGDClassifier` for datasets too large to fit in memory
6. Persist and reload a trained model using `pickle`

---

## Requirements

| Package | Minimum version |
|---|---|
| Python | 3.8+ |
| scikit-learn | 1.1+ |
| numpy | any recent |
| matplotlib | any recent |

Install all dependencies in one command:

```bash
pip install scikit-learn numpy matplotlib
```

> **Note on scikit-learn ≥ 1.1:** The script uses `loss='log_loss'` for `SGDClassifier` (the older `loss='log'` was deprecated in 1.1 and removed in 1.3). The script has been updated to use the current API throughout.

---

## Usage

```bash
python demo_sentiment_analysis.py
```

The script runs all seven sections sequentially and prints annotated output to the terminal. No command-line arguments are required.

---

## Script Structure

| Section | Description |
|---|---|
| **1 – Corpus** | 16 inline IMDB-style movie reviews (8 positive, 8 negative). Notes how to load the full 50k IMDB dataset. |
| **2 – Text Cleaning** | `clean_text()` function: strips HTML tags, extracts emoticons, lowercases, removes punctuation, re-appends emoticons. |
| **3 – Bag-of-Words** | `CountVectorizer` with unigrams and unigram+bigram configurations. Vocabulary inspection. |
| **4 – TF-IDF** | Two-step (`CountVectorizer` + `TfidfTransformer`) and one-step (`TfidfVectorizer`) approaches. IDF value analysis. |
| **5 – Sentiment Classifier** | `TfidfVectorizer` + `LogisticRegression` wrapped in a `Pipeline`. 5-fold cross-validation, n-gram range comparison, prediction on held-out examples. |
| **6 – Out-of-Core Learning** | `HashingVectorizer` + `SGDClassifier.partial_fit()` for batch processing. Accuracy-vs-batches plot saved as PNG. |
| **7 – Model Persistence** | Save and reload the trained pipeline with `pickle`. Verification that reloaded predictions match the original. |

---

## Output Files

Running the script produces the following files:

```
ch08_outofcore_accuracy.png        ← Accuracy vs batch number plot (Section 6)
model/
└── sentiment_pipeline.pkl         ← Saved trained model (Section 7)
```

---

## Saved Model

The trained `TfidfVectorizer + LogisticRegression` pipeline is serialised with `pickle` and saved to:

```
model/sentiment_pipeline.pkl
```

The directory `model/` is created automatically by the script if it does not already exist. To load and use the saved model in a separate script:

```python
import pickle
from demo_sentiment_analysis import clean_text   # reuse the cleaning function

# Load the model
with open("model/sentiment_pipeline.pkl", "rb") as f:
    model = pickle.load(f)

# Run inference
review = "An absolutely brilliant film. Superb performances throughout."
prediction = model.predict([clean_text(review)])
probability = model.predict_proba([clean_text(review)])

print("POSITIVE" if prediction[0] == 1 else "NEGATIVE")
print(f"Confidence: {probability[0][prediction[0]]:.3f}")
```

> **Compatibility note:** The `.pkl` file is tied to the version of scikit-learn used to save it. If you upgrade scikit-learn, re-run the script to regenerate the model file.

---

## Scaling to the Full IMDB Dataset

The script includes two paths to the full 50,000-review IMDB dataset:

**Option A — Hugging Face datasets library:**
```bash
pip install datasets
python -c "from datasets import load_dataset; d = load_dataset('imdb')"
```

**Option B — Stanford download:**
```
https://ai.stanford.edu/~amaas/data/sentiment/
```

Once downloaded, replace the inline `REVIEWS` list with the out-of-core streaming pattern shown in Section 6. The `HashingVectorizer + SGDClassifier.partial_fit()` pattern processes the data one batch at a time and targets > 88% accuracy on the 45k training set.

---

## Exercises

The script ends with four progressive exercises:

| Exercise | Topic |
|---|---|
| 8.1 | Extend the corpus; inspect the most informative features via `clf.coef_` |
| 8.2 | Compare unigram, bigram, trigram CV accuracy |
| 8.3 | Add a Porter Stemmer preprocessing step and measure its impact |
| 8.4 *(Advanced)* | Full out-of-core pipeline on the 50k IMDB dataset; plot accuracy vs reviews processed |

---

## Project Structure

```
.
├── demo_sentiment_analysis.py       ← Main tutorial script
├── README.md                        ← This file
├── ch08_outofcore_accuracy.png      ← Generated: out-of-core accuracy plot
└── model/
    └── sentiment_pipeline.pkl       ← Generated: saved trained model
```

