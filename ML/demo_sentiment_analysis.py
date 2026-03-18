"""
=============================================================================
Python Machine Learning – Tutorial Series  (Updated for Modern Packages)
Ch 8: Applying Machine Learning to Sentiment Analysis
Dr. Priya Lakshmi Narayanan
=============================================================================
LEARNING OBJECTIVES
  1. Build a text cleaning and preprocessing pipeline
  2. Convert text to numeric features using Bag-of-Words
  3. Improve with TF-IDF weighting
  4. Train a sentiment classifier on movie reviews
  5. Implement out-of-core (memory-efficient) learning for large datasets
  6. Persist and load trained models

UPDATED APIs:
  TfidfTransformer(smooth_idf=False) → smooth_idf=True now default, both valid
  SGDClassifier(loss='log')          → loss='log_loss'  (sklearn ≥1.1)
  All deprecated params removed

PACKAGES: pip install scikit-learn numpy pandas matplotlib
=============================================================================
"""

import os
import re
import pickle
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import (CountVectorizer,
                                              TfidfTransformer,
                                              TfidfVectorizer,
                                              HashingVectorizer)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, classification_report


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: WORKING TEXT CORPUS
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("SECTION 1: Text Corpus (IMDB-style movie reviews)")
print("=" * 65)

print("""
The book uses the IMDB 50k movie review dataset (download required).
This tutorial uses a built-in corpus that runs immediately, and shows
exactly how to scale up to the full 50k dataset with out-of-core learning.

For the full IMDB dataset, run:
  python -c "from datasets import load_dataset; d = load_dataset('imdb'); ..."
or download from: https://ai.stanford.edu/~amaas/data/sentiment/
""")

# Inline corpus — representative of real review text
REVIEWS = [
    ("The film was absolutely magnificent. The acting was superb and the "
     "plot kept me gripped from start to finish. A must-watch!", 1),
    ("Complete waste of time. The story made no sense and the acting was "
     "wooden throughout. I nearly fell asleep.", 0),
    ("An outstanding piece of cinema. Beautifully shot, brilliantly acted "
     "and emotionally resonant. One of the best films I have seen.", 1),
    ("Terrible. The dialogue was cringe-worthy and the pacing was dreadful. "
     "I cannot believe this got good reviews.", 0),
    ("Loved every moment. The chemistry between the leads was electric and "
     "the story was both funny and moving.", 1),
    ("Boring and predictable. Nothing you haven't seen a hundred times before. "
     "The special effects were cheap and unconvincing.", 0),
    ("A genuinely thrilling experience. The director builds tension masterfully "
     "and the ending is completely unexpected.", 1),
    ("Poor casting decisions and a script full of plot holes. The director "
     "seemed to have no idea what tone to aim for.", 0),
    ("Visually stunning and narratively bold. The performances are career-best "
     "for everyone involved. Highly recommended.", 1),
    ("Deeply disappointing given the talent involved. The film drags and "
     "the climax feels rushed and unsatisfying.", 0),
    ("Funny, poignant and brilliantly written. I laughed and cried in equal "
     "measure. This is exactly what cinema should be.", 1),
    ("An incoherent mess from beginning to end. The editing was chaotic "
     "and the music choices were baffling.", 0),
    ("Brilliantly crafted with superb attention to detail. Every frame "
     "feels deliberate and meaningful.", 1),
    ("The worst film I have seen this year. Poorly acted, badly directed "
     "and completely devoid of originality.", 0),
    ("A heartfelt and genuinely moving story. The performances are natural "
     "and the script is intelligent and witty.", 1),
    ("Tedious and overwrought. The film mistakes length for depth and "
     "self-importance for artistry.", 0),
]

texts  = [r[0] for r in REVIEWS]
labels = [r[1] for r in REVIEWS]
print(f"Corpus: {len(texts)} reviews  "
      f"({sum(labels)} positive, {len(labels)-sum(labels)} negative)")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: TEXT CLEANING
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 2: Text Cleaning & Preprocessing")
print("=" * 65)


def clean_text(text: str) -> str:
    """
    Clean a single review string:
      1. Remove HTML tags (e.g. <br />, <b>)
      2. Extract emoticons before stripping punctuation
      3. Lowercase and remove non-alphanumeric characters
      4. Re-append emoticons (they carry sentiment signal)
    """
    # Remove HTML markup
    text = re.sub(r"<[^>]+>", "", text)
    # Extract emoticons: :) :( :D ;-) etc.
    emoticons = re.findall(r"(?::|;|=)(?:-)?(?:\)|\(|D|P)", text)
    # Lowercase, remove non-word characters
    text = re.sub(r"[^\w\s]", " ", text.lower())
    # Re-append emoticons (normalised, dash removed)
    text = text + " " + " ".join(emoticons).replace("-", "")
    return text.strip()


cleaned = [clean_text(t) for t in texts]

print("Sample before/after cleaning:")
for orig, clean in zip(texts[:3], cleaned[:3]):
    print(f"  IN : {orig[:75]}...")
    print(f"  OUT: {clean[:75]}...")
    print()

# Demonstrate individual cleaning steps
sample = "The film <b>was</b> GREAT! :) Really amazing... check www.imdb.com"
print(f"HTML + punctuation demo:")
print(f"  IN : {sample}")
print(f"  OUT: {clean_text(sample)}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: BAG-OF-WORDS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 3: Bag-of-Words (BoW) Representation")
print("=" * 65)

print("""
Bag-of-Words converts text to a numeric vector by counting word occurrences.
The "bag" metaphor: word ORDER is discarded; only frequencies matter.

CountVectorizer:
  fit(corpus)       → learn the vocabulary (word → column index mapping)
  transform(corpus) → count word frequencies per document
  Result: sparse matrix of shape [n_documents × vocab_size]

Key parameters:
  ngram_range=(1,2)  include both single words and two-word phrases
  min_df=2           ignore words appearing in fewer than 2 documents
  max_df=0.9         ignore words in more than 90% of documents (stopwords)
  stop_words='english' remove common words (the, a, is, ...)
""")

# Unigrams only
cv_uni = CountVectorizer(ngram_range=(1, 1), min_df=1)
X_bow = cv_uni.fit_transform(cleaned)
print(f"Unigram vocabulary size:      {len(cv_uni.vocabulary_)}")
print(f"Feature matrix (dense) shape: {X_bow.shape}")

# Unigrams + bigrams
cv_bi = CountVectorizer(ngram_range=(1, 2), min_df=1)
X_bow_bi = cv_bi.fit_transform(cleaned)
print(f"Unigram + bigram vocab size:  {len(cv_bi.vocabulary_)}")

# Inspect vocabulary
print("\nSample vocabulary entries:")
vocab_sorted = sorted(cv_uni.vocabulary_.items(), key=lambda x: x[1])
for word, idx in vocab_sorted[:15]:
    print(f"  '{word}' → column {idx}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: TF-IDF
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 4: TF-IDF — Term Frequency × Inverse Document Frequency")
print("=" * 65)

print("""
Problem with raw counts: common words (film, the, and) dominate, but
they carry little sentiment signal.

TF-IDF down-weights words that appear in many documents:
  TF(t, d)  = count of term t in document d  (term frequency)
  IDF(t)    = log( (1+N) / (1+df(t)) ) + 1   (sklearn's smooth formula)
                N = number of documents
                df(t) = documents containing term t

  TF-IDF(t, d) = TF(t, d) × IDF(t)

Words appearing in EVERY document get low IDF ≈ 0 → down-weighted.
Words appearing in FEW documents get high IDF → up-weighted.

After TF-IDF, rows are L2-normalised (unit vectors) for cosine similarity.
""")

# Method 1: CountVectorizer + TfidfTransformer (two-step)
cv = CountVectorizer(ngram_range=(1, 2), min_df=1, stop_words="english")
tfidf_t = TfidfTransformer(use_idf=True, norm="l2", smooth_idf=True)
X_bow_step = cv.fit_transform(cleaned)
X_tfidf_step = tfidf_t.fit_transform(X_bow_step)

# Method 2: TfidfVectorizer (one-step, equivalent)
tfidf_v = TfidfVectorizer(ngram_range=(1, 2), min_df=1,
                            stop_words="english", use_idf=True, norm="l2")
X_tfidf = tfidf_v.fit_transform(cleaned)

print(f"TF-IDF matrix shape:  {X_tfidf.shape}")
print(f"Non-zero entries:     {X_tfidf.nnz}  "
      f"({100*X_tfidf.nnz/(X_tfidf.shape[0]*X_tfidf.shape[1]):.1f}% dense)")

# IDF values — high means rare (informative)
idf_vals = tfidf_v.idf_
feature_names = tfidf_v.get_feature_names_out()
top_idf_idx = np.argsort(idf_vals)[::-1][:10]
print("\nHighest IDF words (rarest / most informative):")
for idx in top_idf_idx:
    print(f"  '{feature_names[idx]}':  IDF={idf_vals[idx]:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: SENTIMENT CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 5: Training a Sentiment Classifier")
print("=" * 65)

# Pipeline: clean → TF-IDF → Logistic Regression
def build_sentiment_pipeline(C=1.0, ngram=(1, 2)):
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=ngram,
                                   stop_words="english",
                                   min_df=1, max_df=0.95)),
        ("clf",   LogisticRegression(C=C, max_iter=500, random_state=42))
    ])

# Cross-validated accuracy on our corpus
pipe_lr = build_sentiment_pipeline(C=1.0)
cv_scores = cross_val_score(pipe_lr, cleaned, labels, cv=5, scoring="accuracy")
print(f"Logistic Regression + TF-IDF  5-fold CV: "
      f"{cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# n-gram comparison
print("\nn-gram range comparison (5-fold CV accuracy):")
for ngram in [(1, 1), (1, 2), (2, 2), (1, 3)]:
    pipe = build_sentiment_pipeline(ngram=ngram)
    cv = cross_val_score(pipe, cleaned, labels, cv=5).mean()
    print(f"  ngram_range={ngram}  CV={cv:.3f}")

# Train final model on all data
pipe_lr.fit(cleaned, labels)
y_pred_all = pipe_lr.predict(cleaned)
print(f"\nFinal model trained on all {len(cleaned)} reviews")
print(f"Training accuracy: {accuracy_score(labels, y_pred_all):.3f}")

# Predict new examples
test_reviews = [
    "What a brilliant and moving film. Absolutely loved it.",
    "Dreadful acting and a confusing plot. Complete waste of money.",
    "It was okay, nothing special but not terrible either.",
]
print("\nPredicting new reviews:")
for review in test_reviews:
    pred = pipe_lr.predict([clean_text(review)])[0]
    prob = pipe_lr.predict_proba([clean_text(review)])[0]
    label = "POSITIVE ✓" if pred == 1 else "NEGATIVE ✗"
    print(f"  [{label}  P={prob[pred]:.3f}]  {review[:60]}...")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: OUT-OF-CORE LEARNING (for large datasets)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 6: Out-of-Core Learning (Memory-Efficient)")
print("=" * 65)

print("""
The full IMDB dataset has 50,000 reviews — too large to load into memory
as a dense TF-IDF matrix. Out-of-core learning processes data in batches.

Key components:
  HashingVectorizer: no vocabulary to store → constant memory usage
      Hashes each word to a column index (no lookup table needed).
      trade-off: no inverse transform (can't inspect features)

  SGDClassifier.partial_fit(): update model weights batch by batch.
      Unlike .fit(), it ADDS to existing weights rather than resetting.

  Updated: loss='log_loss' (sklearn ≥1.1 renames 'log' → 'log_loss')
""")

# Simulate batched processing on our small corpus
# (In production this would iterate over file chunks)
def get_minibatch(texts, labels, batch_size=4):
    """Yield mini-batches from a list."""
    for start in range(0, len(texts), batch_size):
        yield texts[start:start + batch_size], labels[start:start + batch_size]


hv = HashingVectorizer(
    n_features=2 ** 10,       # 1024 hash buckets (small for demo)
    norm="l2",
    alternate_sign=False,     # non-negative values
    stop_words="english")

# Updated: loss='log_loss' instead of deprecated 'log'
sgd = SGDClassifier(
    loss="log_loss",          # logistic regression via SGD
    alpha=1e-4,               # L2 regularisation
    max_iter=1,               # one pass per partial_fit call
    tol=None,                 # no convergence check per batch
    random_state=42)

print("Out-of-core training (batch_size=4):")
all_train_acc = []
for batch_idx, (batch_texts, batch_labels) in enumerate(
        get_minibatch(cleaned, labels, batch_size=4)):
    X_batch = hv.transform([clean_text(t) for t in batch_texts])
    y_batch = np.array(batch_labels)
    sgd.partial_fit(X_batch, y_batch, classes=[0, 1])
    # Evaluate on all data after each batch
    X_all   = hv.transform([clean_text(t) for t in cleaned])
    acc     = accuracy_score(labels, sgd.predict(X_all))
    all_train_acc.append(acc)
    print(f"  Batch {batch_idx + 1}:  acc={acc:.3f}")

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(range(1, len(all_train_acc) + 1), all_train_acc, "o-",
         color="#4FC3F7", markersize=8)
ax.set_xlabel("Batch number"); ax.set_ylabel("Accuracy")
ax.set_title("Out-of-Core SGD — Accuracy vs Batches Processed")
ax.set_ylim([0, 1.05])
plt.tight_layout()
plt.savefig("ch08_outofcore_accuracy.png", dpi=100, bbox_inches="tight")
print("\n  → Saved: ch08_outofcore_accuracy.png")
plt.close()

print("""
For the full IMDB 50k dataset the out-of-core pattern would be:

  def stream_reviews(path, batch_size=1000):
      # Read reviews from disk, one batch at a time
      with open(path, encoding='utf-8') as f:
          batch_texts, batch_labels = [], []
          for line in f:
              text, label = line.rsplit(',', 1)
              batch_texts.append(text); batch_labels.append(int(label))
              if len(batch_texts) == batch_size:
                  yield batch_texts, batch_labels
                  batch_texts, batch_labels = [], []
          if batch_texts:
              yield batch_texts, batch_labels

  for batch_texts, batch_labels in stream_reviews('imdb.csv'):
      X_batch = hv.transform([clean_text(t) for t in batch_texts])
      sgd.partial_fit(X_batch, batch_labels, classes=[0, 1])
""")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: MODEL PERSISTENCE
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("SECTION 7: Saving & Loading the Trained Model")
print("=" * 65)

os.makedirs("model", exist_ok=True)

# Save the pipeline
with open("model/sentiment_pipeline.pkl", "wb") as f:
    pickle.dump(pipe_lr, f)
print("  → Saved: model/sentiment_pipeline.pkl")

# Load and verify
with open("model/sentiment_pipeline.pkl", "rb") as f:
    pipe_loaded = pickle.load(f)

pred_loaded = pipe_loaded.predict([clean_text(test_reviews[0])])
print(f"  Loaded model prediction: "
      f"{'POSITIVE' if pred_loaded[0] == 1 else 'NEGATIVE'}")
print(f"  Original pipeline pred: "
      f"{'POSITIVE' if pipe_lr.predict([clean_text(test_reviews[0])])[0] == 1 else 'NEGATIVE'}")
print("  Match:", pred_loaded[0] == pipe_lr.predict(
    [clean_text(test_reviews[0])])[0])


# ─────────────────────────────────────────────────────────────────────────────
# EXERCISES
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("EXERCISES")
print("=" * 65)
print("""
>>> EXERCISE 8.1
  Extend the corpus to 50+ reviews (write or copy from a review site).
  Compare the top-20 most-informative features (highest |coefficient|) for
  positive vs negative class. Do the words make intuitive sense?
  Hint: pipe_lr.named_steps['clf'].coef_

>>> EXERCISE 8.2
  Compare unigrams, bigrams, and trigrams using 5-fold CV.
  Which achieves the best accuracy? At what point do longer n-grams
  hurt rather than help (too sparse)?

>>> EXERCISE 8.3
  Implement a simple stemming step before vectorisation:
    from nltk.stem import PorterStemmer
    stemmer = PorterStemmer()
    'running' → 'run', 'films' → 'film'
  Does stemming improve CV accuracy on this corpus?

>>> EXERCISE 8.4  (Advanced — full IMDB dataset)
  Download the IMDB dataset (50k reviews). Implement the full out-of-core
  pipeline with HashingVectorizer + SGDClassifier.partial_fit().
  After each 1000 reviews, record the accuracy on a held-out 5000-review
  test set. Plot accuracy vs reviews processed.
  Target: >88% accuracy with all 45k training reviews.
""")

if __name__ == "__main__":
    print("\n✓ sentiment analysis complete.")
