"""
=============================================================================
Python Machine Learning – Tutorial Series  (Updated for Modern Packages)
Ch 3 (Applied): A Tour of ML Classifiers — Breast Cancer Dataset
Dr. Priya Lakshmi Narayanan
=============================================================================
LEARNING OBJECTIVES
  1. Apply the five core classifiers to a real clinical dataset
  2. Understand why accuracy alone is insufficient for imbalanced classes
  3. Interpret precision, recall, and F1 in a medical screening context
  4. Use ROC and Precision-Recall curves for binary classifiers
  5. Identify the most discriminative features via Random Forest importances

KEY DIFFERENCES vs Iris:
  - Binary classification  (0 = malignant, 1 = benign)
  - 30 features  (vs 4) — 2-D boundary plots use top-2 RF features
  - Class imbalance: ~37% malignant, ~63% benign → F1 matters more
  - ROC curves work directly (no label_binarize needed for binary tasks)
  - Precision-Recall curve added — more informative under class imbalance

PACKAGES: pip install scikit-learn numpy matplotlib seaborn
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    ConfusionMatrixDisplay, RocCurveDisplay,
    PrecisionRecallDisplay,
    roc_auc_score, average_precision_score,
    f1_score
)
from sklearn.inspection import DecisionBoundaryDisplay


# ─── UTILITY: 2-D decision boundary on any two specified features ─────────────
def plot_decision_boundary_2feat(clf, X, y, feat_idx=(0, 1),
                                  feat_names=("f0", "f1"),
                                  class_names=("Malignant", "Benign"),
                                  title="", ax=None):
    """
    Fit and plot a 2-D boundary using two features selected by feat_idx.
    clf should already be fitted on the full feature set.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    X_2d = X[:, list(feat_idx)]
    colors = ["#FF9999", "#9999FF"]
    cmap   = ListedColormap(colors)
    DecisionBoundaryDisplay.from_estimator(
        clf, X_2d, cmap=cmap, alpha=0.35, ax=ax)
    scatter_c = ["red", "blue"]
    for idx, cls in enumerate(np.unique(y)):
        ax.scatter(X_2d[y == cls, 0], X_2d[y == cls, 1],
                   color=scatter_c[idx], label=class_names[idx],
                   edgecolors="k", s=30, alpha=0.7)
    ax.set_xlabel(feat_names[0], fontsize=8)
    ax.set_ylabel(feat_names[1], fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7)
    return ax


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: LOAD DATA & EXPLORATORY OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("SECTION 1: Data loading & dataset overview")
print("=" * 65)

bc = load_breast_cancer()
X, y = bc.data, bc.target
feature_names = bc.feature_names        # 30 features
target_names  = bc.target_names         # ['malignant', 'benign']

# 70 / 30 stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y)

print(f"Dataset : Breast Cancer Wisconsin ({X.shape[0]} samples, "
      f"{X.shape[1]} features)")
print(f"Classes : {target_names}  (0 = malignant, 1 = benign)")
print(f"Train   : {X_train.shape[0]}  |  Test: {X_test.shape[0]}")
counts = np.bincount(y_train)
print(f"Class balance (train): malignant={counts[0]}, benign={counts[1]}")
print(f"  Imbalance ratio: {counts[1]/counts[0]:.2f}:1  "
      f"(~{100*counts[0]/counts.sum():.0f}% malignant)")

print("""
WHY ACCURACY CAN MISLEAD ON THIS DATASET:
  If we predicted 'benign' for every sample, accuracy ≈ 63%.
  This looks reasonable but misses ALL malignant cases — clinically
  catastrophic. Recall (sensitivity) for the malignant class is the
  critical metric in screening: we want to minimise false negatives.

  Precision: of predicted malignant, how many truly are?
  Recall:    of actual malignant, how many did we catch?
  F1:        harmonic mean — balances both when classes are unequal.
""")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: LOGISTIC REGRESSION
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("SECTION 2: Logistic Regression")
print("=" * 65)

"""
For binary classification the sigmoid output is directly P(y=1|x).
The decision boundary is a hyperplane in feature space.
With 30 features we cannot visualise it directly — we use the
classification report and ROC/PR curves instead.

L2 regularisation (C=1/λ): controls overfitting on high-dimensional data.
"""

pipe_lr = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    LogisticRegression(C=1.0, max_iter=500, random_state=42))
])
pipe_lr.fit(X_train, y_train)
y_pred_lr = pipe_lr.predict(X_test)

print(f"Logistic Regression accuracy : {accuracy_score(y_test, y_pred_lr):.3f}")
print(f"F1 (malignant class)         : "
      f"{f1_score(y_test, y_pred_lr, pos_label=0):.3f}")
print(classification_report(y_test, y_pred_lr, target_names=target_names))

# Regularisation sweep
print("Regularisation sweep (C parameter):")
print(f"  {'C':>8}  {'Train acc':>10}  {'Test acc':>10}  "
      f"{'F1-malig':>10}")
for C in [0.001, 0.01, 0.1, 1, 10, 100]:
    p = Pipeline([("s", StandardScaler()),
                  ("c", LogisticRegression(C=C, max_iter=500,
                                           random_state=42))])
    p.fit(X_train, y_train)
    f1 = f1_score(y_test, p.predict(X_test), pos_label=0)
    print(f"  {C:>8.3f}  {p.score(X_train,y_train):>10.3f}  "
          f"{p.score(X_test,y_test):>10.3f}  {f1:>10.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: SUPPORT VECTOR MACHINES
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 3: Support Vector Machines (SVM)")
print("=" * 65)

"""
SVM is particularly well-suited to high-dimensional data (30 features here)
because it only relies on support vectors — data points near the margin.

class_weight='balanced' adjusts for imbalance by up-weighting the
minority class (malignant) proportionally to its frequency.
"""

pipe_svm = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    SVC(kernel="rbf", C=1.0, gamma="scale",
                   random_state=42, probability=True,
                   class_weight="balanced"))
])
pipe_svm.fit(X_train, y_train)
y_pred_svm = pipe_svm.predict(X_test)
print(f"SVM (RBF) accuracy  : {accuracy_score(y_test, y_pred_svm):.3f}")
print(f"F1 (malignant class): {f1_score(y_test, y_pred_svm, pos_label=0):.3f}")

# C and gamma grid — printed as table (visualised in Section 7 heatmap)
print("\nC × gamma accuracy grid (test set):")
C_vals     = [0.01, 0.1, 1, 10, 100]
gamma_vals = [0.001, 0.01, 0.1, "scale"]
header = f"  {'C':>6}  " + "  ".join(f"γ={g:>6}" for g in gamma_vals)
print(header)
for C in C_vals:
    row = f"  {C:>6}  "
    for g in gamma_vals:
        p = Pipeline([("s", StandardScaler()),
                      ("c", SVC(kernel="rbf", C=C, gamma=g,
                                class_weight="balanced",random_state=42))])
        p.fit(X_train, y_train)
        row += f"  {p.score(X_test,y_test):>8.3f}"
    print(row)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: DECISION TREE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 4: Decision Trees")
print("=" * 65)

"""
Decision trees are naturally interpretable — each node tests a single
feature threshold, giving a clinician-readable set of rules.

For medical data, limiting max_depth is important:
  - Shallow trees: high bias, but interpretable and less likely to overfit
  - Deep trees: memorise training data, poor generalisation
"""

tree = DecisionTreeClassifier(max_depth=4, criterion="gini",
                               class_weight="balanced", random_state=42)
tree.fit(X_train, y_train)
y_pred_tree = tree.predict(X_test)
print(f"Decision Tree accuracy  : {tree.score(X_test, y_test):.3f}")
print(f"F1 (malignant class)    : {f1_score(y_test, y_pred_tree, pos_label=0):.3f}")

print("\nTree structure (first 3 levels):")
feat_names_short = [f.replace(" (mean)", "").replace(" (worst)", "*")
                     .replace(" (se)", "±") for f in feature_names]
print(export_text(tree, feature_names=list(feature_names), max_depth=3))

# Tree visualisation
fig, ax = plt.subplots(figsize=(16, 6))
plot_tree(tree, feature_names=feature_names, class_names=target_names,
          filled=True, rounded=True, ax=ax, max_depth=3, fontsize=7)
plt.title("Decision Tree — Breast Cancer (max_depth=4, top 3 levels shown)")
plt.savefig("bc_decision_tree.png", dpi=100, bbox_inches="tight")
print("  → Saved: bc_decision_tree.png")
plt.close()

# Feature importances from tree
print("\nTop 10 feature importances (Decision Tree):")
tree_imp = sorted(zip(feature_names, tree.feature_importances_),
                   key=lambda x: x[1], reverse=True)
for name, imp in tree_imp[:10]:
    bar = "█" * int(imp * 50)
    print(f"  {name:<35} {imp:.4f}  {bar}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: k-NEAREST NEIGHBOURS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 5: k-Nearest Neighbours (k-NN)")
print("=" * 65)

"""
With 30 features, k-NN is susceptible to the 'curse of dimensionality':
distances become less meaningful in high-dimensional spaces. Feature
scaling is essential (all features contribute equally to Euclidean distance).

'weights=distance' down-weights farther neighbours, often helping on
datasets where class boundaries are not uniformly distributed.
"""

print("k-NN accuracy and F1 vs k:")
print(f"  {'k':>5}  {'Train acc':>10}  {'Test acc':>10}  {'F1-malig':>10}")
best_k, best_f1 = 1, 0
for k in [1, 3, 5, 7, 11, 15, 21]:
    pipe_knn = Pipeline([
        ("s", StandardScaler()),
        ("c", KNeighborsClassifier(n_neighbors=k, metric="euclidean",
                                    weights="distance"))
    ])
    pipe_knn.fit(X_train, y_train)
    f1 = f1_score(y_test, pipe_knn.predict(X_test), pos_label=0)
    if f1 > best_f1:
        best_f1, best_k = f1, k
    print(f"  {k:>5}  {pipe_knn.score(X_train,y_train):>10.3f}  "
          f"{pipe_knn.score(X_test,y_test):>10.3f}  {f1:>10.3f}")

print(f"\nBest k by F1-malignant: k={best_k}  (F1={best_f1:.3f})")
pipe_knn_best = Pipeline([
    ("s", StandardScaler()),
    ("c", KNeighborsClassifier(n_neighbors=best_k, metric="euclidean",
                                weights="distance"))
])
pipe_knn_best.fit(X_train, y_train)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: RANDOM FOREST
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 6: Random Forest")
print("=" * 65)

"""
Random Forest is robust to irrelevant features — useful here since not all
30 features are equally informative.

Feature importances from an ensemble of 500 trees are more stable and
reliable than from a single decision tree.
"""

rf = RandomForestClassifier(n_estimators=500, max_features="sqrt",
                              criterion="gini", class_weight="balanced",
                              random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print(f"Random Forest accuracy  : {rf.score(X_test, y_test):.3f}")
print(f"F1 (malignant class)    : {f1_score(y_test, y_pred_rf, pos_label=0):.3f}")

print("\nTop 10 feature importances (Random Forest):")
rf_importances = sorted(zip(feature_names, rf.feature_importances_),
                         key=lambda x: x[1], reverse=True)
for name, imp in rf_importances[:10]:
    bar = "█" * int(imp * 50)
    print(f"  {name:<35} {imp:.4f}  {bar}")

# Identify top-2 features for 2-D boundary plots
top2_idx   = np.argsort(rf.feature_importances_)[::-1][:2]
top2_names = [feature_names[i] for i in top2_idx]
print(f"\nTop-2 features for 2-D boundary plots: "
      f"{top2_names[0]}  &  {top2_names[1]}")

# Train 2-D pipelines for each classifier using only the top-2 features
X_train_2d = X_train[:, top2_idx]
X_test_2d  = X_test[:, top2_idx]

def make_2d_pipeline(clf_proto):
    """Return a fresh pipeline trained on the 2-feature subset."""
    p = Pipeline([("s", StandardScaler()), ("c", clf_proto)])
    p.fit(X_train_2d, y_train)
    return p

pipe_lr_2d   = make_2d_pipeline(
    LogisticRegression(C=1.0, max_iter=500, random_state=42))
pipe_svm_2d  = make_2d_pipeline(
    SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced",
        random_state=42))
tree_2d      = make_2d_pipeline(
    DecisionTreeClassifier(max_depth=4, class_weight="balanced",
                            random_state=42))
pipe_knn_2d  = make_2d_pipeline(
    KNeighborsClassifier(n_neighbors=best_k, weights="distance"))
rf_2d        = make_2d_pipeline(
    RandomForestClassifier(n_estimators=200, class_weight="balanced",
                            random_state=42, n_jobs=-1))

fig, axes = plt.subplots(1, 5, figsize=(22, 4))
configs = [
    ("Logistic Regression", pipe_lr_2d),
    ("SVM (RBF)",           pipe_svm_2d),
    ("Decision Tree",       tree_2d),
    (f"k-NN (k={best_k})",  pipe_knn_2d),
    ("Random Forest",       rf_2d),
]
for ax, (name, clf) in zip(axes, configs):
    plot_decision_boundary_2feat(
        clf, X_train_2d, y_train,
        feat_idx=(0, 1),
        feat_names=(top2_names[0].split(" ")[0],
                    top2_names[1].split(" ")[0]),
        title=name, ax=ax)

plt.suptitle(f"Decision Boundaries (top-2 RF features): "
             f"{top2_names[0]}  vs  {top2_names[1]}", y=1.02)
plt.tight_layout()
plt.savefig("bc_decision_boundaries.png", dpi=100, bbox_inches="tight")
print("  → Saved: bc_decision_boundaries.png")
plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: CLASSIFIER COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 7: Head-to-head comparison on Breast Cancer test set")
print("=" * 65)

classifiers = {
    "Logistic Regression": pipe_lr,
    "SVM (RBF)":           pipe_svm,
    "Decision Tree":       Pipeline([("c", tree)]),
    f"k-NN (k={best_k})":  pipe_knn_best,
    "Random Forest":       Pipeline([("c", rf)]),
}

print(f"\n  {'Classifier':<22}  {'Accuracy':>9}  {'F1-Malig':>9}  "
      f"{'AUC-ROC':>9}")
print("  " + "─" * 57)
for name, clf in classifiers.items():
    y_pred = clf.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    f1_m   = f1_score(y_test, y_pred, pos_label=0)
    # roc_auc_score assumes pos_label=1; relabel so malignant(0) becomes 1
    if hasattr(clf, "predict_proba"):
        y_score = 1 - clf.predict_proba(X_test)[:, 1]  # P(malignant)
    else:
        y_score = -clf.decision_function(X_test)        # negate: higher=malignant
    auc_roc = roc_auc_score(1 - y_test, y_score)
    print(f"  {name:<22}  {acc:>9.3f}  {f1_m:>9.3f}  {auc_roc:>9.3f}")

# Confusion matrices
fig, axes = plt.subplots(1, 5, figsize=(22, 4))
for ax, (name, clf) in zip(axes, classifiers.items()):
    ConfusionMatrixDisplay.from_estimator(
        clf, X_test, y_test, display_labels=target_names,
        colorbar=False, ax=ax, cmap="Blues")
    ax.set_title(name, fontsize=9)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
plt.suptitle("Confusion Matrices — Breast Cancer Test Set", y=1.02)
plt.tight_layout()
plt.savefig("bc_confusion_matrices.png", dpi=100, bbox_inches="tight")
print("\n  → Saved: bc_confusion_matrices.png")
plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: ROC CURVES  (binary: works directly with from_estimator)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 8: ROC Curves (binary classification)")
print("=" * 65)

"""
For binary classification RocCurveDisplay.from_estimator works directly.
We set pos_label=0 (malignant) — the clinically critical class.

ROC AUC measures the probability that the classifier ranks a randomly
chosen malignant case higher than a randomly chosen benign case.
"""

fig, ax = plt.subplots(figsize=(8, 6))

roc_candidates = [
    ("Logistic Regression", pipe_lr,                      True),
    ("SVM (RBF)",           pipe_svm,                     True),
    ("Decision Tree",       Pipeline([("c", tree)]),      True),
    (f"k-NN (k={best_k})",  pipe_knn_best,                True),
    ("Random Forest",       Pipeline([("c", rf)]),        True),
]

for name, clf, use_proba in roc_candidates:
    RocCurveDisplay.from_estimator(
        clf, X_test, y_test,
        pos_label=0,
        response_method="predict_proba",
        name=name, ax=ax,
        plot_chance_level=(name == "Logistic Regression"))

ax.set_title("ROC Curves — Breast Cancer Test Set (pos = malignant)")
ax.legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.savefig("bc_roc_curves.png", dpi=100, bbox_inches="tight")
print("  → Saved: bc_roc_curves.png")
plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9: PRECISION-RECALL CURVES
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 9: Precision-Recall Curves")
print("=" * 65)

"""
Under class imbalance, the Precision-Recall curve reveals performance
differences that ROC curves can obscure.

  High recall = few malignant cases missed (low false-negative rate)
  High precision = few benign cases falsely flagged as malignant

For cancer screening, recall is paramount: a missed malignant case (FN)
has far graver consequences than an unnecessary follow-up biopsy (FP).
Average Precision (AP) summarises the PR curve as a single number.
"""

fig, ax = plt.subplots(figsize=(8, 6))

for name, clf, use_proba in roc_candidates:
    PrecisionRecallDisplay.from_estimator(
        clf, X_test, y_test,
        pos_label=0,
        response_method="predict_proba",
        name=name, ax=ax)

ax.set_title("Precision-Recall Curves — Breast Cancer (pos = malignant)")
ax.legend(loc="upper right", fontsize=8)
# Baseline: proportion of positive (malignant) samples
baseline = np.mean(y_test == 0)
ax.axhline(baseline, linestyle="--", color="k", alpha=0.5,
           label=f"No-skill baseline (P={baseline:.2f})")
plt.tight_layout()
plt.savefig("bc_pr_curves.png", dpi=100, bbox_inches="tight")
print("  → Saved: bc_pr_curves.png")
plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10: FEATURE IMPORTANCE — FULL COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 10: Feature Importance Analysis")
print("=" * 65)

"""
Random Forest importances aggregate across 500 trees — more stable than
a single tree's Gini importances. Logistic Regression coefficients (after
standardisation) indicate direction and magnitude of each feature's
contribution to the decision boundary.
"""

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# ── Random Forest importances (top 15) ──────────────────────────────────────
ax = axes[0]
sorted_idx = np.argsort(rf.feature_importances_)[::-1][:15]
ax.barh([feature_names[i] for i in sorted_idx[::-1]],
        rf.feature_importances_[sorted_idx[::-1]],
        color="steelblue", edgecolor="white")
ax.set_xlabel("Importance (mean decrease in Gini impurity)")
ax.set_title("Random Forest — Top 15 Feature Importances")
ax.tick_params(axis="y", labelsize=8)

# ── Logistic Regression coefficients (standardised) ─────────────────────────
ax = axes[1]
lr_coef = pipe_lr.named_steps["clf"].coef_[0]   # shape (30,)
sorted_coef_idx = np.argsort(np.abs(lr_coef))[::-1][:15]
colors_coef = ["#d73027" if c < 0 else "#4575b4"
               for c in lr_coef[sorted_coef_idx[::-1]]]
ax.barh([feature_names[i] for i in sorted_coef_idx[::-1]],
        lr_coef[sorted_coef_idx[::-1]],
        color=colors_coef, edgecolor="white")
ax.axvline(0, color="k", linewidth=0.8)
ax.set_xlabel("Coefficient value (negative → malignant, positive → benign)")
ax.set_title("Logistic Regression — Top 15 Feature Coefficients\n"
             "(red = associated with malignant class)")
ax.tick_params(axis="y", labelsize=8)

plt.tight_layout()
plt.savefig("bc_feature_importances.png", dpi=100, bbox_inches="tight")
print("  → Saved: bc_feature_importances.png")
plt.close()

print("\nTop 10 RF importances:")
for name, imp in rf_importances[:10]:
    bar = "█" * int(imp * 50)
    print(f"  {name:<35} {imp:.4f}  {bar}")


# ─────────────────────────────────────────────────────────────────────────────
# EXERCISES
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("EXERCISES")
print("=" * 65)
print("""
>>> EXERCISE 3B.1
  The five classifiers all have class_weight='balanced'. Remove this
  parameter and retrain. How do accuracy, F1-malignant, and recall
  for the malignant class change? Which metric is most affected and why?

>>> EXERCISE 3B.2
  Vary the decision threshold of Logistic Regression:
    y_prob = pipe_lr.predict_proba(X_test)[:, 0]   # P(malignant)
    for threshold in [0.2, 0.3, 0.5, 0.7, 0.8]:
        y_pred = (y_prob >= threshold).astype(int)
  Plot precision and recall for the malignant class vs threshold on the
  same axes. At what threshold does recall for malignant exceed 0.97?
  What does that cost in terms of precision?

>>> EXERCISE 3B.3
  Use GridSearchCV to tune the Random Forest:
    param_grid = {
        'n_estimators': [100, 300, 500],
        'max_depth':    [None, 5, 10],
        'max_features': ['sqrt', 'log2']
    }
    scoring='f1' with pos_label=0 (hint: use make_scorer).
  Does tuning improve over the defaults? By how much?

>>> EXERCISE 3B.4  (Advanced)
  Apply PCA before classification:
    Pipeline([StandardScaler(), PCA(n_components=k), LogisticRegression()])
  for k = 2, 5, 10, 15, 20, 30.
  Plot F1-malignant vs n_components. How many components are needed to
  retain >95% of the performance of the full 30-feature model?
  Explain the result in terms of the variance explained by each component.
""")

if __name__ == "__main__":
    print("\n✓ Ch 3 (Breast Cancer) complete.")
