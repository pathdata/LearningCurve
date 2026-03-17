"""
=============================================================================
Python Machine Learning – Tutorial Series  (Updated for Modern Packages)
Ch 3: A Tour of ML Classifiers Using Scikit-learn
Dr. Priya Lakshmi Narayanan
=============================================================================
LEARNING OBJECTIVES
  1. Use the unified scikit-learn estimator API (fit / predict / score)
  2. Train and compare five core classifiers on the Iris dataset
  3. Understand Logistic Regression with L2 regularisation
  4. Build and tune an SVM with RBF kernel
  5. Interpret decision tree splits and feature importances
  6. Visualise 2-D decision boundaries for any classifier
  7. Wrap preprocessing + model in a reusable Pipeline

UPDATED APIs (book used deprecated modules):
  sklearn.cross_validation  →  sklearn.model_selection
  sklearn.grid_search       →  sklearn.model_selection
  StratifiedShuffleSplit    →  train_test_split(stratify=y)
  All deprecated params removed / renamed

PACKAGES: pip install scikit-learn numpy matplotlib
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                              ConfusionMatrixDisplay, RocCurveDisplay,
                              roc_curve, auc)
from sklearn.inspection import DecisionBoundaryDisplay


# ─── UTILITY: decision boundary plot (replaces book's custom function) ───────
def plot_decision_boundary(clf, X, y, title="", ax=None):
    """
    Plot a 2-D decision boundary using sklearn's DecisionBoundaryDisplay.
    Only uses the first two features of X.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#FF9999", "#9999FF", "#99FF99"]
    cmap   = ListedColormap(colors[:len(np.unique(y))])
    DecisionBoundaryDisplay.from_estimator(
        clf, X[:, :2], cmap=cmap, alpha=0.3, ax=ax)
    scatter_colors = ["red", "blue", "green"]
    for idx, cls in enumerate(np.unique(y)):
        ax.scatter(X[y == cls, 0], X[y == cls, 1],
                   color=scatter_colors[idx], label=f"Class {cls}",
                   edgecolors="k", s=40)
    ax.set_title(title)
    ax.legend(loc="upper left")
    return ax


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: LOAD DATA & THE SCIKIT-LEARN ESTIMATOR API
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("SECTION 1: Data loading & the sklearn estimator API")
print("=" * 65)

iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names
target_names  = iris.target_names

# 70 / 30 stratified split  (stratify= replaces StratifiedShuffleSplit)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y)

print(f"Dataset : Iris  ({X.shape[0]} samples, {X.shape[1]} features)")
print(f"Classes : {target_names}")
print(f"Train   : {X_train.shape[0]}  |  Test: {X_test.shape[0]}")
print(f"Class balance (train): {np.bincount(y_train)}")

# ─── The universal 3-method contract ─────────────────────────────────────────
print("""
Every sklearn estimator follows the same contract:
  clf.fit(X_train, y_train)   ← learn parameters from training data
  clf.predict(X_test)          ← predict class labels for new data
  clf.score(X_test, y_test)    ← return accuracy (or R² for regressors)

Transformers add:
  transformer.transform(X)     ← apply learned transformation
  transformer.fit_transform(X) ← fit and transform in one step
""")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: LOGISTIC REGRESSION
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("SECTION 2: Logistic Regression")
print("=" * 65)

"""
Logistic Regression models class probabilities via the sigmoid function:
  P(y=1 | x) = 1 / (1 + exp(−(w·x + b)))

L2 regularisation (default) adds λ||w||² to the cost.
C = 1/λ: smaller C → stronger regularisation → simpler boundary.

Multi-class: one-vs-rest (OvR) or softmax (multinomial).
"""

# Pipeline: standardise → classify
pipe_lr = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    LogisticRegression(C=100.0, max_iter=300,
                                   random_state=42))
])
pipe_lr.fit(X_train, y_train)
y_pred_lr = pipe_lr.predict(X_test)

print(f"Logistic Regression accuracy: {accuracy_score(y_test, y_pred_lr):.3f}")
print(classification_report(y_test, y_pred_lr, target_names=target_names))

# Effect of C (regularisation strength)
print("\nRegularisation sweep (C parameter):")
print(f"  {'C':>10}  {'Train acc':>10}  {'Test acc':>10}")
for C in [0.001, 0.01, 0.1, 1, 10, 100]:
    p = Pipeline([("s", StandardScaler()),
                  ("c", LogisticRegression(C=C, max_iter=500))])
    p.fit(X_train, y_train)
    print(f"  {C:>10.3f}  {p.score(X_train,y_train):>10.3f}  "
          f"{p.score(X_test,y_test):>10.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: SUPPORT VECTOR MACHINES
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 3: Support Vector Machines (SVM)")
print("=" * 65)

"""
SVM finds the hyperplane that maximises the margin between classes.
Kernel trick maps data to a higher-dimensional space implicitly,
allowing non-linear decision boundaries.

Key parameters:
  C     : penalty for misclassification (trade-off margin vs errors)
  gamma : RBF kernel bandwidth (how far a single sample's influence reaches)
  kernel: 'linear', 'rbf', 'poly', 'sigmoid'
"""

pipe_svm = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    SVC(kernel="rbf", C=1.0, gamma=0.2, random_state=42,
                   probability=True))  # probability=True for predict_proba
])
pipe_svm.fit(X_train, y_train)
print(f"SVM (RBF) accuracy: {pipe_svm.score(X_test, y_test):.3f}")

# Visualise C and gamma effects on 2-feature subset
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("SVM Decision Boundaries: Effect of C and gamma", fontsize=13)
X_2d_train = X_train[:, :2]
X_2d_test  = X_test[:, :2]

for ax, gamma in zip(axes[0], [0.01, 0.2, 5.0]):
    svm_2d = Pipeline([("s", StandardScaler()),
                       ("c", SVC(kernel="rbf", C=1.0, gamma=gamma))])
    svm_2d.fit(X_2d_train, y_train)
    plot_decision_boundary(svm_2d, X_2d_train, y_train,
                           title=f"C=1.0, γ={gamma}", ax=ax)

for ax, C in zip(axes[1], [0.01, 1.0, 100.0]):
    svm_2d = Pipeline([("s", StandardScaler()),
                       ("c", SVC(kernel="rbf", C=C, gamma=0.2))])
    svm_2d.fit(X_2d_train, y_train)
    plot_decision_boundary(svm_2d, X_2d_train, y_train,
                           title=f"C={C}, γ=0.2", ax=ax)

plt.tight_layout()
plt.savefig("ch03_svm_boundaries.png", dpi=100, bbox_inches="tight")
print("  → Saved: ch03_svm_boundaries.png")
plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: DECISION TREE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 4: Decision Trees")
print("=" * 65)

"""
Decision Trees split on the feature that maximises information gain:
  IG = I(parent) - sum(N_child/N_parent * I(child))

Impurity measures:
  Gini: I_G = 1 - Σ p²_k
  Entropy: I_H = -Σ p_k log₂(p_k)

Key hyperparameter: max_depth — controls overfitting.
"""

tree = DecisionTreeClassifier(max_depth=4, criterion="gini", random_state=42)
tree.fit(X_train, y_train)
print(f"Decision Tree accuracy: {tree.score(X_test, y_test):.3f}")

# Print tree structure as text
print("\nTree structure (first 3 levels):")
print(export_text(tree, feature_names=feature_names, max_depth=3))

# Visual plot of the tree
fig, ax = plt.subplots(figsize=(14, 6))
plot_tree(tree, feature_names=feature_names, class_names=target_names,
          filled=True, rounded=True, ax=ax, max_depth=3)
plt.title("Decision Tree (max_depth=4, showing top 3 levels)")
plt.savefig("ch03_decision_tree.png", dpi=100, bbox_inches="tight")
print("  → Saved: ch03_decision_tree.png")
plt.close()

# Feature importances
print("\nFeature importances:")
for name, imp in sorted(zip(feature_names, tree.feature_importances_),
                          key=lambda x: x[1], reverse=True):
    bar = "█" * int(imp * 40)
    print(f"  {name:<30} {imp:.4f}  {bar}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: k-NEAREST NEIGHBOURS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 5: k-Nearest Neighbours (k-NN)")
print("=" * 65)

"""
k-NN classifies a new sample by majority vote of its k nearest neighbours
in the feature space.

No training phase (lazy learner) — all computation at prediction time.
MUST scale features: distances are meaningless on unscaled data.

Hyperparameter k controls bias-variance trade-off:
  k=1  → low bias, high variance (overfitting)
  k→N  → high bias, low variance (underfitting)
"""

print("\nk-NN accuracy vs k:")
print(f"  {'k':>5}  {'Train acc':>10}  {'Test acc':>10}")
for k in [1, 3, 5, 7, 11, 21]:
    pipe_knn = Pipeline([("s", StandardScaler()),
                          ("c", KNeighborsClassifier(n_neighbors=k,
                                                     metric="euclidean"))])
    pipe_knn.fit(X_train, y_train)
    print(f"  {k:>5}  {pipe_knn.score(X_train,y_train):>10.3f}  "
          f"{pipe_knn.score(X_test,y_test):>10.3f}")

pipe_knn_best = Pipeline([("s", StandardScaler()),
                            ("c", KNeighborsClassifier(n_neighbors=5))])
pipe_knn_best.fit(X_train, y_train)
print(f"\nBest k-NN (k=5) accuracy: {pipe_knn_best.score(X_test, y_test):.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: RANDOM FOREST
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 6: Random Forest")
print("=" * 65)

"""
Random Forest = Bagging of decision trees + random feature subsets.
Each tree sees:
  - A bootstrap sample of training data
  - sqrt(n_features) random features at each split

Reduces variance vs a single tree. Usually does NOT need feature scaling.
"""

rf = RandomForestClassifier(n_estimators=500, max_features="sqrt",
                              criterion="gini", random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
print(f"Random Forest accuracy: {rf.score(X_test, y_test):.3f}")

print("\nFeature importances (Random Forest):")
for name, imp in sorted(zip(feature_names, rf.feature_importances_),
                          key=lambda x: x[1], reverse=True):
    bar = "█" * int(imp * 40)
    print(f"  {name:<30} {imp:.4f}  {bar}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: CLASSIFIER COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 7: Head-to-head comparison on Iris test set")
print("=" * 65)

classifiers = {
    "Logistic Regression": pipe_lr,
    "SVM (RBF)":           pipe_svm,
    "Decision Tree":       Pipeline([("c", tree)]),  # tree already fitted
    "k-NN (k=5)":          pipe_knn_best,
    "Random Forest":       Pipeline([("c", rf)]),
}

print(f"\n  {'Classifier':<22}  {'Test Accuracy':>14}")
print("  " + "─" * 40)
for name, clf in classifiers.items():
    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"  {name:<22}  {acc:>14.3f}")

# Confusion matrices side by side
fig, axes = plt.subplots(1, 5, figsize=(20, 3))
for ax, (name, clf) in zip(axes, classifiers.items()):
    ConfusionMatrixDisplay.from_estimator(
        clf, X_test, y_test, display_labels=target_names,
        colorbar=False, ax=ax)
    ax.set_title(name, fontsize=9)
    ax.set_xlabel(""); ax.set_ylabel("")
plt.suptitle("Confusion Matrices — Iris Test Set", y=1.02)
plt.tight_layout()
plt.savefig("ch03_confusion_matrices.png", dpi=100, bbox_inches="tight")
print("\n  → Saved: ch03_confusion_matrices.png")
plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: MULTICLASS ROC CURVES (modern sklearn API)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 8: Multi-class ROC Curves (One-vs-Rest)")
print("=" * 65)

# RocCurveDisplay.from_estimator only supports binary classifiers.
# For multiclass (Iris has 3 classes) we binarize labels and compute
# the micro-averaged ROC curve manually, then plot with RocCurveDisplay.

classes = np.unique(y_test)
y_test_bin = label_binarize(y_test, classes=classes)   # shape (n, 3)

fig, ax = plt.subplots(figsize=(8, 6))

candidates = [
    ("Logistic Regression", pipe_lr,  True),   # has predict_proba
    ("SVM (RBF)",           pipe_svm, False),  # decision_function
    ("Random Forest",       Pipeline([("c", rf)]), True),
]

for name, clf, use_proba in candidates:
    if use_proba:
        y_score = clf.predict_proba(X_test)          # shape (n, 3)
    else:
        # SVC with decision_function returns shape (n, 3) for OvR
        y_score = clf.decision_function(X_test)
        # Normalise to [0,1] range so curves are comparable
        y_score = (y_score - y_score.min()) / (y_score.max() - y_score.min())

    # Micro-average: flatten all classes into one ROC curve
    fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
    roc_auc = auc(fpr, tpr)
    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                    estimator_name=name).plot(ax=ax)

# Chance-level diagonal
ax.plot([0, 1], [0, 1], "k--", label="Chance level (AUC = 0.50)")
ax.set_title("ROC Curves — One-vs-Rest Micro-average (Iris, 3 classes)")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig("ch03_roc_curves.png", dpi=100, bbox_inches="tight")
print("  → Saved: ch03_roc_curves.png")
plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# EXERCISES
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("EXERCISES")
print("=" * 65)
print("""
>>> EXERCISE 3.1
  Load the breast_cancer dataset (load_breast_cancer()).
  Train all five classifiers. Which achieves highest F1 on the positive class?
  Why is F1 a better metric than accuracy here?

>>> EXERCISE 3.2
  For the SVM, create a grid of C=[0.001,0.01,0.1,1,10,100] and
  gamma=[0.001,0.01,0.1,1] values. For EACH combination, train and record
  the train/test accuracy. Plot as a heatmap (seaborn.heatmap). Where is
  the best generalisation? Where is the model overfitting?

>>> EXERCISE 3.3
  Set max_depth to 1, 2, 3, 4, 5, None for a Decision Tree. Plot train
  accuracy AND test accuracy vs max_depth on the same axes. At which depth
  does overfitting begin? What does this tell you about the bias-variance
  trade-off?

>>> EXERCISE 3.4  (Advanced)
  Build a Pipeline that: standardises → applies PCA (n_components=2) →
  trains a Logistic Regression. Compare this to a plain LR pipeline.
  Does PCA improve or hurt performance? Why might that be on this dataset?
""")

if __name__ == "__main__":
    print("\n✓ Ch 3 complete. Next: ch04_data_preprocessing.py")
