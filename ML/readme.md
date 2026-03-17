# Python Machine Learning Concepts
## A Tour of ML Classifiers Using Scikit-learn

**Dr. Priya Lakshmi Narayanan**  
*Python Machine Learning – Tutorial Series*

---

## Overview

This chapter introduces five core scikit-learn classifiers through two progressively challenging exercises. The Iris dataset provides a clean, balanced introduction to the classifier API. The Breast Cancer dataset then applies the same framework to a real clinical problem — where class imbalance, high dimensionality, and the cost of false negatives make evaluation much more nuanced.

Both scripts are self-contained and produce annotated plots saved to the working directory.

---

## Files

| File | Dataset | Classification | Features |
|------|---------|----------------|----------|
| `ch03_classifiers.py` | Iris | 3-class | 4 |
| `ch03_breast_cancer.py` | Breast Cancer Wisconsin | Binary | 30 |

---

## Requirements

```bash
pip install scikit-learn numpy matplotlib
```

Python 3.9+ and scikit-learn ≥ 1.3 are recommended. The `multi_class` parameter has been removed from `LogisticRegression` in sklearn ≥ 1.5 — the scripts reflect this.

---

## Exercise 1 — Iris Dataset (`ch03_classifiers.py`)

### Dataset
- **569 → 150 samples**, 4 features, 3 balanced classes (setosa, versicolor, virginica)
- 70/30 stratified train/test split

### Sections

| Section | Topic | Key Concept |
|---------|-------|-------------|
| 1 | Data loading & sklearn API | `fit` / `predict` / `score` contract |
| 2 | Logistic Regression | L2 regularisation, C sweep |
| 3 | Support Vector Machines | RBF kernel, C and gamma effects |
| 4 | Decision Trees | Gini impurity, feature importances |
| 5 | k-Nearest Neighbours | Bias-variance trade-off vs k |
| 6 | Random Forest | Bagging, ensemble importances |
| 7 | Classifier comparison | Accuracy table, confusion matrices |
| 8 | ROC Curves (multiclass) | Micro-averaged One-vs-Rest ROC |

### Output plots
- `ch03_svm_boundaries.png` — 2×3 grid of decision boundaries across C and gamma values
- `ch03_decision_tree.png` — Tree structure visualisation (top 3 levels)
- `ch03_confusion_matrices.png` — All 5 classifiers side by side
- `ch03_roc_curves.png` — Micro-averaged OvR ROC curves

### Key note on multiclass ROC
`RocCurveDisplay.from_estimator` only supports **binary** classifiers. For the 3-class Iris problem, labels are binarized with `label_binarize` and a micro-averaged ROC curve is computed manually via `roc_curve(y_test_bin.ravel(), y_score.ravel())` before plotting with `RocCurveDisplay`.

### Exercises

**3.1** — Train all five classifiers on `load_breast_cancer()`. Which achieves the highest F1 on the positive class? Why is F1 better than accuracy here?

**3.2** — SVM grid search: sweep C = [0.001–100] × gamma = [0.001–1]. Plot train/test accuracy as a seaborn heatmap. Identify the best generalisation region and where overfitting begins.

**3.3** — Decision tree depth study: plot train and test accuracy vs `max_depth` (1–None). At which depth does overfitting begin? What does this reveal about the bias-variance trade-off?

**3.4** *(Advanced)* — Pipeline with PCA: `StandardScaler → PCA(n_components=k) → LogisticRegression`. Does dimensionality reduction help or hurt on Iris? Why?

---

## Exercise 2 — Breast Cancer Dataset (`ch03_breast_cancer.py`)

### Dataset
- **569 samples**, 30 features, binary classes (0 = malignant, 1 = benign)
- Class imbalance: ~37% malignant, ~63% benign
- 70/30 stratified train/test split

### Why this dataset matters
Accuracy alone can reach ~63% by predicting "benign" for every sample — yet this misses every malignant case. The exercises here build the habit of evaluating classifiers with **recall**, **F1**, and **Precision-Recall curves** alongside accuracy, which is essential practice for any real-world classification problem.

### Sections

| Section | Topic | Key Concept |
|---------|-------|-------------|
| 1 | Data loading & class imbalance | Why accuracy misleads |
| 2 | Logistic Regression | C sweep reported with F1-malignant |
| 3 | SVM | `class_weight='balanced'`, C×gamma grid |
| 4 | Decision Trees | Interpretable rules for clinical data |
| 5 | k-Nearest Neighbours | Best k selected by F1, not accuracy |
| 6 | Random Forest | Top-2 feature selection for 2-D plots |
| 7 | Classifier comparison | Accuracy + F1 + AUC-ROC table |
| 8 | ROC Curves (binary) | Direct `from_estimator` with `pos_label=0` |
| 9 | Precision-Recall Curves | More informative under imbalance |
| 10 | Feature importance analysis | RF importances + LR coefficients |

### Output plots
- `bc_decision_tree.png` — Tree with clinical feature names
- `bc_decision_boundaries.png` — All 5 classifiers on top-2 RF features (worst area vs worst perimeter)
- `bc_confusion_matrices.png` — Side-by-side with malignant/benign labels
- `bc_roc_curves.png` — Binary ROC, `pos_label=0` (malignant)
- `bc_pr_curves.png` — Precision-Recall curves with no-skill baseline
- `bc_feature_importances.png` — RF importances + LR coefficients (directional)

### Exercises

**3B.1** — Remove `class_weight='balanced'` from all classifiers and retrain. How do accuracy, F1-malignant, and recall for the malignant class change? Which metric is most affected and why?

**3B.2** — Threshold tuning on Logistic Regression:
```python
y_prob = pipe_lr.predict_proba(X_test)[:, 0]  # P(malignant)
for threshold in [0.2, 0.3, 0.5, 0.7, 0.8]:
    y_pred = (y_prob >= threshold).astype(int)
```
Plot precision and recall for the malignant class vs threshold on the same axes. At what threshold does recall exceed 0.97? What does that cost in precision?

**3B.3** — GridSearchCV for Random Forest:
```python
param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth':    [None, 5, 10],
    'max_features': ['sqrt', 'log2']
}
```
Use `scoring='f1'` with `pos_label=0`. Does tuning improve over the defaults?

**3B.4** *(Advanced)* — PCA pipeline for `k = 2, 5, 10, 15, 20, 30`. Plot F1-malignant vs `n_components`. How many components are needed to retain >95% of the full 30-feature performance? Explain using explained variance.

---

## Classifier Quick Reference

| Classifier | Needs scaling | Handles imbalance | Interpretable | Key hyperparameters |
|------------|:---:|:---:|:---:|---------------------|
| Logistic Regression | ✓ | via `class_weight` | Partially (coefficients) | `C` |
| SVM (RBF) | ✓ | via `class_weight` | ✗ | `C`, `gamma` |
| Decision Tree | ✗ | via `class_weight` | ✓ | `max_depth`, `criterion` |
| k-NN | ✓ | via `weights` | ✗ | `n_neighbors`, `metric` |
| Random Forest | ✗ | via `class_weight` | Partially (importances) | `n_estimators`, `max_features` |

---

## Common Pitfalls Fixed in These Scripts

- **`multi_class` removed** — `LogisticRegression(multi_class=...)` deprecated and removed in sklearn ≥ 1.5. Simply omit the argument; modern sklearn selects the strategy automatically.
- **Multiclass ROC** — `RocCurveDisplay.from_estimator` raises a `ValueError` for >2 classes. Solution: use `label_binarize` + manual `roc_curve` on the ravelled arrays (Iris script).
- **Class imbalance** — All breast cancer classifiers use `class_weight='balanced'`; best k for k-NN is selected by F1-malignant rather than accuracy.

---

## Learning Progression

```
ch03_classifiers.py          ch03_breast_cancer.py
────────────────────         ──────────────────────────
Balanced classes         →   Imbalanced classes
4 features               →   30 features
Accuracy sufficient      →   F1, AUC-ROC, PR curves needed
Multiclass ROC (manual)  →   Binary ROC (direct API)
Visualise boundaries     →   Visualise importances + coefficients
```

---

*Part of the Python Machine Learning Tutorial Series — Dr. Priya Lakshmi Narayanan*
