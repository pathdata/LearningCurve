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

## Overview

Two scripts apply the scikit-learn classifier framework to the **Breast Cancer Wisconsin** dataset — a real clinical binary classification problem where class imbalance, high dimensionality, and the cost of missed diagnoses all demand more careful evaluation than the balanced Iris dataset.

| Script | Focus | Classifiers |
|--------|-------|-------------|
| `ch03_breast_cancer.py` | Foundations — five classifiers, decision boundaries, feature importances | LR, SVM, Decision Tree, k-NN, Random Forest |
| `ch03_breast_cancer_v2.py` | Production — SMOTE, GridSearchCV, model persistence | LR, k-NN, Random Forest, SVM (tuned) |

Run `ch03_breast_cancer.py` first to understand the dataset and baseline classifiers, then `ch03_breast_cancer_v2.py` for the full imbalanced-learning and tuning workflow.

---

## Dataset

**Breast Cancer Wisconsin (Diagnostic)**  
`sklearn.datasets.load_breast_cancer()`

| Property | Value |
|----------|-------|
| Samples | 569 |
| Features | 30 (mean, SE, and worst of 10 cell nucleus measurements) |
| Classes | 0 = malignant (212), 1 = benign (357) |
| Imbalance ratio | ~1 : 1.69 (malignant : benign) |
| Split | 70% train / 30% test, stratified |

### Why accuracy is not enough

Predicting "benign" for every sample gives ~63% accuracy but misses every malignant case — clinically catastrophic. The scripts emphasise **recall** (sensitivity) and **F1** for the malignant class throughout.

---

## Requirements

### `ch03_breast_cancer.py`

```bash
pip install scikit-learn numpy matplotlib
```

### `ch03_breast_cancer_v2.py`

```bash
pip install scikit-learn imbalanced-learn numpy matplotlib joblib
```

Python 3.9+ and scikit-learn ≥ 1.3 recommended.

---

## Script 1 — `ch03_breast_cancer.py`

### Learning objectives

1. Apply five core classifiers to a real clinical dataset
2. Understand why accuracy alone is insufficient for imbalanced classes
3. Interpret precision, recall, and F1 in a medical screening context
4. Use ROC and Precision-Recall curves for binary classification
5. Identify the most discriminative features via Random Forest importances

### Sections

| Section | Topic | Key concept |
|---------|-------|-------------|
| 1 | Data loading & class imbalance | Why accuracy misleads on skewed data |
| 2 | Logistic Regression | L2 regularisation sweep with F1 reporting |
| 3 | Support Vector Machines | `class_weight='balanced'`, C × gamma grid |
| 4 | Decision Trees | Interpretable clinical rules, Gini impurity |
| 5 | k-Nearest Neighbours | Best k selected by F1, not accuracy |
| 6 | Random Forest | Top-2 feature selection for 2-D boundary plots |
| 7 | Classifier comparison | Accuracy + F1 + ROC-AUC table, confusion matrices |
| 8 | ROC Curves | Binary `from_estimator`, `pos_label=0` (malignant) |
| 9 | Precision-Recall Curves | No-skill baseline, Average Precision |
| 10 | Feature importance analysis | RF importances + LR coefficients (directional) |

### Imbalance handling strategy

All five classifiers use `class_weight='balanced'`, which up-weights malignant samples proportionally to their frequency during training. No resampling is performed.

### Output files

| File | Contents |
|------|----------|
| `bc_decision_tree.png` | Tree with clinical feature names (top 3 levels) |
| `bc_decision_boundaries.png` | All 5 classifiers on the top-2 RF features |
| `bc_confusion_matrices.png` | Side-by-side matrices for all 5 classifiers |
| `bc_roc_curves.png` | ROC curves, positive class = malignant |
| `bc_pr_curves.png` | Precision-Recall curves with no-skill baseline |
| `bc_feature_importances.png` | RF importances + LR coefficients side by side |

### Exercises

**3B.1** — Remove `class_weight='balanced'` from all classifiers. How do accuracy, F1-malignant, and recall change? Which metric is most affected and why?

**3B.2** — Threshold tuning on Logistic Regression:
```python
y_prob = pipe_lr.predict_proba(X_test)[:, 0]   # P(malignant)
for threshold in [0.2, 0.3, 0.5, 0.7, 0.8]:
    y_pred = (y_prob >= threshold).astype(int)
```
Plot precision and recall for the malignant class vs threshold. At what threshold does recall exceed 0.97? What does that cost in precision?

**3B.3** — GridSearchCV for Random Forest:
```python
param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth':    [None, 5, 10],
    'max_features': ['sqrt', 'log2']
}
```
Use `scoring='f1'` with `pos_label=0`. Does tuning improve over the defaults?

**3B.4** *(Advanced)* — PCA pipeline: `StandardScaler → PCA(n_components=k) → LogisticRegression` for k = 2, 5, 10, 15, 20, 30. Plot F1-malignant vs n_components. How many components are needed to retain >95% of full-feature performance?

---

## Script 2 — `ch03_breast_cancer_v2.py`

### Learning objectives

1. Apply SMOTE inside an imblearn Pipeline to prevent data leakage during cross-validation
2. Run GridSearchCV on SVM with a CV score heatmap
3. Compare LR / k-NN / RF / SVM on accuracy, F1, ROC-AUC, and PR-AUC
4. Plot confusion matrices, ROC curves, and PR curves for all four models
5. Save the best model in both `.joblib` and `.pkl` formats

### What is SMOTE?

SMOTE (Synthetic Minority Over-sampling Technique) generates synthetic malignant samples by interpolating between real ones in feature space. Placing SMOTE inside an `imblearn.pipeline.Pipeline` ensures:

- Synthetic samples are created only from training data
- The test set is never resampled
- Each CV fold resamples independently — giving an unbiased CV estimate

### Sections

| Section | Topic | Key concept |
|---------|-------|-------------|
| 1 | Data loading | Class distribution before and after split |
| 2 | imblearn Pipelines | SMOTE step between scaler and classifier |
| 3 | SVM GridSearchCV | 5-fold CV over 25 C × gamma combinations, F1 heatmap |
| 4 | Classifier comparison | Accuracy, F1-M, ROC-AUC, PR-AUC, Recall-0 table |
| 5 | Confusion matrices | FN cell highlighted in red; normalised rates printed |
| 6 | ROC curves | All four classifiers with AUC in legend |
| 7 | Precision-Recall curves | All four classifiers with AP in legend |
| 8 | Combined dashboard | Confusion matrices + ROC + PR in one figure |
| 9 | Save best model | joblib + pickle, reload verification asserted |

### SVM GridSearch parameter grid

```python
C_grid     = [0.01, 0.1, 1, 10, 100]
gamma_grid = [0.001, 0.01, 0.1, 1, "scale"]
scoring    = make_scorer(f1_score, pos_label=0)   # F1 for malignant class
cv         = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

Two plots are produced: a heatmap of mean CV test and train F1 (best cell outlined in green), and a line plot of F1 vs gamma for each C value with ±1 std shading.

### Model selection and saving

The best model is selected automatically by **F1-malignant** across the four classifiers. It is saved in two formats:

```
saved_models/
├── best_model.joblib      # recommended — efficient for numpy arrays
├── best_model.pkl         # portable — no joblib dependency needed
└── model_metadata.txt     # metrics, pipeline description, reload instructions
```

Both files are verified by asserting that reloaded predictions match the original before the script exits.

**Reload (joblib):**
```python
import joblib
model  = joblib.load("saved_models/best_model.joblib")
y_pred = model.predict(X_new)
y_prob = model.predict_proba(X_new)[:, 0]   # P(malignant)
```

**Reload (pickle):**
```python
import pickle
with open("saved_models/best_model.pkl", "rb") as f:
    model = pickle.load(f)
y_pred = model.predict(X_new)
```

### Output files

| File | Contents |
|------|----------|
| `bc_svm_gridsearch_heatmap.png` | C × gamma CV F1 heatmaps (test and train) |
| `bc_svm_cv_scores.png` | CV F1 line plot across gamma values per C |
| `bc_confusion_matrices_v2.png` | All 4 classifiers, FN cell highlighted in red |
| `bc_roc_curves_v2.png` | ROC curves with AUC, positive class = malignant |
| `bc_pr_curves_v2.png` | PR curves with AP, no-skill baseline |
| `bc_dashboard.png` | Combined figure: confusion matrices + ROC + PR |
| `saved_models/best_model.joblib` | Best fitted pipeline (joblib format) |
| `saved_models/best_model.pkl` | Best fitted pipeline (pickle format) |
| `saved_models/model_metadata.txt` | Performance metrics and reload instructions |

---

## Key differences between the two scripts

| Feature | `ch03_breast_cancer.py` | `ch03_breast_cancer_v2.py` |
|---------|------------------------|---------------------------|
| Classifiers | LR, SVM, Decision Tree, k-NN, RF | LR, k-NN, RF, SVM (tuned) |
| Imbalance strategy | `class_weight='balanced'` | SMOTE inside imblearn Pipeline |
| Pipeline type | `sklearn.pipeline.Pipeline` | `imblearn.pipeline.Pipeline` |
| SVM tuning | Manual C × gamma table | GridSearchCV with CV heatmap |
| Evaluation metrics | Accuracy, F1, ROC-AUC | Accuracy, F1, ROC-AUC, PR-AUC, Recall-0 |
| Decision boundary plots | Yes (top-2 RF features) | No |
| Feature importance plots | Yes (RF + LR coefficients) | No |
| Model saving | No | joblib + pickle + metadata |
| Dashboard figure | No | Yes (Section 8) |

---

## Classifier performance summary (test set)

Results from `ch03_breast_cancer_v2.py` with SMOTE + tuned SVM:

| Classifier | Accuracy | F1 (malignant) | ROC-AUC | Recall-0 |
|------------|:--------:|:--------------:|:-------:|:--------:|
| Logistic Regression | 0.965 | 0.955 | 0.998 | 0.984 |
| k-NN (k=7) | 0.953 | 0.938 | 0.993 | 0.938 |
| Random Forest | 0.953 | 0.938 | 0.994 | 0.953 |
| SVM (tuned) | **0.982** | **0.976** | 0.995 | 0.969 |

Best model: **SVM** with C=100, gamma=0.001 (selected by F1-malignant).

---

## Common pitfalls

| Issue | Cause | Fix applied in scripts |
|-------|-------|------------------------|
| High accuracy, poor recall | Ignoring class imbalance | `class_weight='balanced'` (v1) / SMOTE (v2) |
| Data leakage with SMOTE | Resampling before CV split | SMOTE placed inside imblearn Pipeline |
| k-NN tuned by accuracy | Best k by accuracy ≠ best k by F1 | k selected by `f1_score(pos_label=0)` |
| `multi_class` error | Removed in sklearn ≥ 1.5 | Parameter omitted; sklearn auto-selects |
| ROC API version mismatch | `name` vs `estimator_name` in older sklearn | Curves plotted directly with `ax.plot()` |

---




*Part of the Python Machine Learning Tutorial Series — Dr. Priya Lakshmi Narayanan*
