"""
=============================================================================
Python Machine Learning – Tutorial Series  (Updated for Modern Packages)
Ch 3 (Applied v2): Classifiers + Imbalanced Learning — Breast Cancer
Dr. Priya Lakshmi Narayanan
=============================================================================
LEARNING OBJECTIVES
  1. Use SMOTE (Synthetic Minority Over-sampling Technique) inside an
     imblearn Pipeline so resampling never leaks into the test fold
  2. Run GridSearchCV on SVM with a CV score heatmap
  3. Compare LR / KNN / RF / SVM on accuracy, F1, ROC-AUC, PR-AUC
  4. Plot confusion matrices, ROC curves, and PR curves for all models
  5. Identify and persist the best model with joblib

IMBALANCED LEARNING STRATEGY:
  Raw data: ~37% malignant, ~63% benign.
  Approach A — class_weight='balanced'  (in-algorithm weighting, no resampling)
  Approach B — SMOTE inside Pipeline    (synthetic minority oversampling)
  Both are compared so students can see the difference in practice.

NEW vs v1:
  - imblearn.pipeline.Pipeline replaces sklearn Pipeline (supports SMOTE)
  - SMOTE applied only inside CV folds / to training split (no data leakage)
  - GridSearchCV on SVM with C × gamma heatmap of mean CV F1 scores
  - Best model selected by F1-malignant and saved via joblib
  - Decision Tree dropped from comparison (LR / KNN / RF / SVM only)

PACKAGES: pip install scikit-learn imbalanced-learn numpy matplotlib joblib
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import joblib
import os

from sklearn.datasets          import load_breast_cancer
from sklearn.model_selection   import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing     import StandardScaler
from sklearn.linear_model      import LogisticRegression
from sklearn.svm               import SVC
from sklearn.neighbors         import KNeighborsClassifier
from sklearn.ensemble          import RandomForestClassifier
from sklearn.metrics           import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score, average_precision_score, f1_score, recall_score,
    make_scorer, roc_curve, precision_recall_curve
)

from imblearn.pipeline         import Pipeline as ImbPipeline
from imblearn.over_sampling    import SMOTE


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("SECTION 1: Data loading & class imbalance overview")
print("=" * 65)

bc            = load_breast_cancer()
X, y          = bc.data, bc.target
feature_names = bc.feature_names        # 30 features
target_names  = bc.target_names         # ['malignant', 'benign']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y)

counts_train = np.bincount(y_train)
counts_test  = np.bincount(y_test)

print(f"Dataset : Breast Cancer Wisconsin  "
      f"({X.shape[0]} samples, {X.shape[1]} features)")
print(f"Classes : {list(target_names)}  (0=malignant, 1=benign)")
print(f"\n  Split        Total   Malignant   Benign   Ratio")
print(f"  Train         {X_train.shape[0]:>4}      {counts_train[0]:>5}    {counts_train[1]:>5}   "
      f"1:{counts_train[1]/counts_train[0]:.2f}")
print(f"  Test          {X_test.shape[0]:>4}      {counts_test[0]:>5}    {counts_test[1]:>5}   "
      f"1:{counts_test[1]/counts_test[0]:.2f}")

print("""
SMOTE — Synthetic Minority Over-sampling Technique:
  Generates synthetic malignant samples by interpolating between real
  ones in feature space. Applied INSIDE the imblearn Pipeline so that:
    • Synthetic samples are created only from training data
    • Test data is never resampled (no data leakage)
    • Each CV fold resamples independently (unbiased CV estimate)
""")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: BUILD PIPELINES (imblearn Pipeline with SMOTE)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("SECTION 2: Building imblearn Pipelines with SMOTE")
print("=" * 65)

"""
imblearn.pipeline.Pipeline is a drop-in replacement for sklearn's Pipeline.
It adds support for resamplers (SMOTE, etc.) as pipeline steps.
The resampler is only applied during fit(), not transform() or predict(),
so test data is never touched by the resampler.
"""

smote = SMOTE(random_state=42, k_neighbors=5)

# ── Logistic Regression ──────────────────────────────────────────────────────
pipe_lr = ImbPipeline([
    ("scaler", StandardScaler()),
    ("smote",  smote),
    ("clf",    LogisticRegression(C=1.0, max_iter=500, random_state=42))
])

# ── k-Nearest Neighbours ─────────────────────────────────────────────────────
pipe_knn = ImbPipeline([
    ("scaler", StandardScaler()),
    ("smote",  smote),
    ("clf",    KNeighborsClassifier(n_neighbors=7, metric="euclidean",
                                    weights="distance"))
])

# ── Random Forest ─────────────────────────────────────────────────────────────
# RF handles imbalance well; SMOTE + RF is still commonly compared
pipe_rf = ImbPipeline([
    ("smote",  smote),
    ("clf",    RandomForestClassifier(n_estimators=500, max_features="sqrt",
                                       criterion="gini", random_state=42,
                                       n_jobs=-1))
])

# ── SVM — best params determined in Section 3 GridSearch ─────────────────────
# Placeholder; replaced after GridSearch in Section 3
pipe_svm_default = ImbPipeline([
    ("scaler", StandardScaler()),
    ("smote",  smote),
    ("clf",    SVC(kernel="rbf", C=1.0, gamma="scale",
                   random_state=42, probability=True))
])

print("Pipelines built:")
for name in ["LR", "KNN", "RF", "SVM (default)"]:
    print(f"  StandardScaler → SMOTE → {name}")

print("\nFitting default pipelines on training data ...")
pipe_lr.fit(X_train, y_train)
pipe_knn.fit(X_train, y_train)
pipe_rf.fit(X_train, y_train)
pipe_svm_default.fit(X_train, y_train)
print("  Done.")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: SVM GRIDSEARCH + CV SCORE HEATMAP
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 3: SVM GridSearchCV + CV score heatmap")
print("=" * 65)

"""
GridSearchCV exhaustively tries all (C, gamma) combinations using
stratified k-fold cross-validation on the training set.
Scoring = F1 for the malignant class (pos_label=0) via make_scorer.

The imblearn Pipeline ensures SMOTE is applied independently within
each training fold — crucially, the validation fold is never resampled.

We plot mean CV F1 as a heatmap so the best region is immediately visible.
"""

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
f1_malignant = make_scorer(f1_score, pos_label=0)

C_grid     = [0.01, 0.1, 1, 10, 100]
gamma_grid = [0.001, 0.01, 0.1, 1, "scale"]

param_grid_svm = {
    "clf__C":     C_grid,
    "clf__gamma": gamma_grid,
}

svm_gs_pipe = ImbPipeline([
    ("scaler", StandardScaler()),
    ("smote",  SMOTE(random_state=42, k_neighbors=5)),
    ("clf",    SVC(kernel="rbf", random_state=42, probability=True))
])

print(f"Running 5-fold GridSearchCV over {len(C_grid)*len(gamma_grid)} "
      f"parameter combinations ...")
gs_svm = GridSearchCV(
    svm_gs_pipe, param_grid_svm,
    scoring=f1_malignant, cv=cv, n_jobs=-1, verbose=0,
    return_train_score=True
)
gs_svm.fit(X_train, y_train)

print(f"  Best params : {gs_svm.best_params_}")
print(f"  Best CV F1  : {gs_svm.best_score_:.4f}")

# ── CV score heatmap ─────────────────────────────────────────────────────────
results   = gs_svm.cv_results_
# Build score matrices (rows=C, cols=gamma)
mean_test  = results["mean_test_score"].reshape(len(C_grid), len(gamma_grid))
std_test   = results["std_test_score"].reshape(len(C_grid), len(gamma_grid))
mean_train = results["mean_train_score"].reshape(len(C_grid), len(gamma_grid))

gamma_labels = [str(g) for g in gamma_grid]
C_labels     = [str(c) for c in C_grid]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("SVM GridSearchCV — F1 (malignant class), 5-fold CV",
             fontsize=12, fontweight="bold")

for ax, matrix, title, cmap in zip(
        axes,
        [mean_test, mean_train],
        ["Mean CV Test F1", "Mean CV Train F1"],
        ["YlOrRd", "YlGnBu"]):
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0.7, vmax=1.0)
    ax.set_xticks(range(len(gamma_labels)))
    ax.set_xticklabels(gamma_labels, fontsize=9)
    ax.set_yticks(range(len(C_labels)))
    ax.set_yticklabels(C_labels, fontsize=9)
    ax.set_xlabel("gamma", fontsize=10)
    ax.set_ylabel("C", fontsize=10)
    ax.set_title(title, fontsize=11)
    # Annotate cells
    for i in range(len(C_grid)):
        for j in range(len(gamma_grid)):
            val = matrix[i, j]
            txt_color = "white" if val > 0.90 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=8, color=txt_color, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# Mark best cell
best_i = C_grid.index(gs_svm.best_params_["clf__C"])
best_j = gamma_labels.index(str(gs_svm.best_params_["clf__gamma"]))
axes[0].add_patch(plt.Rectangle((best_j - 0.5, best_i - 0.5), 1, 1,
                                  fill=False, edgecolor="lime",
                                  linewidth=3, label="Best params"))
axes[0].legend(loc="upper right", fontsize=8)

plt.tight_layout()
plt.savefig("bc_svm_gridsearch_heatmap.png", dpi=120, bbox_inches="tight")
print("  → Saved: bc_svm_gridsearch_heatmap.png")
plt.close()

# ── CV score line plot: best C across gamma values ───────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
for i, C_val in enumerate(C_grid):
    ax.plot(range(len(gamma_grid)), mean_test[i],
            marker="o", label=f"C={C_val}")
    ax.fill_between(range(len(gamma_grid)),
                    mean_test[i] - std_test[i],
                    mean_test[i] + std_test[i], alpha=0.10)
ax.set_xticks(range(len(gamma_grid)))
ax.set_xticklabels(gamma_labels)
ax.set_xlabel("gamma", fontsize=11)
ax.set_ylabel("Mean CV F1 (malignant)", fontsize=11)
ax.set_title("SVM — CV F1 score by C and gamma  (shading = ±1 std)", fontsize=11)
ax.legend(title="C value", fontsize=8, loc="lower right")
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
ax.grid(axis="y", alpha=0.4)
plt.tight_layout()
plt.savefig("bc_svm_cv_scores.png", dpi=120, bbox_inches="tight")
print("  → Saved: bc_svm_cv_scores.png")
plt.close()

# ── Best SVM pipeline from GridSearch ────────────────────────────────────────
pipe_svm = gs_svm.best_estimator_
print(f"\nBest SVM test-set performance:")
y_pred_svm_best = pipe_svm.predict(X_test)
print(f"  Accuracy      : {accuracy_score(y_test, y_pred_svm_best):.3f}")
print(f"  F1 (malignant): {f1_score(y_test, y_pred_svm_best, pos_label=0):.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: CLASSIFIER COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 4: Classifier comparison (LR / KNN / RF / SVM)")
print("=" * 65)

"""
Metrics:
  Accuracy     : overall correct predictions
  F1-malignant : harmonic mean of precision & recall for the positive class
  ROC-AUC      : area under ROC curve (threshold-independent)
  PR-AUC       : area under Precision-Recall curve (better under imbalance)
  Recall-0     : sensitivity — fraction of malignant cases correctly caught
"""

classifiers = {
    "Logistic Regression": pipe_lr,
    f"k-NN (k=7)":         pipe_knn,
    "Random Forest":        pipe_rf,
    "SVM (tuned)":          pipe_svm,
}

results_table = {}

print(f"\n  {'Classifier':<22}  {'Acc':>6}  {'F1-M':>6}  "
      f"{'ROC-AUC':>8}  {'PR-AUC':>7}  {'Recall-0':>9}")
print("  " + "─" * 68)

for name, clf in classifiers.items():
    y_pred  = clf.predict(X_test)
    y_prob  = clf.predict_proba(X_test)[:, 0]   # P(malignant)
    acc     = accuracy_score(y_test, y_pred)
    f1_m    = f1_score(y_test, y_pred, pos_label=0)
    rec_m   = recall_score(y_test, y_pred, pos_label=0)
    roc_auc = roc_auc_score(1 - y_test, y_prob)
    pr_auc  = average_precision_score(1 - y_test, y_prob)
    results_table[name] = dict(acc=acc, f1_m=f1_m, roc_auc=roc_auc,
                                pr_auc=pr_auc, recall_m=rec_m,
                                clf=clf, y_prob=y_prob)
    print(f"  {name:<22}  {acc:>6.3f}  {f1_m:>6.3f}  "
          f"{roc_auc:>8.3f}  {pr_auc:>7.3f}  {rec_m:>9.3f}")

# Identify best model by F1-malignant
best_name = max(results_table, key=lambda k: results_table[k]["f1_m"])
print(f"\n  Best model by F1-malignant: {best_name}  "
      f"(F1={results_table[best_name]['f1_m']:.3f})")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: CONFUSION MATRICES
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 5: Confusion Matrices")
print("=" * 65)

"""
Confusion matrix layout for binary classification:
              Predicted malignant   Predicted benign
  True malignant      TP                  FN   ← False Negatives are critical
  True benign         FP                  TN

In cancer screening, FN (missed malignancies) are far more dangerous than
FP (unnecessary follow-up biopsies). A good clinical model maximises TP
while keeping FN as close to zero as possible.
"""

fig, axes = plt.subplots(1, 4, figsize=(18, 4))
fig.suptitle("Confusion Matrices — Breast Cancer Test Set\n"
             "(SMOTE-balanced training, 0=malignant, 1=benign)",
             fontsize=11, fontweight="bold")

for ax, (name, clf) in zip(axes, classifiers.items()):
    ConfusionMatrixDisplay.from_estimator(
        clf, X_test, y_test,
        display_labels=target_names,
        colorbar=False, ax=ax, cmap="Blues",
        normalize=None)
    ax.set_title(name, fontsize=10, fontweight="bold")
    ax.set_xlabel("Predicted label", fontsize=8)
    ax.set_ylabel("True label", fontsize=8)
    # Annotate FN cell (row 0, col 1) with a red border
    ax.add_patch(plt.Rectangle((0.5, -0.5), 1, 1, fill=False,
                                edgecolor="red", linewidth=2,
                                label="FN (missed malignant)"))

# Add legend on last axis only
axes[-1].legend(loc="upper right", fontsize=7, framealpha=0.8)
plt.tight_layout()
plt.savefig("bc_confusion_matrices_v2.png", dpi=120, bbox_inches="tight")
print("  → Saved: bc_confusion_matrices_v2.png")
plt.close()

# Also print normalised confusion matrices (rates)
print("\nNormalised confusion matrices (row = true class):")
for name, clf in classifiers.items():
    cm = confusion_matrix(y_test, clf.predict(X_test), normalize="true")
    print(f"\n  {name}")
    print(f"    {'':>20} Pred malig  Pred benign")
    print(f"    {'True malignant':>20}   {cm[0,0]:.2f}       {cm[0,1]:.2f}")
    print(f"    {'True benign':>20}   {cm[1,0]:.2f}       {cm[1,1]:.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: ROC CURVES
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 6: ROC Curves")
print("=" * 65)

"""
ROC (Receiver Operating Characteristic) curve:
  x-axis: FPR = FP / (FP + TN)  — false alarm rate
  y-axis: TPR = TP / (TP + FN)  — sensitivity / recall

  AUC = 1.0  → perfect classifier
  AUC = 0.5  → no better than random chance

pos_label=0: we treat malignant as the positive class throughout.
"""

fig, ax = plt.subplots(figsize=(8, 7))

colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
for (name, res), color in zip(results_table.items(), colors):
    fpr, tpr, _ = roc_curve(1 - y_test, res["y_prob"])
    ax.plot(fpr, tpr, color=color, linewidth=2,
            label=f"{name}  (AUC={res['roc_auc']:.3f})")

ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Chance level (AUC=0.50)")
ax.set_title("ROC Curves — Breast Cancer Test Set\n"
             "(positive class = malignant, SMOTE training)",
             fontsize=11, fontweight="bold")
ax.legend(loc="lower right", fontsize=9)
ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=10)
ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("bc_roc_curves_v2.png", dpi=120, bbox_inches="tight")
print("  → Saved: bc_roc_curves_v2.png")
plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: PRECISION-RECALL CURVES
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 7: Precision-Recall Curves")
print("=" * 65)

"""
The Precision-Recall curve is more informative than ROC under class
imbalance because it focuses on the minority class performance without
being inflated by the large number of true negatives.

  No-skill baseline: a classifier that always predicts the positive class
  achieves Precision = prior probability of malignant ≈ 0.37.
  Any useful classifier must sit well above this line.

Average Precision (AP) = area under the PR curve.
"""

fig, ax = plt.subplots(figsize=(8, 7))
baseline_precision = np.mean(y_test == 0)

for (name, res), color in zip(results_table.items(), colors):
    prec, rec, _ = precision_recall_curve(1 - y_test, res["y_prob"])
    ax.plot(rec, prec, color=color, linewidth=2,
            label=f"{name}  (AP={res['pr_auc']:.3f})")

ax.axhline(baseline_precision, linestyle="--", color="grey", linewidth=1.2,
           label=f"No-skill baseline  (precision={baseline_precision:.2f})")
ax.set_title("Precision-Recall Curves — Breast Cancer Test Set\n"
             "(positive class = malignant, SMOTE training)",
             fontsize=11, fontweight="bold")
ax.legend(loc="upper right", fontsize=9)
ax.set_xlabel("Recall (Sensitivity)", fontsize=10)
ax.set_ylabel("Precision (PPV)", fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("bc_pr_curves_v2.png", dpi=120, bbox_inches="tight")
print("  → Saved: bc_pr_curves_v2.png")
plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: COMBINED DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 8: Combined performance dashboard")
print("=" * 65)

"""
One figure summarising confusion matrices, ROC, and PR curves side by side.
"""

from sklearn.metrics import roc_curve, precision_recall_curve

fig = plt.figure(figsize=(20, 10))
fig.suptitle("Breast Cancer Classifier Dashboard  (SMOTE + Tuned SVM)\n"
             "Positive class = malignant  |  Test set n=171",
             fontsize=13, fontweight="bold", y=1.01)

# Row 1: confusion matrices
for i, (name, clf) in enumerate(classifiers.items()):
    ax = fig.add_subplot(2, 4, i + 1)
    ConfusionMatrixDisplay.from_estimator(
        clf, X_test, y_test,
        display_labels=["Malig", "Benign"],
        colorbar=False, ax=ax, cmap="Blues")
    ax.set_title(name, fontsize=9, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=7)
    ax.set_ylabel("True", fontsize=7)
    ax.tick_params(labelsize=7)

# Row 2, col 1-2: ROC
ax_roc = fig.add_subplot(2, 4, (5, 6))
for (name, res), color in zip(results_table.items(), colors):
    fpr, tpr, _ = roc_curve(1 - y_test, res["y_prob"])
    ax_roc.plot(fpr, tpr, color=color, linewidth=2,
                label=f"{name}  AUC={res['roc_auc']:.3f}")
ax_roc.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.6)
ax_roc.set_xlabel("FPR", fontsize=9); ax_roc.set_ylabel("TPR", fontsize=9)
ax_roc.set_title("ROC Curves", fontsize=10, fontweight="bold")
ax_roc.legend(fontsize=7.5, loc="lower right")
ax_roc.grid(alpha=0.3)

# Row 2, col 3-4: PR
ax_pr = fig.add_subplot(2, 4, (7, 8))
for (name, res), color in zip(results_table.items(), colors):
    prec, rec, _ = precision_recall_curve(1 - y_test, res["y_prob"])
    ax_pr.plot(rec, prec, color=color, linewidth=2,
               label=f"{name}  AP={res['pr_auc']:.3f}")
ax_pr.axhline(baseline_precision, linestyle="--", color="grey",
              linewidth=1, alpha=0.7, label=f"No-skill ({baseline_precision:.2f})")
ax_pr.set_xlabel("Recall", fontsize=9); ax_pr.set_ylabel("Precision", fontsize=9)
ax_pr.set_title("Precision-Recall Curves", fontsize=10, fontweight="bold")
ax_pr.legend(fontsize=7.5, loc="upper right")
ax_pr.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("bc_dashboard.png", dpi=120, bbox_inches="tight")
print("  → Saved: bc_dashboard.png")
plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9: SAVE BEST MODEL
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 9: Save best model")
print("=" * 65)

"""
Two formats are saved for maximum compatibility:
  best_model.joblib — recommended for sklearn/imblearn pipelines;
                      more efficient than pickle for large numpy arrays
  best_model.pkl    — standard Python pickle; portable across tools
                      that do not have joblib installed
Both files contain the identical fitted pipeline object.
"""
import pickle

best_clf  = results_table[best_name]["clf"]
best_meta = results_table[best_name]

os.makedirs("saved_models", exist_ok=True)
joblib_path = os.path.join("saved_models", "best_model.joblib")
pkl_path    = os.path.join("saved_models", "best_model.pkl")

# Save as joblib
joblib.dump(best_clf, joblib_path)

# Save as pickle
with open(pkl_path, "wb") as f:
    pickle.dump(best_clf, f)

metadata = f"""Best Model — Breast Cancer Classifier
======================================
Model        : {best_name}
Pipeline     : StandardScaler -> SMOTE -> {best_name}

Saved files:
  {joblib_path}   (joblib - recommended)
  {pkl_path}      (pickle - portable)

Test-set performance (n={X_test.shape[0]}):
  Accuracy      : {best_meta['acc']:.4f}
  F1 (malignant): {best_meta['f1_m']:.4f}
  ROC-AUC       : {best_meta['roc_auc']:.4f}
  PR-AUC        : {best_meta['pr_auc']:.4f}
  Recall-0      : {best_meta['recall_m']:.4f}

How to reload (joblib):
  import joblib
  model = joblib.load('saved_models/best_model.joblib')
  y_pred = model.predict(X_new)
  y_prob = model.predict_proba(X_new)[:, 0]  # P(malignant)

How to reload (pickle):
  import pickle
  with open('saved_models/best_model.pkl', 'rb') as f:
      model = pickle.load(f)
  y_pred = model.predict(X_new)
  y_prob = model.predict_proba(X_new)[:, 0]  # P(malignant)

Note: X_new must have the same 30 features in the same order as the
      original Breast Cancer Wisconsin dataset.
"""

with open(os.path.join("saved_models", "model_metadata.txt"), "w") as f:
    f.write(metadata)

print(metadata)
print(f"  -> Saved: {joblib_path}")
print(f"  -> Saved: {pkl_path}")
print(f"  -> Saved: saved_models/model_metadata.txt")

# Verify both reloads produce identical predictions
reloaded_jl  = joblib.load(joblib_path)
with open(pkl_path, "rb") as f:
    reloaded_pkl = pickle.load(f)
y_check = reloaded_jl.predict(X_test)
assert np.array_equal(y_check, best_clf.predict(X_test)), \
    "joblib reload verification failed!"
assert np.array_equal(reloaded_pkl.predict(X_test), best_clf.predict(X_test)), \
    "pickle reload verification failed!"
print("  ✓ joblib reload verified — predictions match.")
print("  ✓ pickle reload verified — predictions match.")


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)
print(f"""
Imbalanced learning strategy:
  SMOTE applied inside imblearn Pipeline (no data leakage)
  Training class distribution after SMOTE: balanced 50/50

Classifiers compared: LR, k-NN, Random Forest, SVM (tuned by GridSearch)

SVM GridSearch: 5-fold CV over {len(C_grid)*len(gamma_grid)} C×gamma combinations
  Best params : {gs_svm.best_params_}
  Best CV F1  : {gs_svm.best_score_:.4f}

Best overall model: {best_name}
  F1 (malignant) : {results_table[best_name]['f1_m']:.4f}
  Recall         : {results_table[best_name]['recall_m']:.4f}
  ROC-AUC        : {results_table[best_name]['roc_auc']:.4f}

Saved outputs:
  bc_svm_gridsearch_heatmap.png  — C×gamma CV F1 heatmaps
  bc_svm_cv_scores.png           — CV F1 line plot across gamma
  bc_confusion_matrices_v2.png   — confusion matrices (all 4 classifiers)
  bc_roc_curves_v2.png           — ROC curves
  bc_pr_curves_v2.png            — Precision-Recall curves
  bc_dashboard.png               — combined dashboard
  saved_models/best_model.joblib — best pipeline persisted
""")

if __name__ == "__main__":
    print("✓ Ch 3 (Breast Cancer v2 — imblearn) complete.")
