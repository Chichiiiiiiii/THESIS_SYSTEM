"""
ensemble.py
────────────
Trains traditional ML classifiers + Ensemble on cleaned_data.csv.
Models: Logistic Regression, Naive Bayes, SVM, Random Forest
Shows per-model results AND overall ensemble results for comparison.

Usage:
    python ensemble.py
"""

import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────
DATA_PATH  = "Datasets/cleaned_data.csv"
SAVE_DIR   = "saved_models"
os.makedirs(SAVE_DIR, exist_ok=True)

# ─────────────────────────────────────────────
#  1. Load Data
# ─────────────────────────────────────────────
print("=" * 60)
print("  Loading dataset...")
print("=" * 60)

df     = pd.read_csv(DATA_PATH)
X_text = df['final_text'].fillna("")
y      = df['label']

print(f"  Total samples : {len(df)}")
print(f"  Fake          : {(y == 1).sum()}")
print(f"  Real          : {(y == 0).sum()}")

# ─────────────────────────────────────────────
#  2. TF-IDF Vectorization
# ─────────────────────────────────────────────
print("\n[1/6] Fitting TF-IDF vectorizer...")
tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
X     = tfidf.fit_transform(X_text)

# ─────────────────────────────────────────────
#  3. Train / Test Split  (80 / 20)
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train={X_train.shape[0]}  Test={X_test.shape[0]}")

# ─────────────────────────────────────────────
#  Helper: evaluate and print results
# ─────────────────────────────────────────────
def evaluate_model(name, model, X_test, y_test, save_cm=True):
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    prec   = precision_score(y_test, y_pred, zero_division=0)
    rec    = recall_score(y_test, y_pred, zero_division=0)
    f1     = f1_score(y_test, y_pred, zero_division=0)
    cm     = confusion_matrix(y_test, y_pred)

    print(f"\n{'─' * 60}")
    print(f"  {name}")
    print(f"{'─' * 60}")
    print(f"  Accuracy : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1-Score : {f1:.4f}")
    print()
    print(classification_report(y_test, y_pred,
                                target_names=["Real", "Fake"],
                                zero_division=0))

    tn, fp, fn, tp = cm.ravel()
    print(f"  Confusion Matrix:")
    print(f"                   Predicted Real   Predicted Fake")
    print(f"  Actual Real          {tn:<10}       {fp:<10}")
    print(f"  Actual Fake          {fn:<10}       {tp:<10}")
    print(f"\n  TN={tn}  FP={fp}  FN={fn}  TP={tp}")

    if save_cm:
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Real", "Fake"],
                    yticklabels=["Real", "Fake"],
                    linewidths=0.5, annot_kws={"size": 13, "weight": "bold"})
        plt.title(f"{name} — Confusion Matrix", fontsize=13, fontweight="bold")
        plt.xlabel("Predicted Label")
        plt.ylabel("Actual Label")
        plt.tight_layout()
        fname = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        plt.savefig(os.path.join(SAVE_DIR, f"cm_{fname}.png"), dpi=150)
        plt.close()

    return {"model": name, "accuracy": acc, "precision": prec,
            "recall": rec, "f1": f1}

# ─────────────────────────────────────────────
#  4. Train & Evaluate Each Model
# ─────────────────────────────────────────────
results = []

# ── Logistic Regression ──
print("\n[2/6] Training Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
lr_model.fit(X_train, y_train)
results.append(evaluate_model("Logistic Regression", lr_model, X_test, y_test))

# ── Naive Bayes ──
print("\n[3/6] Training Naive Bayes...")
nb_model = MultinomialNB(alpha=0.1)
nb_model.fit(X_train, y_train)
results.append(evaluate_model("Naive Bayes", nb_model, X_test, y_test))

# ── SVM ──
print("\n[4/6] Training SVM...")
svm_model = CalibratedClassifierCV(LinearSVC(max_iter=2000, C=1.0))
svm_model.fit(X_train, y_train)
results.append(evaluate_model("SVM", svm_model, X_test, y_test))

# ── Random Forest ──
print("\n[5/6] Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
rf_model.fit(X_train, y_train)
results.append(evaluate_model("Random Forest", rf_model, X_test, y_test))

# ── Ensemble (Voting Classifier) ──
print("\n[6/6] Training Ensemble (Voting Classifier)...")
ensemble_model = VotingClassifier(
    estimators=[
        ('lr',  lr_model),
        ('nb',  nb_model),
        ('svm', svm_model),
        ('rf',  rf_model),
    ],
    voting='soft'
)
ensemble_model.fit(X_train, y_train)
results.append(evaluate_model("Ensemble (All Models)", ensemble_model, X_test, y_test))

# ─────────────────────────────────────────────
#  5. Summary Comparison Table
# ─────────────────────────────────────────────
print(f"\n{'=' * 60}")
print("  SUMMARY — All Models Comparison")
print(f"{'=' * 60}")
print(f"  {'Model':<28} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8}")
print(f"  {'─'*28} {'─'*9} {'─'*10} {'─'*8} {'─'*8}")

for r in results:
    marker = " ✔" if r["model"] == "Ensemble (All Models)" else ""
    print(f"  {r['model']:<28} {r['accuracy']*100:>8.2f}%"
          f" {r['precision']*100:>9.2f}%"
          f" {r['recall']*100:>7.2f}%"
          f" {r['f1']*100:>7.2f}%{marker}")

best = max(results[:-1], key=lambda x: x['f1'])
ens  = results[-1]
print(f"\n  Best individual model : {best['model']} (F1={best['f1']*100:.2f}%)")
print(f"  Ensemble performance  : F1={ens['f1']*100:.2f}%")

# ─────────────────────────────────────────────
#  6. Save comparison CSV
# ─────────────────────────────────────────────
results_df = pd.DataFrame(results)
results_df['accuracy']  = results_df['accuracy'].map(lambda x: f"{x*100:.2f}%")
results_df['precision'] = results_df['precision'].map(lambda x: f"{x*100:.2f}%")
results_df['recall']    = results_df['recall'].map(lambda x: f"{x*100:.2f}%")
results_df['f1']        = results_df['f1'].map(lambda x: f"{x*100:.2f}%")
results_df.columns      = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
csv_path = os.path.join(SAVE_DIR, "model_comparison.csv")
results_df.to_csv(csv_path, index=False)
print(f"\n  Comparison table saved → '{csv_path}'")

#  7. Save all models
print(f"\n{'=' * 60}")
print("  Saving models...")
print(f"{'=' * 60}")

joblib.dump(tfidf,          os.path.join(SAVE_DIR, "tfidf_vectorizer.pkl"))
joblib.dump(lr_model,       os.path.join(SAVE_DIR, "logistic_regression.pkl"))
joblib.dump(nb_model,       os.path.join(SAVE_DIR, "naive_bayes.pkl"))
joblib.dump(svm_model,      os.path.join(SAVE_DIR, "svm.pkl"))
joblib.dump(rf_model,       os.path.join(SAVE_DIR, "random_forest.pkl"))
joblib.dump(ensemble_model, os.path.join(SAVE_DIR, "ensemble_model.pkl"))

print(f"  All models saved to '{SAVE_DIR}/'")
print(f"\n✅ Training complete!")