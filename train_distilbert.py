"""
train_distilbert.py
────────────────────
Fine-tunes distilbert-base-multilingual-cased on the
already-cleaned dataset produced by preprocess.py.

Expected file:
    Datasets/cleaned_data.csv
    Columns: cleaned_text (or text), label

Usage:
    python train_distilbert.py
"""

import os
import pandas as pd
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


#  Config
CLEANED_DATA_PATH = "Datasets/cleaned_data.csv"
SAVE_DIR          = "saved_models/distilbert_finetuned"
CM_SAVE_PATH      = "saved_models/confusion_matrix.png"

MODEL_NAME   = "distilbert-base-multilingual-cased"
MAX_LEN      = 256
BATCH_SIZE   = 16
EPOCHS       = 4
LR           = 2e-5
WARMUP_RATIO = 0.1
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"[Device] {DEVICE}")
print(f"[Model ] {MODEL_NAME}")

#  Load already-cleaned dataset
if not os.path.exists(CLEANED_DATA_PATH):
    raise FileNotFoundError(
        f"\n[!] Cannot find '{CLEANED_DATA_PATH}'.\n"
        "    Please run preprocess.py first to generate the cleaned dataset."
    )

print(f"\n[Loading Cleaned Dataset] {CLEANED_DATA_PATH}")
df = pd.read_csv(CLEANED_DATA_PATH, low_memory=False)
print(f"  Columns found: {list(df.columns)}")

# Auto-detect text column
for col in ["final_text", "cleaned_text", "text", "content", "sentence", "Tweets", "message"]:
    if col in df.columns:
        df = df.rename(columns={col: "text"})
        print(f"  Text column detected: '{col}'")
        break

# Auto-detect label column
for col in ["label", "Label", "fake", "Fake", "class", "target"]:
    if col in df.columns:
        df = df.rename(columns={col: "label"})
        print(f"  Label column detected: '{col}'")
        break

# Keep only text and label
df = df[["text", "label"]].copy()
df = df.dropna()
df["text"] = df["text"].astype(str)

# Normalise labels → 0 / 1
if df["label"].dtype == object:
    df["label"] = df["label"].str.lower().map(
        {"fake": 1, "false": 1, "1": 1,
         "real": 0, "true":  0, "0": 0}
    )

# Drop rows where label could not be mapped
df = df.dropna(subset=["label"])
df["label"] = df["label"].astype(int)

# Remove empty text rows
df = df[df["text"].str.len() > 10].reset_index(drop=True)

print(f"  {len(df)} total samples  "
      f"(Fake={df.label.sum()}  Real={(df.label == 0).sum()})")


#  Train / Val / Test split  (70 / 15 / 15)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"],
    test_size=0.15, random_state=42, stratify=df["label"]
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train,
    test_size=0.15 / 0.85, random_state=42, stratify=y_train
)
print(f"  Train={len(X_train)}  Val={len(X_val)}  Test={len(X_test)}")


#  Dataset class
class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts     = list(texts)
        self.labels    = list(labels)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }


#  Tokenizer & DataLoaders
print(f"\n[Tokenizer] Loading {MODEL_NAME} ...")
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

train_loader = DataLoader(
    NewsDataset(X_train, y_train, tokenizer),
    batch_size=BATCH_SIZE, shuffle=True,  num_workers=0
)
val_loader = DataLoader(
    NewsDataset(X_val, y_val, tokenizer),
    batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)
test_loader = DataLoader(
    NewsDataset(X_test, y_test, tokenizer),
    batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)

#  Model
print(f"[Model] Loading {MODEL_NAME} ...")
model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2
).to(DEVICE)

total_steps  = len(train_loader) * EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)

optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
)


#  Training helpers
def train_one_epoch():
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch in train_loader:
        optimizer.zero_grad()
        ids  = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        lbls = batch["labels"].to(DEVICE)
        out  = model(input_ids=ids, attention_mask=mask, labels=lbls)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += out.loss.item()
        preds       = out.logits.argmax(dim=1)
        correct    += (preds == lbls).sum().item()
        total      += lbls.size(0)
    return total_loss / len(train_loader), correct / total


def evaluate(loader):
    model.eval()
    all_preds, all_labels, total_loss = [], [], 0.0
    with torch.no_grad():
        for batch in loader:
            ids  = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            lbls = batch["labels"].to(DEVICE)
            out  = model(input_ids=ids, attention_mask=mask, labels=lbls)
            total_loss += out.loss.item()
            all_preds.extend(out.logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(lbls.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, zero_division=0)
    return total_loss / len(loader), acc, f1, all_preds, all_labels


#  Confusion matrix helper
def plot_confusion_matrix(labels, preds, save_path):
    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()

    print(f"\n{'─'*55}")
    print("  Confusion Matrix")
    print(f"{'─'*55}")
    print(f"                 Predicted Real   Predicted Fake")
    print(f"  Actual Real       {tn:^10}       {fp:^10}")
    print(f"  Actual Fake       {fn:^10}       {tp:^10}")
    print(f"\n  True Negatives  (TN) = {tn}  — Real correctly identified")
    print(f"  False Positives (FP) = {fp}  — Real wrongly marked as Fake")
    print(f"  False Negatives (FN) = {fn}  — Fake wrongly marked as Real")
    print(f"  True Positives  (TP) = {tp}  — Fake correctly identified")

    # Plot and save
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Real", "Fake"],
        yticklabels=["Real", "Fake"],
        linewidths=0.5,
        linecolor="gray",
        annot_kws={"size": 14, "weight": "bold"},
    )
    plt.title("DistilBERT — Confusion Matrix", fontsize=14, fontweight="bold", pad=14)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("Actual Label",    fontsize=12)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\n  Confusion matrix saved → '{save_path}'")

#  Training loop
os.makedirs(SAVE_DIR, exist_ok=True)
best_f1 = 0.0

print(f"\n{'─'*55}")
print(f"  Training DistilBERT  |  {EPOCHS} epochs  |  {DEVICE}")
print(f"{'─'*55}")

for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_acc              = train_one_epoch()
    va_loss, va_acc, va_f1, _, _ = evaluate(val_loader)

    print(f"  Epoch {epoch}/{EPOCHS}"
          f"  train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}"
          f"  val_loss={va_loss:.4f}  val_acc={va_acc:.4f}  val_f1={va_f1:.4f}")

    if va_f1 > best_f1:
        best_f1 = va_f1
        model.save_pretrained(SAVE_DIR)
        tokenizer.save_pretrained(SAVE_DIR)
        print(f"   ✔ Best model saved  (val_f1={best_f1:.4f})")


#  Final Test Evaluation
print(f"\n{'─'*55}")
print("  Final Test Evaluation")
print(f"{'─'*55}")

model = DistilBertForSequenceClassification.from_pretrained(SAVE_DIR).to(DEVICE)
_, te_acc, te_f1, preds, labels = evaluate(test_loader)

print(f"  Accuracy : {te_acc:.4f}")
print(f"  F1-Score : {te_f1:.4f}")
print(f"  Precision: {precision_score(labels, preds, zero_division=0):.4f}")
print(f"  Recall   : {recall_score(labels, preds, zero_division=0):.4f}")
print()
print(classification_report(labels, preds,
                             target_names=["Real", "Fake"],
                             zero_division=0))

# Plot and save confusion matrix
plot_confusion_matrix(labels, preds, CM_SAVE_PATH)

print(f"\n Fine-tuned DistilBERT saved to '{SAVE_DIR}/'")