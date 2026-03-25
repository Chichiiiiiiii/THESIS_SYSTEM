"""
train_distilbert.py
────────────────────
Fine-tunes distilbert-base-multilingual-cased on English + Taglish
fake-news datasets. Matches the existing project structure:
    Datasets/English.csv
    Datasets/Taglish.csv

Expected CSV columns (auto-detected):
    text / content / sentence / Tweets / message / statement  → article text
    label  →  0 = Real, 1 = Fake  (or "real"/"fake" strings)

Usage:
    python train_distilbert.py

After training, the model is saved to:
    saved_models/distilbert_finetuned/
"""

import os
import re
import string
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from bs4 import BeautifulSoup

# ─────────────────────────────────────────────
#  Config  (change these if needed)
# ─────────────────────────────────────────────
TAGLISH_PATH = "Datasets/Taglish.csv"
ENGLISH_PATH = "Datasets/English.csv"
SAVE_DIR     = "saved_models/distilbert_finetuned"

MODEL_NAME   = "distilbert-base-multilingual-cased"  # handles Tagalog + English
MAX_LEN      = 256
BATCH_SIZE   = 16
EPOCHS       = 4
LR           = 2e-5
WARMUP_RATIO = 0.1
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"[Device] {DEVICE}")
print(f"[Model ] {MODEL_NAME}")

# ─────────────────────────────────────────────
#  Stopwords  (English + Tagalog)
# ─────────────────────────────────────────────
STOP_EN = {
    "i","me","my","myself","we","our","ours","ourselves","you","your",
    "the","is","at","which","on","a","an","and","or","but","in","of",
    "to","for","with","as","by","that","this","it","its","be","are",
    "was","were","been","being","have","has","had","do","does","did",
    "will","would","could","should","may","might","shall","can","need"
}
STOP_TL = {
    "ang","mga","ng","sa","at","na","si","ni","kay","para","ay",
    "ito","nito","nila","siya","sila","kami","tayo","kayo","ako",
    "mo","ko","niya","namin","natin","ninyo","nila","din","rin",
    "lang","po","ho","ba","nga","pala","kasi","pero","dahil","kung"
}
ALL_STOPWORDS = STOP_EN | STOP_TL

# ─────────────────────────────────────────────
#  Preprocessing  (matches preprocess.py style)
# ─────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Same cleaning logic as the existing preprocess.py."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = BeautifulSoup(text, "html.parser").get_text()   # remove HTML
    text = re.sub(r"http\S+|www\S+", " ", text)            # remove URLs
    text = re.sub(r"\d+", "", text)                         # remove digits
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\W", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    # remove stopwords
    tokens = [w for w in text.split() if w not in ALL_STOPWORDS and len(w) > 2]
    return " ".join(tokens)

# ─────────────────────────────────────────────
#  Column detection  (matches preprocess.py)
# ─────────────────────────────────────────────
def find_text_col(df: pd.DataFrame) -> str:
    targets = ["text","content","sentence","Tweets","message","statement"]
    for col in targets:
        if col in df.columns:
            return col
    return df.columns[0]

def find_label_col(df: pd.DataFrame) -> str:
    targets = ["label","Label","fake","Fake","class","Class","target"]
    for col in targets:
        if col in df.columns:
            return col
    return df.columns[-1]

# ─────────────────────────────────────────────
#  Load & prepare data
# ─────────────────────────────────────────────
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    text_col  = find_text_col(df)
    label_col = find_label_col(df)
    print(f"  Columns detected → text: '{text_col}'  label: '{label_col}'")
    df = df[[text_col, label_col]].copy()
    df.columns = ["text", "label"]
    df = df.dropna()

    # Normalise labels → 0 / 1
    if df["label"].dtype == object:
        df["label"] = df["label"].str.lower().map(
            {"fake": 1, "false": 1, "1": 1,
             "real": 0, "true": 0, "0": 0}
        )
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    # Clean text
    df["text"] = df["text"].apply(clean_text)
    df = df[df["text"].str.len() > 10]
    return df

frames = []
for path, lang in [(ENGLISH_PATH, "English"), (TAGLISH_PATH, "Taglish")]:
    if os.path.exists(path):
        print(f"\n[Loading {lang}] {path}")
        df = load_csv(path)
        print(f"  {len(df)} samples  "
              f"(Fake={df.label.sum()}  Real={(df.label==0).sum()})")
        frames.append(df)
    else:
        print(f"[!] {path} not found — skipping.")

if not frames:
    raise FileNotFoundError(
        "No datasets found. Make sure Datasets/English.csv "
        "and Datasets/Taglish.csv exist."
    )

df = pd.concat(frames, ignore_index=True).sample(frac=1, random_state=42)
print(f"\n[Combined] {len(df)} total samples")

# 70 / 15 / 15 split
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"],
    test_size=0.15, random_state=42, stratify=df["label"]
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train,
    test_size=0.15 / 0.85, random_state=42, stratify=y_train
)
print(f"  Train={len(X_train)}  Val={len(X_val)}  Test={len(X_test)}")

# ─────────────────────────────────────────────
#  Dataset class
# ─────────────────────────────────────────────
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

# ─────────────────────────────────────────────
#  Tokenizer & DataLoaders
# ─────────────────────────────────────────────
print(f"\n[Tokenizer] Loading {MODEL_NAME} …")
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

train_loader = DataLoader(
    NewsDataset(X_train, y_train, tokenizer),
    batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)
val_loader = DataLoader(
    NewsDataset(X_val, y_val, tokenizer),
    batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)
test_loader = DataLoader(
    NewsDataset(X_test, y_test, tokenizer),
    batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)

# ─────────────────────────────────────────────
#  Model
# ─────────────────────────────────────────────
print(f"[Model] Loading {MODEL_NAME} …")
model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
).to(DEVICE)

total_steps  = len(train_loader) * EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)

optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
)

# ─────────────────────────────────────────────
#  Training helpers
# ─────────────────────────────────────────────
def train_one_epoch():
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch in train_loader:
        optimizer.zero_grad()
        ids   = batch["input_ids"].to(DEVICE)
        mask  = batch["attention_mask"].to(DEVICE)
        lbls  = batch["labels"].to(DEVICE)
        out   = model(input_ids=ids, attention_mask=mask, labels=lbls)
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

# ─────────────────────────────────────────────
#  Training loop
# ─────────────────────────────────────────────
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs("saved_models", exist_ok=True)

best_f1 = 0.0
print(f"\n{'─'*55}")
print(f"  Training DistilBERT  |  {EPOCHS} epochs  |  {DEVICE}")
print(f"{'─'*55}")

for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_acc             = train_one_epoch()
    va_loss, va_acc, va_f1, _, _ = evaluate(val_loader)

    print(f"  Epoch {epoch}/{EPOCHS}"
          f"  train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}"
          f"  val_loss={va_loss:.4f}  val_acc={va_acc:.4f}  val_f1={va_f1:.4f}")

    if va_f1 > best_f1:
        best_f1 = va_f1
        model.save_pretrained(SAVE_DIR)
        tokenizer.save_pretrained(SAVE_DIR)
        print(f"   ✔ Best model saved  (val_f1={best_f1:.4f})")

# ─────────────────────────────────────────────
#  Final test evaluation
# ─────────────────────────────────────────────
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

print(f"\n✅  Fine-tuned DistilBERT saved to '{SAVE_DIR}/'")
print("    Load it later with:")
print("    from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast")
print(f"    model     = DistilBertForSequenceClassification.from_pretrained('{SAVE_DIR}')")
print(f"    tokenizer = DistilBertTokenizerFast.from_pretrained('{SAVE_DIR}')")