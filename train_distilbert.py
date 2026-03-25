"""
train_distilbert.py
────────────────────
Fine-tunes a multilingual DistilBERT model on English + Tagalog fake-news data.

Requirements:
    pip install transformers datasets torch scikit-learn pandas

Usage:
    python train_distilbert.py --en data/english_news.csv --tl data/tagalog_news.csv
"""

import argparse
import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, classification_report)

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)

# ─────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────
MODEL_NAME   = "distilbert-base-multilingual-cased"   # multilingual → handles Tagalog & English
MAX_LEN      = 256
BATCH_SIZE   = 16
EPOCHS       = 4
LR           = 2e-5
WARMUP_RATIO = 0.1
DROPOUT      = 0.3
SAVE_DIR     = "models/distilbert_finetuned"
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"[Device] Using: {DEVICE}")

# ─────────────────────────────────────────────
#  Preprocessing
# ─────────────────────────────────────────────
def preprocess(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:1024]   # cap length

# ─────────────────────────────────────────────
#  Dataset
# ─────────────────────────────────────────────
class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=MAX_LEN):
        self.texts     = list(texts)
        self.labels    = list(labels)
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long)
        }

# ─────────────────────────────────────────────
#  Data loading
# ─────────────────────────────────────────────
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)[["text","label"]].dropna()
    if df["label"].dtype == object:
        df["label"] = df["label"].str.lower().map(
            {"fake":1,"false":1,"1":1,"real":0,"true":0,"0":0}
        )
    df["label"] = df["label"].astype(int)
    df["text"]  = df["text"].apply(preprocess)
    return df[df["text"].str.len() > 10]

# ─────────────────────────────────────────────
#  Training loop
# ─────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch in loader:
        optimizer.zero_grad()
        input_ids  = batch["input_ids"].to(DEVICE)
        attn_mask  = batch["attention_mask"].to(DEVICE)
        labels     = batch["labels"].to(DEVICE)
        outputs    = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
        loss       = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        preds       = outputs.logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)
    return total_loss / len(loader), correct / total

def eval_epoch(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            input_ids  = batch["input_ids"].to(DEVICE)
            attn_mask  = batch["attention_mask"].to(DEVICE)
            labels     = batch["labels"].to(DEVICE)
            outputs    = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
            total_loss += outputs.loss.item()
            preds       = outputs.logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return (
        total_loss / len(loader),
        accuracy_score(all_labels, all_preds),
        f1_score(all_labels, all_preds, zero_division=0),
        all_preds,
        all_labels
    )

# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────
def main(en_path, tl_path):
    frames = []
    for path, lang in [(en_path,"English"),(tl_path,"Tagalog")]:
        if path and os.path.exists(path):
            df = load_csv(path)
            print(f"[{lang}] {len(df)} samples")
            frames.append(df)

    if not frames:
        print("[!] No datasets found. Aborting.")
        return

    df = pd.concat(frames, ignore_index=True).sample(frac=1, random_state=42)
    print(f"[Combined] {len(df)} samples")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.15, random_state=42, stratify=df["label"]
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15/0.85, random_state=42, stratify=y_train
    )

    print(f"  Train={len(X_train)} | Val={len(X_val)} | Test={len(X_test)}")
    print(f"\n[Loading tokenizer] {MODEL_NAME} …")

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    train_ds = NewsDataset(X_train, y_train, tokenizer)
    val_ds   = NewsDataset(X_val,   y_val,   tokenizer)
    test_ds  = NewsDataset(X_test,  y_test,  tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"[Loading model] {MODEL_NAME} …")
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2,
        hidden_dropout_prob=DROPOUT,
        attention_probs_dropout_prob=DROPOUT
    ).to(DEVICE)

    total_steps  = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    best_f1 = 0.0
    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f"\n{'─'*55}")
    print(f"  Training for {EPOCHS} epochs on {DEVICE}")
    print(f"{'─'*55}")

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc       = train_epoch(model, train_loader, optimizer, scheduler)
        va_loss, va_acc, va_f1, _, _ = eval_epoch(model, val_loader)

        print(f"  Epoch {epoch}/{EPOCHS}  "
              f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  "
              f"val_loss={va_loss:.4f}  val_acc={va_acc:.4f}  val_f1={va_f1:.4f}")

        if va_f1 > best_f1:
            best_f1 = va_f1
            model.save_pretrained(SAVE_DIR)
            tokenizer.save_pretrained(SAVE_DIR)
            print(f"   ✅ Best model saved (val_f1={best_f1:.4f})")

    # Final test evaluation
    print(f"\n{'─'*55}")
    print("  Final Test Evaluation (best checkpoint)")
    print(f"{'─'*55}")
    model = DistilBertForSequenceClassification.from_pretrained(SAVE_DIR).to(DEVICE)
    _, te_acc, te_f1, preds, labels = eval_epoch(model, test_loader)

    print(f"  Accuracy : {te_acc:.4f}")
    print(f"  F1-Score : {te_f1:.4f}")
    print(f"  Precision: {precision_score(labels, preds, zero_division=0):.4f}")
    print(f"  Recall   : {recall_score(labels, preds, zero_division=0):.4f}")
    print(classification_report(labels, preds, target_names=["Real","Fake"], zero_division=0))
    print(f"\n✅ Fine-tuned DistilBERT saved to '{SAVE_DIR}/'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--en", default="data/english_news.csv")
    parser.add_argument("--tl", default="data/tagalog_news.csv")
    args = parser.parse_args()
    main(args.en, args.tl)
