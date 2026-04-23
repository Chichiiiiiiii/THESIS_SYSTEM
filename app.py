"""
app.py
───────
Flask web application for fake news detection.
Connects trained DistilBERT + Traditional ML models.
Models: Logistic Regression, Naive Bayes, SVM, Random Forest, DistilBERT

Run:
    python app.py
"""

from flask import Flask, render_template, request, jsonify
import joblib
import os
import re
import string
import torch
import numpy as np
from newspaper import Article, ArticleException
from bs4 import BeautifulSoup
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
)

# ─────────────────────────────────────────────
#  Paths
# ─────────────────────────────────────────────
DISTILBERT_DIR = "saved_models/distilbert_finetuned"
TFIDF_PATH     = "saved_models/tfidf_vectorizer.pkl"
LR_PATH        = "saved_models/logistic_regression.pkl"
SVM_PATH       = "saved_models/svm.pkl"
NB_PATH        = "saved_models/naive_bayes.pkl"
RF_PATH        = "saved_models/random_forest.pkl"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────
#  Stopwords (English + Tagalog)
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
#  Preprocessing
# ─────────────────────────────────────────────
def clean_text(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    text = str(text).lower()
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\W", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [w for w in text.split() if w not in ALL_STOPWORDS and len(w) > 2]
    return " ".join(tokens)

# ─────────────────────────────────────────────
#  Load Traditional ML Models
# ─────────────────────────────────────────────
def load_pkl(path):
    if os.path.exists(path):
        return joblib.load(path)
    print(f"[!] Model not found: {path}")
    return None

print("[Loading] Traditional ML models...")
tfidf_vectorizer = load_pkl(TFIDF_PATH)
lr_model         = load_pkl(LR_PATH)
svm_model        = load_pkl(SVM_PATH)
nb_model         = load_pkl(NB_PATH)
rf_model         = load_pkl(RF_PATH)

ML_CLASSIFIERS = {
    "Logistic Regression": lr_model,
    "Naive Bayes":         nb_model,
    "SVM":                 svm_model,
    "Random Forest":       rf_model,
}

# ─────────────────────────────────────────────
#  Load DistilBERT Model
# ─────────────────────────────────────────────
distilbert_model     = None
distilbert_tokenizer = None

if os.path.exists(DISTILBERT_DIR):
    print("[Loading] DistilBERT model...")
    distilbert_tokenizer = DistilBertTokenizerFast.from_pretrained(DISTILBERT_DIR)
    distilbert_model     = DistilBertForSequenceClassification.from_pretrained(DISTILBERT_DIR)
    distilbert_model.eval()
    distilbert_model.to(DEVICE)
    print("[OK] DistilBERT loaded.")
else:
    print(f"[!] DistilBERT not found at '{DISTILBERT_DIR}'. Train it first.")

# ─────────────────────────────────────────────
#  Prediction Functions
# ─────────────────────────────────────────────
def predict_traditional(text: str) -> dict:
    if tfidf_vectorizer is None:
        return {}
    cleaned = clean_text(text)
    vec     = tfidf_vectorizer.transform([cleaned])
    results = {}
    for name, model in ML_CLASSIFIERS.items():
        if model is None:
            continue
        pred = model.predict(vec)[0]
        try:
            prob = model.predict_proba(vec)[0]
            conf = round(float(max(prob)) * 100, 1)
        except AttributeError:
            conf = 100.0
        results[name] = {
            "label":      "FAKE" if pred == 1 else "REAL",
            "confidence": conf
        }
    return results


def predict_distilbert(text: str) -> dict:
    if distilbert_model is None or distilbert_tokenizer is None:
        return {}
    enc = distilbert_tokenizer(
        text,
        max_length=256,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    input_ids = enc["input_ids"].to(DEVICE)
    attn_mask = enc["attention_mask"].to(DEVICE)
    with torch.no_grad():
        logits = distilbert_model(
            input_ids=input_ids,
            attention_mask=attn_mask
        ).logits
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred  = int(np.argmax(probs))
    conf  = round(float(max(probs)) * 100, 1)
    return {
        "label":      "FAKE" if pred == 1 else "REAL",
        "confidence": conf
    }


def ensemble_vote(ml_results: dict, distilbert_result: dict) -> dict:
    all_results = {**ml_results}
    if distilbert_result:
        all_results["DistilBERT"] = distilbert_result

    if not all_results:
        return {"label": "INDETERMINATE", "confidence": 0,
                "fake_votes": 0, "real_votes": 0}

    fake_votes = sum(1 for v in all_results.values() if v["label"] == "FAKE")
    real_votes = len(all_results) - fake_votes
    avg_conf   = round(float(np.mean([v["confidence"] for v in all_results.values()])), 1)
    label      = "FAKE" if fake_votes > real_votes else "REAL"

    return {
        "label":      label,
        "confidence": avg_conf,
        "fake_votes": fake_votes,
        "real_votes": real_votes
    }

# ─────────────────────────────────────────────
#  Web Scraping
# ─────────────────────────────────────────────
def scrape_article(url: str) -> dict:
    try:
        article = Article(url)
        article.download()
        article.parse()
        return {
            "title":   article.title   or "No title found",
            "authors": article.authors or [],
            "date":    str(article.publish_date)[:10] if article.publish_date else "Unknown",
            "text":    article.text    or "",
            "error":   None
        }
    except ArticleException as e:
        return {"title": "", "authors": [], "date": "", "text": "", "error": str(e)}
    except Exception as e:
        return {"title": "", "authors": [], "date": "", "text": "", "error": str(e)}

# ─────────────────────────────────────────────
#  Flask App
# ─────────────────────────────────────────────
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("main.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    url = request.form.get("url", "").strip()

    if not url:
        return render_template("main.html",
                               error="Please enter a news article URL.")

    meta = scrape_article(url)

    if meta["error"] or not meta["text"]:
        return render_template("main.html",
                               error="Could not extract article. "
                                     "Please check the URL and try again.")

    text = meta["text"]

    ml_results        = predict_traditional(text)
    distilbert_result = predict_distilbert(text)
    final             = ensemble_vote(ml_results, distilbert_result)

    return render_template(
        "result.html",
        url               = url,
        title             = meta["title"],
        authors           = ", ".join(meta["authors"]) if meta["authors"] else "Unknown",
        date              = meta["date"],
        preview           = text[:500] + ("..." if len(text) > 500 else ""),
        ml_results        = ml_results,
        distilbert_result = distilbert_result,
        final             = final,
    )

@app.route("/health")
def health():
    return jsonify({
        "status":            "ok",
        "distilbert_loaded": distilbert_model is not None,
        "tfidf_loaded":      tfidf_vectorizer is not None,
    })

if __name__ == "__main__":
    app.run(debug=True)