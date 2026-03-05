import re
import pandas as pd
import requests
import joblib
import emoji
from bs4 import BeautifulSoup
from googlesearch import search
from newspaper import Article, ArticleException
from nltk import pos_tag, word_tokenize
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer
from nltk import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob, Word 
from better_profanity import profanity

# ================= DATA =================
df_tl = pd.read_csv("Datasets/Taglish.csv", nrows=1000)
df_en = pd.read_csv("Datasets/English.csv", nrows=1000)

# --- Standardize Tagalog dataset ---
if "article" in df_tl.columns:
    df_tl = df_tl.rename(columns={"article": "text"})

if df_tl["label"].dtype != object:
    df_tl["label"] = df_tl["label"].map({0: "FAKE", 1: "REAL"})

df_tl["language"] = "tl"

# defragment frame (fix warning)
df_tl = df_tl.copy()

# --- Standardize English dataset ---
if "article" in df_en.columns and "text" not in df_en.columns:
    df_en = df_en.rename(columns={"article": "text"})

df_en = df_en[["text", "label"]]
df_en["language"] = "en"

# --- Merge bilingual dataset ---
df = pd.concat([df_tl, df_en], ignore_index=True)
df = df.dropna(subset=["text", "label"])

# --- Slang dictionary ---
file_path = r"words-slang.txt"
with open(file_path, "r", encoding="utf-8") as f:
    slang_words = set(w.strip().lower() for w in f)


