import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from sklearn.svm import LinearSVC
from preprocess import clean_text
from html.parser import HTMLParser

df = pd.read_csv("Taglish.csv", nrows=1000)

# lowercase ALL text columns
for col in df.columns:
    df[col] = df[col].astype(str)
    df[col] = df[col].str.lower()
    df[col] = df[col].str.replace(r"[^a-z0-9\s]", "", regex=True)
    #df[col] = df[col].astype(str).str.replace(" ", "") # for spacing columns

# tokenization ALL text columns // df["tokens"] = df["text"].apply(lambda x: re.findall(r"\b\w+\b", x))

print(df)


