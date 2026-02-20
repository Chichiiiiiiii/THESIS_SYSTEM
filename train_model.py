import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from preprocess import clean_text
from html.parser import HTMLParser

df = pd.read_csv("Taglish.csv", nrows=1000)

# lowercase ALL text columns
for col in df.columns:
    df[col] = df[col].astype(str).str.lower()
    df[col] = df[col].astype(str).str.replace(" ", "")


print(df)


