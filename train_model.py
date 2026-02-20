import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from preprocess import clean_text
from html.parser import HTMLParser

df = pd.read_csv("Taglish.csv", nrows=1000)

df["title"]=df["title"].str.lower()
df["text"]=df["text"].str.lower()

print(df.head())
