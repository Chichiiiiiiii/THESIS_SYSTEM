import pandas as pd
import re
import string
from bs4 import BeautifulSoup
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

# 1. Load data
df_tl = pd.read_csv("Datasets/Taglish.csv", nrows=1000)
df_en = pd.read_csv("Datasets/English.csv", nrows=1000)

# 2. Define Cleaning Function
def clean_text(text):
    if pd.isna(text): return ""
    text = str(text).lower()
    # Remove HTML
    text = BeautifulSoup(text, "html.parser").get_text()
    # Remove Digits
    text = re.sub(r'\d+', '', text)
    # Remove Punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove non-word characters
    text = re.sub(r'\W', ' ', text)
    return text.strip()

# 3. Function to find the text column
def find_text_col(df):
    targets = ['text', 'content', 'sentence', 'Tweets', 'message', 'statement']
    for col in targets:
        if col in df.columns:
            return col
    return df.columns[0]

col_tl = find_text_col(df_tl)
col_en = find_text_col(df_en)

# 4. APPLY CLEANING FIRST (Produces Clean Strings)
print(f"Cleaning Taglish using column: '{col_tl}'")
df_tl['cleaned_text'] = df_tl[col_tl].apply(clean_text)

print(f"Cleaning English using column: '{col_en}'")
df_en['cleaned_text'] = df_en[col_en].apply(clean_text)

# 5. APPLY TOKENIZER SECOND (Produces Lists of Words)
# We apply this to the 'cleaned_text' column we just created
tokenizer = RegexpTokenizer(r'\w+')

df_tl['tokens'] = df_tl['cleaned_text'].apply(lambda x: tokenizer.tokenize(x))
df_en['tokens'] = df_en['cleaned_text'].apply(lambda x: tokenizer.tokenize(x))

# 6. Instead of downloading, just use a hardcoded set
stop_words_en = {"i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "the", "is", "at", "which", "on"}
stop_words_tl = {"ang", "mga", "ng", "sa", "at", "na", "si", "ni", "kay", "para", "ay", "ito", "nito", "nila"}

all_stopwords = stop_words_en.union(stop_words_tl)

#tokenization step:
df_tl['tokens'] = df_tl['cleaned_text'].apply(lambda x: [w for w in x.split() if w not in all_stopwords])
df_en['tokens'] = df_en['cleaned_text'].apply(lambda x: [w for w in x.split() if w not in all_stopwords])

# 7. De-fragment and Final Check
df_tl = df_tl.copy()
df_en = df_en.copy()

print("\n--- Taglish Result ---")
print(df_tl[[col_tl, 'tokens']].head(3))

print("\n--- English Result ---")
print(df_en[[col_en, 'tokens']].head(3))