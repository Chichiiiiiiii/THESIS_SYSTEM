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
df = pd.read_csv("Datasets/Taglish.csv", nrows=1000)

file_path = r"words-slang.txt"
file = open(file_path, 'r')
slang_words = file.readlines()

# ================= CONTRACTIONS =================
contractions_dict = { ... }   # keep your full dict here

def correct_words(words):
    corrected_words = []
    for word in words:
        key = word.replace("’", "'")
        if key in contractions_dict:
            corrected_words.append(contractions_dict[key])
        elif word in contractions_dict:
            corrected_words.append(contractions_dict[word])
        else:
            corrected_words.append(word.replace("’s", " is").replace("'s", " is"))
    return corrected_words

# ================= TEXT PROCESS =================
stemmer = PorterStemmer()

def process_text(text):
    text = ''.join([emoji.demojize(i, delimiters=(' ', ' ')) for i in text])
    words = text.strip().split()
    words = [word.lower() for word in words]
    words = [re.sub(r"[^a-z\s’'#@*0-9]+", '', word) for word in words]
    words = correct_words(words)
    text = ' '.join([re.sub(r"[’']", '', word) for word in words])
    text = re.sub(r'http\S+', '', text)
    tokens = word_tokenize(text)
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)

def clean_text(text):
    words = text.strip().split()
    words = [word.lower() for word in words]
    words = [re.sub(r"[^a-z\s’']+", '', word) for word in words]
    words = correct_words(words)
    text = ' '.join([re.sub(r"[’']", '', word) for word in words])
    text = re.sub(r'http\S+', '', text)
    return text

# ================= FEATURES =================
def search_similar_articles(query, num_results=5):
    similar_articles = []
    for link in search(query['title'], num_results=num_results, sleep_interval=2):
        try:
            news_article = Article(link)
            news_article.download()
            news_article.parse()

            if news_article.title and news_article.text:
                response = requests.get(link)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, "html.parser")
                    paragraphs = soup.find_all("p")
                    article_text = '\n'.join([p.get_text() for p in paragraphs])

                    processed_text1 = process_text(query['text'])
                    processed_text2 = process_text(article_text)

                    cv = CountVectorizer()
                    cv.fit([processed_text1, processed_text2])

                    v1 = cv.transform([processed_text1])
                    v2 = cv.transform([processed_text2])

                    sim = cosine_similarity(v1, v2)[0][0]
                    similar_articles.append(
                        {"url": link, "article": news_article.title, "similarity": sim}
                    )
        except ArticleException:
            continue

    df = pd.DataFrame(similar_articles) #will extract the files 
    if not df.empty:
        df = df.sort_values(by="similarity", ascending=False)
    return df

def fake_news_det(news):
    tfidf_v = joblib.load('saved_models/vectorizer.pkl')
    model = joblib.load('saved_models/passive_aggressive_classifier.pkl')
    processed_news = process_text(news)
    vectorized = tfidf_v.transform([processed_news])
    return model.predict(vectorized)

def get_sentiments(text): #Will lowercase depends on the mood.
    vs = TextBlob(text.lower()).sentiment[0]
    if vs > 0: return 'Positive'
    if vs < 0: return 'Negative'
    return 'Neutral'

def lemmatize_words(text):
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    return [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in words]

def get_wordnet_pos(word):
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,"N": wordnet.NOUN,"V": wordnet.VERB,"R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def count_misspelled_words(text):
    text = clean_text(text)
    total_words = lemmatize_words(text)
    miss = 0
    for word in total_words:
        if word in slang_words:
            miss += 1
        else:
            w = Word(word)
            result = w.spellcheck()
            if word != result[0][0] and result[0][1] == 1:
                miss += 1
    return miss * 100 / len(total_words) if total_words else 0

def count_offensive_words(text):
    profanity.load_censor_words()
    words = word_tokenize(text.lower())
    return sum(profanity.contains_profanity(w) for w in words)

print(df)
