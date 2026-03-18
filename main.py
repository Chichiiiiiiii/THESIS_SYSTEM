from flask import Flask, render_template, request
import joblib
import pandas as pd
import requests
from bs4 import BeautifulSoup
from googlesearch import search
from newspaper import Article, ArticleException
from nltk import pos_tag, WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer
import re
import emoji
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob, Word
from nltk import word_tokenize
from better_profanity import profanity

contractions_dict = {"ain't": "are not", "aren't": "are not", "It’s": "it is",
                     "can't": "cannot", "can't've": "cannot have",
                     "'cause": "because", "could've": "could have", "couldn't": "could not",
                     "couldn't've": "could not have", "didn't": "did not", "doesn't": "does not",
                     "don't": "do not", "hadn't": "had not", "hadn't've": "had not have",
                     "hasn't": "has not", "haven't": "have not", "he'd": "he would",
                     "he'd've": "he would have", "he'll": "he will", "he'll've": "he will have",
                     "how'd": "how did", "how'd'y": "how do you", "how'll": "how will",
                     "i'd": "i would", "i'd've": "i would have", "i'll": "i will",
                     "i'll've": "i will have", "i'm": "i am", "I've": "i have", "isn't": "is not",
                     "it'd": "it would", "it'd've": "it would have", "it'll": "it will",
                     "it'll've": "it will have", "let's": "let us", "ma'am": "madam",
                     "mayn't": "may not", "might've": "might have", "mightn't": "might not",
                     "mightn't've": "might not have", "must've": "must have", "mustn't": "must not",
                     "mustn't've": "must not have", "needn't": "need not",
                     "needn't've": "need not have", "o'clock": "of the clock", "oughtn't": "ought not",
                     "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not",
                     "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have",
                     "she'll": "she will", "she'll've": "she will have", "should've": "should have",
                     "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have",
                     "that'd": "that would", "that'd've": "that would have", "there'd": "there would",
                     "there'd've": "there would have", "they'd": "they would",
                     "they'd've": "they would have", "they'll": "they will",
                     "they'll've": "they will have", "they're": "they are", "they've": "they have",
                     "to've": "to have", "wasn't": "was not", "we'd": "we would",
                     "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have",
                     "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will",
                     "what'll've": "what will have", "what're": "what are", "what've": "what have",
                     "when've": "when have", "where'd": "where did", "where've": "where have",
                     "who'll": "who will", "who'll've": "who will have", "who've": "who have",
                     "why've": "why have", "will've": "will have", "won't": "will not",
                     "won't've": "will not have", "would've": "would have", "wouldn't": "would not",
                     "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
                     "y'all'd've": "you all would have", "y'all're": "you all are",
                     "y'all've": "you all have", "you'd": "you would", "you'd've": "you would have",
                     "you'll": "you will", "you'll've": "you will have", "you're": "you are"}


def correct_words(words):
    corrected_words = []
    for word in words:
        key = word.replace("’", "'")
        if key in contractions_dict.keys():
            corrected_words.append(contractions_dict[key])
        elif word in contractions_dict.keys():
            corrected_words.append(contractions_dict[word])
        else:
            corrected_words.append(word.replace("’s", " is").replace("'s", " is"))
    return corrected_words


# %%
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


def search_similar_articles(query, num_results=5):
    similar_articles = []
    for j, link in enumerate(search(query['title'], num_results=num_results, sleep_interval=2)):
        try:
            news_article = Article(link)
            news_article.download()
            news_article.parse()
            article_title = news_article.title
            if article_title and news_article.text:
                response = requests.get(link)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, "html.parser")
                    paragraphs = soup.find_all("p")
                    article_text = '\n'.join([paragraph.get_text() for paragraph in paragraphs])
                    processed_text1 = process_text(query['text'])
                    processed_text2 = process_text(process_text(article_text))
                    cv = CountVectorizer()
                    cv.fit([processed_text1, processed_text2])
                    vectorized_text1 = cv.transform([processed_text1])
                    vectorized_text2 = cv.transform([processed_text2])
                    similarity_score = cosine_similarity(vectorized_text1, vectorized_text2)[0][0]
                    similar_articles.append({"url": link, "article": article_title, "similarity": similarity_score})
        except ArticleException:
            continue
    similar_articles_df = pd.DataFrame(similar_articles)
    similar_articles_df = similar_articles_df.sort_values(by='similarity', ascending=False)
    return similar_articles_df


def fake_news_det(news):
    tfidf_v = joblib.load('saved_models/vectorizer.pkl')
    model = joblib.load('saved_models/passive_aggressive_classifier.pkl')
    processed_news = process_text(news)
    input_data = [processed_news]
    vectorized_input_data = tfidf_v.transform(input_data)
    prediction = model.predict(vectorized_input_data)
    return prediction


def get_sentiments(text):
    text = text.strip().lower()
    vs = TextBlob(text).sentiment[0]
    if vs > 0:
        return 'Positive'
    elif vs < 0:
        return 'Negative'
    else:
        return 'Neutral'


#For Spliting
def lemmatize_words(text):
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    lemmatized_words = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in words]
    return lemmatized_words


def get_wordnet_pos(word):
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)


file_path = r"words-slang.txt"
file = open(file_path, 'r')
slang_words = file.readlines()


def count_misspelled_words(text):
    text = clean_text(text)
    misspelled_words = 0
    total_words = lemmatize_words(text)
    for word in total_words:
        if word in slang_words:
            misspelled_words += 1
        else:
            word = Word(word)
            result = word.spellcheck()
            if word != result[0][0] and result[0][1] == 1:
                misspelled_words += 1
    return misspelled_words * 100 / len(total_words)


def count_offensive_words(text):
    profanity.load_censor_words()
    count = 0
    text = text.strip().lower()
    words = word_tokenize(text)
    for word in words:
        if profanity.contains_profanity(word) == 1:
            count += 1
    return count


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('main.html')


trusted_news_providers = [
    
]


@app.route('/result', methods=['POST'])
def predict():
    query = {}
    if request.method == 'POST':
        query['title'] = request.form['title']
        query['text'] = request.form['text']
        news = query['title'] + ' ' + query['text']
        sentiment_intensity = get_sentiments(news)
        count = round(count_misspelled_words(news), 2)
        misspelled_count = str(count) + '%'
        offensive_count = count_offensive_words(news)
        pred = fake_news_det(news)
        percentage = 80
        score = 0
        similarity_df = search_similar_articles(query)
        similarity_df = similarity_df.sort_values(by='similarity', ascending=False)
        domain = re.search(r"(https?://)?(www\d?\.)?(?P<name>[\w.-]+)\.\w+", similarity_df['url'][0])
        if domain and domain.group("name") not in trusted_news_providers and similarity_df['similarity'][0] > 0.8:
            score += 10
        if offensive_count > 0:
            score += 5
        if count > 5:
            score += 5
        if pred == 0:
            percentage += score
        else:
            percentage += 20 - score
        percentage = str(percentage) + '%'
        return render_template('result.html', prediction=pred, similarity_table=similarity_df,
                               sentiment_intensity=sentiment_intensity, misspelled_count=misspelled_count,
                               offensive_count=offensive_count, percentage=percentage)


if __name__ == '__main__':
    app.run(debug=True)
