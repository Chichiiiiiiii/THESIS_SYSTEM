from flask import Flask, render_template, request
import re
from preprocess import (
    search_similar_articles,
    fake_news_det,
    get_sentiments,
    count_misspelled_words,
    count_offensive_words
)

app = Flask(__name__)

trusted_news_providers = [
    
]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/result", methods=["POST"])
def predict():
    query = {
        "author": request.form["author"],
        "title": request.form["title"],
        "publish_date": request.form["publish_date"],
        "text": request.form["text"]
    }

    news = query["title"] + " " + query["text"]

    sentiment = get_sentiments(news)
    miss = round(count_misspelled_words(news), 2)
    offensive = count_offensive_words(news)
    pred = fake_news_det(news) #Will show if the news is fake or real using my AI model.

    similarity_df = search_similar_articles(query)

    percentage = 80
    score = 0

    if not similarity_df.empty:
        domain = re.search(r"(https?://)?(www\d?\.)?(?P<name>[\w.-]+)\.\w+", similarity_df["url"][0])
        if domain and domain.group("name") not in trusted_news_providers and similarity_df["similarity"][0] > 0.8:
            score += 10

    if offensive > 0: score += 5
    if miss > 5: score += 5

    percentage += score if pred == 0 else 20 - score
    percentage = str(percentage) + "%"

    return render_template(
        "index.html",
        prediction=pred,
        similarity_table=similarity_df,
        sentiment_intensity=sentiment,
        misspelled_count=str(miss)+"%",
        offensive_count=offensive,
        percentage=percentage
    )

if __name__ == "__main__":
    app.run(debug=True)
