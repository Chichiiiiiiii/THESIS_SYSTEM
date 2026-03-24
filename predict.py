import joblib

# Load the saved assets
model = joblib.load('logistic_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

# New sentence to test
new_text = ["Gusto ko mag-aral ng Python today"]
new_vec = tfidf.transform(new_text) # Must use the LOADED tfidf
prediction = model.predict(new_vec)

print("Taglish" if prediction[0] == 1 else "English")