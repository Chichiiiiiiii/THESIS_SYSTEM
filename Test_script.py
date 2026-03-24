import joblib

# Load everything
tfidf = joblib.load('tfidf_vectorizer.pkl')
model_lr = joblib.load('model_lr.pkl')
model_nb = joblib.load('model_nb.pkl')

def test_models(text):
    vec = tfidf.transform([text])
    
    # Get Predictions
    pred_lr = model_lr.predict(vec)[0]
    pred_nb = model_nb.predict(vec)[0]
    
    # Get Probabilities (Confidence)
    prob_lr = model_lr.predict_proba(vec)[0][pred_lr]
    prob_nb = model_nb.predict_proba(vec)[0][pred_nb]
    
    print(f"Text: '{text}'")
    print(f"LR: {'Taglish' if pred_lr==1 else 'English'} ({prob_lr:.2%})")
    print(f"NB: {'Taglish' if pred_nb==1 else 'English'} ({prob_nb:.2%})")
    print("-" * 30)

text_to_test = "Mahal ako ni beshy, joke!"
test_models(text_to_test)

