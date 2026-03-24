import pandas as pd
import joblib
from sklearn import ensemble
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import VotingClassifier

#This is for Ensemble

# 1. Load Data
df = pd.read_csv("Datasets/cleaned_data.csv")
# CRITICAL: Fill any NaN values. TF-IDF will crash if it hits a 'null' value.
# This happens if a row became empty after cleaning (e.g., it was only emojis)
X_text = df['final_text'].fillna("") 
y = df['label']

# 2. THE TF-IDF STEP / Vectorize
# We use 'fit_transform' here because this is the FIRST time 
# the model sees this specific combined vocabulary.
tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
X = tfidf.fit_transform(X_text)

# 3. Split for validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
lr_acc = accuracy_score(y_test, lr_model.predict(X_test))

# 5. Train Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
nb_acc = accuracy_score(y_test, nb_model.predict(X_test))

# 6. CREATE AND TRAIN THE ENSEMBLE
# We "fit" the ensemble using the numeric matrix (X) and the labels (y)
ensemble = VotingClassifier(
    estimators=[('lr', lr_model), ('nb', nb_model)],
    voting='soft'
)

print("Training the ensemble model...")
ensemble.fit(X_train, y_train)

# 6. EVALUATE
y_pred = ensemble.predict(X_test)
print("\nEnsemble Performance:")
print(classification_report(y_test, y_pred))

# 7. EXPORT EVERYTHING
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
joblib.dump(lr_model, 'model_lr.pkl')
joblib.dump(nb_model, 'model_nb.pkl')
joblib.dump(ensemble, 'ensemble_model.pkl')
