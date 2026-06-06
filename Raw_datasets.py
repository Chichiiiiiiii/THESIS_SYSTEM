from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd


# Load cleaned dataset

df_combined = pd.read_csv("Datasets/cleaned_data.csv")
df = df_combined[['final_text', 'label']].dropna()

X = df['final_text']
y = df['label']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Vectorize
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Predict
pred = model.predict(X_test_vec)

# Overall accuracy
print("Overall Accuracy:", accuracy_score(y_test, pred))

# -----------------------------
# PER-LANGUAGE ACCURACY
# -----------------------------

y_test = np.array(y_test)
pred = np.array(pred)

english_mask = (y_test == 0)
taglish_mask = (y_test == 1)

english_acc = accuracy_score(y_test[english_mask], pred[english_mask])
taglish_acc = accuracy_score(y_test[taglish_mask], pred[taglish_mask])

print("English Accuracy:", english_acc)
print("Taglish Accuracy:", taglish_acc)