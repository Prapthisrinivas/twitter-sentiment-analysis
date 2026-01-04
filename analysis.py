import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

# Load dataset
df = pd.read_csv("data/tweets.csv", header=None)
df.columns = ["tweet_id", "topic", "sentiment", "text"]

# Clean text function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Apply cleaning
df["clean_text"] = df["text"].astype(str).apply(clean_text)

# Remove stopwords
stop_words = set(stopwords.words("english"))
df["clean_text"] = df["clean_text"].apply(
    lambda x: " ".join(word for word in x.split() if word not in stop_words)
)

# Show output
print(df[["text", "clean_text"]].head())
print(df.shape)

import matplotlib.pyplot as plt

# Sentiment count
sentiment_counts = df["sentiment"].value_counts()
print("\nSentiment Distribution:")
print(sentiment_counts)

# Plot
sentiment_counts.plot(kind="bar")
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Tweets")
plt.show()

# Topic vs Sentiment
topic_sentiment = df.groupby(["topic", "sentiment"]).size().unstack().fillna(0)
print(topic_sentiment.head())

from collections import Counter

all_words = " ".join(df["clean_text"]).split()
word_freq = Counter(all_words)

print("\nTop 20 Most Common Words:")
print(word_freq.most_common(20))


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# Features and labels
X = df["clean_text"]
y = df["sentiment"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Convert text to TF-IDF features
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Predictions
y_pred = model.predict(X_test_tfidf)

# Evaluation
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# Test custom sentence
sample_text = ["I love this product", "This is the worst experience ever"]
sample_tfidf = vectorizer.transform(sample_text)
predictions = model.predict(sample_tfidf)

for text, pred in zip(sample_text, predictions):
    print(f"{text} --> {pred}")


import joblib
import os

# Create models directory
os.makedirs("models", exist_ok=True)

# Save model and vectorizer
joblib.dump(model, "models/sentiment_model.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

print("\nModel and vectorizer saved successfully!")

