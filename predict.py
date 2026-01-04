import joblib

# Load saved model and vectorizer
model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# Test new tweets
new_tweets = [
    "I love this game so much",
    "This update ruined everything",
    "The service was okay, nothing special"
]

# Transform and predict
new_tweets_tfidf = vectorizer.transform(new_tweets)
predictions = model.predict(new_tweets_tfidf)

# Display results
for tweet, sentiment in zip(new_tweets, predictions):
    print(f"{tweet} --> {sentiment}")
