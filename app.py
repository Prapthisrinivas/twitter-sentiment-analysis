import streamlit as st
import joblib

# Load model & vectorizer
model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

st.set_page_config(page_title="Twitter Sentiment Analyzer", layout="centered")

st.title("🐦 Twitter Sentiment Analyzer")
st.write("Enter a tweet and predict its sentiment")

# Text input
user_input = st.text_area("Enter tweet text here:")

# Predict button
if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text")
    else:
        text_vector = vectorizer.transform([user_input])
        prediction = model.predict(text_vector)[0]

        if prediction == "Positive":
            st.success(f"😊 Sentiment: {prediction}")
        elif prediction == "Negative":
            st.error(f"😡 Sentiment: {prediction}")
        else:
            st.info(f"😐 Sentiment: {prediction}")
