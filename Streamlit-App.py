import streamlit as st
import joblib

# Function to preprocess and predict the sentiment of a review
def preprocess_and_predict(review, tfidf_vectorizer, model, label_encoder):
    # Preprocess the review
    cleaned_review = review.lower().replace('[^\w\s]', '')
    transformed_review = tfidf_vectorizer.transform([cleaned_review])

    # Predict the sentiment
    sentiment_prediction = model.predict(transformed_review)
    
    # Decode the prediction
    decoded_sentiment = label_encoder.inverse_transform(sentiment_prediction)[0]
    return decoded_sentiment

# Load the trained model and components at the start of your Streamlit script
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
lr_model = joblib.load('logistic_regression_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Streamlit app layout
st.title("Spotify Review Sentiment Analysis")

# User input
user_review = st.text_area("Enter your review here:")

if st.button('Analyze Sentiment'):
    if user_review:
        # Call the preprocess and predict function
        sentiment = preprocess_and_predict(user_review, tfidf_vectorizer, lr_model, label_encoder)
        st.write(f"The predicted sentiment is: **{sentiment}**")
    else:
        st.write("Please enter a review to analyze.")

