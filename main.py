import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import joblib
# Function to load vectorizer and model with error handling
def load_vectorizer():
    vectorizer_path = 'tfidf_vectorizer.sav'  # Replace with the actual path to your vectorizer file
    try:
        with open(vectorizer_path, 'rb') as file:
            vectorizer = joblib.load(file)
        st.success("Vectorizer successfully loaded.")
        return vectorizer
    except Exception as e:
        st.error(f"Error loading vectorizer: {e}")
        st.stop()  # Stop the app if the vectorizer cannot be loaded

def load_model():
    model_path = 'sentiment_classifier.sav'  # Replace with the actual path to your model file
    try:
        with open(model_path, 'rb') as file:
            model = joblib.load(file)
        st.success("Model successfully loaded.")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()  # Stop the app if the model cannot be loaded

# Streamlit UI
st.title('Sentiment Analysis App')

# Load vectorizer and model
tfidf_vec = load_vectorizer()
best_model = load_model()

# User input section
st.header("Predict Sentiment")
user_input = st.text_input('Enter your text here:')
if st.button('Predict Sentiment'):
    if not user_input.strip():
        st.warning('Please enter some text.')
    else:
        # Vectorize and predict
        text_vectorized = tfidf_vec.transform([user_input])
        prediction = best_model.predict(text_vectorized)
        st.success(f'Predicted Sentiment Group: **{prediction[0]}**')

# Sample texts section
with st.expander("🧪 Test Sample Texts", expanded=False):
    st.write("Try these pre-defined samples:")
    
    sample_texts = [
        "I absolutely love this product! It works wonders.",
        "This is the worst experience I've ever had.",
    ]

    # Display sample texts
    for text in sample_texts:
        st.code(text)

    # Button to test all samples
    if st.button('Run All Sample Tests'):
        results = []
        for text in sample_texts:
            text_vectorized = tfidf_vec.transform([text])
            prediction = best_model.predict(text_vectorized)
            results.append({
                "text": text,
                "sentiment": prediction[0]
            })
        st.session_state.sample_results = results

    # Display stored results if available
    if 'sample_results' in st.session_state:
        st.divider()
        st.subheader("Test Results")
        for result in st.session_state.sample_results:
            st.markdown(f"""
            Text:  
            `{result['text']}`  
            Sentiment: **{result['sentiment']}**  
            """)