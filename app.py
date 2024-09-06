import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Load the model and CountVectorizer
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('cv.pkl', 'rb') as cv_file:
    cv = pickle.load(cv_file)

# Text preprocessing function
def preprocess_text(text):
    text = re.sub(pattern='[^a-zA-Z]', repl=' ', string=text)
    text = text.lower()
    text = text.split()
    ps = PorterStemmer()
    text = [ps.stem(word) for word in text if word not in stopwords.words('english')]
    return ' '.join(text)

# Streamlit application
st.title('Sentiment Analysis App')

# Text input from the user
comment = st.text_area("Enter your comment:")

if st.button('Predict'):
    if comment:
        # Preprocess the comment
        processed_comment = preprocess_text(comment)
        
        # Transform the comment to the vector space
        vectorized_comment = cv.transform([processed_comment]).toarray()
        
        # Predict the sentiment
        prediction = model.predict(vectorized_comment)[0]
        
        # Convert prediction to readable form
        sentiment = 'Positive' if prediction == 1 else 'Negative'
        
        st.write(f'The sentiment of the comment is: {sentiment}')
    else:
        st.write("Please enter a comment to analyze.")
