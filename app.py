import streamlit as st
import pickle

import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')


def transform_text(text):
    ps = PorterStemmer()

    # Convert text to lowercase
    text = text.lower()

    # Tokenize text using regex (splitting by non-alphanumeric characters)
    tokens = re.findall(r'\b\w+\b', text)

    # Remove stopwords and apply stemming
    filtered_words = [
        ps.stem(word) for word in tokens
        if word not in stopwords.words('english')
    ]

    return " ".join(filtered_words)
tfidf= pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))

st.title('Email Spam Classifier')

input_sms=st.text_input('Enter the Message :')

if st.button('Predict'):


# 1.preprocess
    transformed_sms = transform_text(input_sms)
# 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
# 3. predict
    result = model.predict(vector_input)[0]
# 4. Display
    if result == 1:
        st.header('Spam')
    else:
        st.header('Not Spam')