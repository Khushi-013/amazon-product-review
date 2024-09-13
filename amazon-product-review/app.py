# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

# Define the text cleaning function
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetical characters
    text = text.lower()  # Convert to lowercase
    return text

# Streamlit Title and Sidebar
st.title("Amazon Product Review Sentiment Analysis")
st.sidebar.title("Options")

# Data Upload Section
uploaded_file = st.sidebar.file_uploader("Upload Dataset (CSV)", type="csv")

if uploaded_file:
    # Load the dataset
    df = pd.read_csv(uploaded_file, sep=';', encoding='latin1', on_bad_lines='skip')

    st.write("### Dataset Overview")
    st.write(df)

    # 1. Data Preprocessing
    # Clean the 'text' column
    df['cleaned_text'] = df['text'].apply(lambda x: clean_text(str(x)))
    df.dropna(subset=['cleaned_text'], inplace=True)

    # 2. Sentiment Classification based on 'stars'
    def classify_sentiment(stars):
        if stars >= 4:
            return 'positive'
        elif stars == 3:
            return 'neutral'
        else:
            return 'negative'

    df['sentiment'] = df['stars'].apply(classify_sentiment)

    # Display Data Summary
    st.write("### Data After Preprocessing")
    st.write(df[['cleaned_text', 'sentiment']])

    # Sentiment Distribution Plot
    sentiment_counts = df['sentiment'].value_counts()
    st.write("### Sentiment Distribution")
    st.bar_chart(sentiment_counts)

    # 3. TF-IDF Vectorization
    vectorizer = TfidfVectorizer(max_features=5000)
    X = df['cleaned_text']
    y = df['sentiment']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Model Training and Evaluation
    model = LogisticRegression()
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"### Model Accuracy: {accuracy:.2f}")
    st.text("Classification Report")
    st.text(classification_report(y_test, y_pred))

    # Confusion Matrix
    st.write("### Confusion Matrix")
    conf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Neutral', 'Positive'],
                yticklabels=['Negative', 'Neutral', 'Positive'])
    st.pyplot(plt.gcf())
    plt.clf()

    # Word Clouds for each sentiment
    st.write("### Word Clouds")
    sentiment_to_filter = st.sidebar.radio("Select Sentiment for Word Cloud", ['positive', 'neutral', 'negative'])

    sentiment_reviews = df[df['sentiment'] == sentiment_to_filter]['cleaned_text']
    wordcloud = WordCloud(width=800, height=400, max_words=100, background_color='white').generate(' '.join(sentiment_reviews))
    
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt.gcf())
    plt.clf()

    # Save Model Option
    if st.sidebar.button("Save Model"):
        joblib.dump(model, 'sentiment_model.pkl')
        joblib.dump(vectorizer, 'vectorizer.pkl')
        st.sidebar.write("Model and Vectorizer saved successfully!")

else:
    st.write("Please upload a CSV file to get started.")

# Import necessary libraries
import streamlit as st
import pandas as pd
import re
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Download VADER lexicon
nltk.download('vader_lexicon')

# Initialize the SentimentIntensityAnalyzer for VADER
sia = SentimentIntensityAnalyzer()

# Streamlit Sidebar Title for Review Analysis
st.sidebar.title("Review Analysis")

# Text area to accept user review
user_review = st.sidebar.text_area("Enter your review:", "Type your review here...")

# Review analysis using NLTK's VADER
if user_review:
    # Use VADER to analyze sentiment
    vader_result = sia.polarity_scores(user_review)

    # Display the VADER sentiment analysis result
    st.sidebar.write("**VADER Sentiment Analysis**")
    st.sidebar.write(f"Positive Score: {vader_result['pos']}")
    st.sidebar.write(f"Neutral Score: {vader_result['neu']}")
    st.sidebar.write(f"Negative Score: {vader_result['neg']}")
    st.sidebar.write(f"Compound Score: {vader_result['compound']}")

