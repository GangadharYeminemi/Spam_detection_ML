import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import re
import joblib
import os

# Hardcoded stopword list
STOPWORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he',
    'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'were', 'will',
    'with', 'you', 'your', 'this', 'have', 'i', 'me', 'my', 'we', 'our', 'they', 'their'
}

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    tokens = [token for token in tokens if token not in STOPWORDS]
    return ' '.join(tokens)

# Load and preprocess dataset
def load_data(file_path):
    df = pd.read_csv(file_path, encoding='latin-1')
    df = df[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'text'})
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    df['processed_text'] = df['text'].apply(preprocess_text)
    return df

# Train and save model
def train_model(file_path, model_path='model.pkl', vectorizer_path='vectorizer.pkl'):
    try:
        if not os.path.exists(file_path):
            print("Error: Dataset file not found.")
            return False
        
        # Load and preprocess data
        df = load_data(file_path)
        
        # Extract features and labels
        X = df['processed_text']
        y = df['label']
        
        # Convert text to TF-IDF features
        vectorizer = TfidfVectorizer(max_features=5000)
        X_tfidf = vectorizer.fit_transform(X)
        
        # Train Naive Bayes classifier
        model = MultinomialNB()
        model.fit(X_tfidf, y)
        
        # Save model and vectorizer
        joblib.dump(model, model_path)
        joblib.dump(vectorizer, vectorizer_path)
        print(f"Model saved to {model_path}")
        print(f"Vectorizer saved to {vectorizer_path}")
        return True
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        return False

if __name__ == "__main__":
    dataset_path = 'spam.csv'
    train_model(dataset_path)