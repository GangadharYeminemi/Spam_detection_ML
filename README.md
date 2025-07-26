 SMS Spam Detection

Overview
This project classifies SMS messages as **Spam** or **Ham** using traditional machine learning. It uses a Naive Bayes classifier trained on a labeled SMS dataset.

Methodology
- Text preprocessing: cleaning, lowercasing, stopword removal
- Feature extraction: TF-IDF vectorizer
- Classification: Multinomial Naive Bayes
- Model persistence using `joblib`

 Tech Stack
- Python, Pandas, Scikit-learn
- TfidfVectorizer, Naive Bayes

Output
- Trained model: `model.pkl`
- Vectorizer: `vectorizer.pkl`


