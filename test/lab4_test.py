"""
Lab 4 Test: Text Classification

This script tests the TextClassifier implementation with the
Twitter Financial News Sentiment dataset.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

from src.models.text_classifier import TextClassifier


def load_data():
    """Load the sentiment dataset."""
    train_df = pd.read_csv('data/lab4/sent_train.csv')
    valid_df = pd.read_csv('data/lab4/sent_valid.csv')
    return train_df, valid_df


def test_small_dataset():
    """Test with a small in-memory dataset."""
    print("=" * 50)
    print("Test 1: Small Dataset")
    print("=" * 50)
    
    # Small dataset
    texts = [
        "This movie is fantastic and I love it!",
        "I hate this film, it's terrible.",
        "The acting was superb, a truly great experience.",
        "What a waste of time, absolutely boring.",
        "Highly recommend this, a masterpiece.",
        "Could not finish watching, so bad."
    ]
    labels = [1, 0, 1, 0, 1, 0]  # 1 for positive, 0 for negative
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.33, random_state=42
    )
    
    # Initialize and train
    vectorizer = TfidfVectorizer()
    model = LogisticRegression(solver='liblinear')
    classifier = TextClassifier(vectorizer, model)
    classifier.fit(X_train, y_train)
    
    # Predict and evaluate
    predictions = classifier.predict(X_test)
    metrics = classifier.evaluate(y_test, predictions)
    
    print(f"Test texts: {X_test}")
    print(f"True labels: {y_test}")
    print(f"Predictions: {list(predictions)}")
    print(f"\nMetrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")


def test_large_dataset():
    """Test with the full sentiment dataset."""
    print("\n" + "=" * 50)
    print("Test 2: Large Dataset (Twitter Financial News)")
    print("=" * 50)
    
    # Load data
    train_df, valid_df = load_data()
    
    X_train = train_df['text'].tolist()
    y_train = train_df['label'].tolist()
    X_test = valid_df['text'].tolist()
    y_test = valid_df['label'].tolist()
    
    print(f"Train size: {len(X_train)}")
    print(f"Test size: {len(X_test)}")
    
    # Test 1: Logistic Regression + TF-IDF (Baseline)
    print("\n--- Baseline: Logistic Regression + TF-IDF ---")
    vectorizer_lr = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')
    model_lr = LogisticRegression(solver='lbfgs', max_iter=1000)
    
    classifier_lr = TextClassifier(vectorizer_lr, model_lr)
    classifier_lr.fit(X_train, y_train)
    
    pred_lr = classifier_lr.predict(X_test)
    metrics_lr = classifier_lr.evaluate(y_test, pred_lr)
    
    print("Metrics:")
    for name, value in metrics_lr.items():
        print(f"  {name}: {value:.4f}")
    
    # Test 2: Naive Bayes + TF-IDF
    print("\n--- Improved: Naive Bayes + TF-IDF ---")
    vectorizer_nb = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')
    model_nb = MultinomialNB(alpha=0.1)
    
    classifier_nb = TextClassifier(vectorizer_nb, model_nb)
    classifier_nb.fit(X_train, y_train)
    
    pred_nb = classifier_nb.predict(X_test)
    metrics_nb = classifier_nb.evaluate(y_test, pred_nb)
    
    print("Metrics:")
    for name, value in metrics_nb.items():
        print(f"  {name}: {value:.4f}")
    
    # Summary
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    print(f"{'Model':<35} {'Accuracy':<12} {'F1-Score':<12}")
    print("-" * 59)
    print(f"{'Logistic Regression + TF-IDF':<35} {metrics_lr['accuracy']:<12.4f} {metrics_lr['f1_score']:<12.4f}")
    print(f"{'Naive Bayes + TF-IDF':<35} {metrics_nb['accuracy']:<12.4f} {metrics_nb['f1_score']:<12.4f}")


if __name__ == "__main__":
    test_small_dataset()
    test_large_dataset()
