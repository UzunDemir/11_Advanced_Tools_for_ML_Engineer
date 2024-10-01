import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
import nltk
from nltk.corpus import stopwords
import re
import mlflow
import mlflow.sklearn
import lightgbm as lgb

# Download Russian stopwords
nltk.download('stopwords')
russian_stopwords = set(stopwords.words('russian'))

# Load the JSONL file
def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)

# Preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in russian_stopwords]
    return ' '.join(words)

# Start MLflow run
with mlflow.start_run():
    # Load the data
    df = load_jsonl('kinopoisk.jsonl')

    # Preprocess the content
    df['processed_content'] = df['content'].apply(preprocess_text)

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_content'], df['grade3'], test_size=0.2, random_state=42
    )

    # Create bag of words representation
    vectorizer = CountVectorizer()
    X_train_bow = vectorizer.fit_transform(X_train)
    X_test_bow = vectorizer.transform(X_test)

    # Train the LightGBM classifier
    clf = lgb.LGBMClassifier()
    clf.fit(X_train_bow, y_train)

    # Make predictions
    y_pred = clf.predict(X_test_bow)

    # Evaluate the model
    report = classification_report(y_test, y_pred, output_dict=True)
    print(classification_report(y_test, y_pred))

    # Log metrics
    mlflow.log_param("model", "LightGBM")
    mlflow.log_param("features", "Bag of Words")
    mlflow.log_param("test_size", 0.2)
    mlflow.log_metric("accuracy", report['accuracy'])
    mlflow.log_metric("precision", report['macro avg']['precision'])
    mlflow.log_metric("recall", report['macro avg']['recall'])
    mlflow.log_metric("f1_score", report['macro avg']['f1-score'])

    # Log the model
    mlflow.sklearn.log_model(clf, "model", input_example=X_train_bow[0].toarray())
