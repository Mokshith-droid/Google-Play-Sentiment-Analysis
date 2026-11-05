from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    classification_report, ConfusionMatrixDisplay
)

# Load dataset
file_path = Path(__file__).resolve().parent / "data" / "sentiment-analysis-dataset-google-play-app-reviews.csv"
if not file_path.exists():
    raise FileNotFoundError("Place the Google Play reviews CSV in the data folder.")
df = pd.read_csv(file_path)

print("First 5 rows:")
print(df.head())

print("\nDataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

# Check required columns
if "content" not in df.columns or "score" not in df.columns:
    raise ValueError(
        "Dataset must contain 'content' and 'score' columns."
    )

# Keep required columns and remove missing values
df = df[["content", "score"]].dropna().copy()

df["score"] = pd.to_numeric(df["score"], errors="coerce")
df = df.dropna(subset=["score"])

# Keep valid ratings from 1 to 5
df = df[df["score"].between(1, 5)].copy()

# Convert ratings to sentiment labels
def get_sentiment(score):
    if score <= 2:
        return "Negative"
    elif score == 3:
        return "Neutral"
    else:
        return "Positive"

df["sentiment"] = df["score"].apply(get_sentiment)

# Clean review text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df["clean_content"] = df["content"].apply(clean_text)

# Remove empty reviews and duplicate review text
df = df[df["clean_content"] != ""].copy()
df = df.drop_duplicates(subset=["clean_content"])

print("\nSentiment distribution:")
print(df["sentiment"].value_counts())

# Input and target
X = df["clean_content"]
y = df["sentiment"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Define machine learning models
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, random_state=42
    ),
    "Multinomial Naive Bayes": MultinomialNB(),
    "Complement Naive Bayes": ComplementNB(),
    "Linear SVM": LinearSVC(random_state=42),
    "SGD Classifier": SGDClassifier(random_state=42)
}

results = []
trained_models = {}

# Train and evaluate each model
for name, model in models.items():
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words=None
        )),
        ("classifier", model)
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Balanced Accuracy": balanced_accuracy_score(
            y_test, predictions
        ),
        "Precision": precision_score(
            y_test, predictions,
            average="macro", zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions,
            average="macro", zero_division=0
        ),
        "Macro F1": f1_score(
            y_test, predictions,
            average="macro", zero_division=0
        )
    })

    trained_models[name] = pipeline

# Compare algorithms
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by="Macro F1", ascending=False
)

print("\nMODEL COMPARISON:")
print(results_df.to_string(index=False))
results_df.to_csv(Path(__file__).resolve().parent / "results" / "model_comparison.csv", index=False)

# Select best model based on Macro F1
best_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_name]
best_predictions = best_model.predict(X_test)

print("\nBest Model:", best_name)
print("\nClassification Report:")
print(
    classification_report(
        y_test, best_predictions, zero_division=0
    )
)

# Plot comparison graph
results_df.set_index("Model")[
    ["Accuracy", "Precision", "Recall", "Macro F1"]
].plot(kind="bar", figsize=(12, 6))

plt.title("Sentiment Analysis - Algorithm Comparison")
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig(Path(__file__).resolve().parent / "results" / "model_comparison.png", dpi=200, bbox_inches="tight")
plt.savefig(Path(__file__).resolve().parent / "results" / "confusion_matrix.png", dpi=200, bbox_inches="tight")
plt.show()

# Plot confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_test,
    best_predictions,
    labels=["Negative", "Neutral", "Positive"],
    xticks_rotation="vertical"
)

plt.title("Sentiment Analysis - Confusion Matrix")
plt.tight_layout()
plt.show()

# Save the best model
joblib.dump(best_model, Path(__file__).resolve().parent / "results" / "best_sentiment_model.pkl")
print("\nBest model saved in results/best_sentiment_model.pkl")

# Predict sentiment for new reviews
print("\nEnter a review to predict sentiment.")
print("Type 'exit' to stop.")

while True:
    review = input("\nEnter review: ")

    if review.strip().lower() == "exit":
        break

    if not review.strip():
        print("Please enter a review.")
        continue

    prediction = best_model.predict([clean_text(review)])
    print("Predicted Sentiment:", prediction[0])