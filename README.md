# Google Play App Review Sentiment Analysis

## Problem
Classify Google Play app reviews as Negative, Neutral or Positive to summarize customer feedback.

## Dataset
- File: `sentiment-analysis-dataset-google-play-app-reviews.csv`
- Source: [Kaggle – Sentiment Analysis Dataset: Google Play App Reviews](https://www.kaggle.com/datasets/farhaouimouhamed/sentiment-analysis-dataset-google-play-app-reviews)
- Review text: `content`
- Star rating: `score`
- Labels: 1–2 = Negative, 3 = Neutral, 4–5 = Positive.

## Algorithms
Logistic Regression, Multinomial Naive Bayes, Complement Naive Bayes, Linear SVM and SGD Classifier.

## Method
1. Load the CSV and keep the `content` and `score` columns.
2. Remove missing or invalid ratings.
3. Convert star ratings into three sentiment classes.
4. Clean review text and remove empty and duplicate reviews.
5. Convert text into TF-IDF features (unigrams and bigrams).
6. Use an 80/20 stratified train-test split.
7. Compare accuracy, balanced accuracy, macro precision, macro recall and macro F1.
8. Save the best model and use it to predict sentiment for new reviews.

## Run
```bash
pip install -r requirements.txt
python main.py
```
Put the CSV inside `data/` before running. At the end, type a review to test it; type `exit` to stop.

## Outputs
Model comparison CSV, charts and the best trained model are saved in `results/`.

## Results reference
The project report recorded Logistic Regression as best by macro F1 (0.6422), with accuracy 0.6531 on its reported test split. Run this repository version to regenerate the metrics.

## Limitation
Sentiment labels are derived from star ratings. A rating may not always match the emotion expressed in the review text.
