# Sentiment Analysis with TF-IDF and Logistic Regression

## Overview
This project performs binary sentiment classification (positive/negative) on movie review texts using TF-IDF vectorization and classical machine learning models.

## Dataset
- 2000 samples (1600 train / 400 test) — adjust according to your real numbers
- Balanced classes: pos / neg

## Approach
1. Text preprocessing and TF-IDF vectorization (`max_features=5000`, `min_df=5`, `max_df=0.8`, `ngram_range=(1,2)`)
2. Model training and comparison:
   - Logistic Regression
   - Multinomial Naive Bayes
3. Hyperparameter tuning (regularization strength `C`) to address overfitting
4. Model evaluation using accuracy, classification report, confusion matrix, and ROC-AUC
5. Error analysis on misclassified samples

Logistic Regression (C=0.7) was selected as the final model due to higher test accuracy and a smaller train-test gap, indicating better generalization given the limited dataset size.

## Cross-Validation
5-fold CV accuracy: 0.856 ± 0.027
