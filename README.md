# Sentiment Analysis on Amazon Fine Food Reviews using NLP, Machine Learning & BERT (Transformers)

This project performs automated sentiment classification on Amazon Fine Food Reviews using Natural Language Processing (NLP) techniques with both traditional Machine Learning models and a BERT Transformer deep learning model. Reviews are classified into Positive, Neutral, and Negative sentiments.

---

## Objectives

- Predict the sentiment of user reviews automatically
- Compare traditional ML vs. BERT transformer-based models
- Develop a complete NLP pipeline with training, evaluation, and deployment-ready output

---

## Key Features

- Complete data preprocessing pipeline
- Text cleaning with lemmatization and stopword removal
- Balanced dataset to reduce class bias
- Exploratory Data Analysis (EDA) visualizations
- Multiple Machine Learning models:
  - Naive Bayes
  - Logistic Regression
  - Random Forest
  - Linear SVM
  - XGBoost
  - LightGBM
- BERT fine-tuning for improved contextual understanding
- Ensemble voting approach for enhanced accuracy and stability
- Model saving for reuse and deployment

---

## Dataset

Amazon Fine Food Reviews  
Ratings (1–5) are mapped to sentiment:
- 1–2 → Negative
- 3 → Neutral
- 4–5 → Positive

Primary columns used:
- **Text** — review content
- **Score** — star rating for sentiment mapping

Dataset should be placed in:
  data/Reviews.csv

---

## Tech Stack

| Component | Tools |
|----------|------|
| Programming | Python 3.x |
| NLP | NLTK, Emoji, WordCloud |
| ML Models | Scikit-learn, XGBoost, LightGBM |
| Transformers | PyTorch, HuggingFace Transformers |
| Visualization | Matplotlib, Seaborn |
| Others | pandas, numpy, tqdm, pickle, json, logging |

---
