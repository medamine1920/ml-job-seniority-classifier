ML Job Seniority Classification using NLP

OVERVIEW
This project builds a machine learning system that predicts the seniority level
(entry or mid) of machine learning job postings based solely on their job descriptions.

The goal is to demonstrate how unstructured text data can be transformed into
meaningful signals using Natural Language Processing (NLP) and classical
machine learning models.

This project is a continuation of a broader job market analysis and focuses
specifically on predictive modeling and evaluation.

------------------------------------------------------------

DATASET
Source: Kaggle – Machine Learning job postings (U.S.-based)

Input:
- job_description_clean (preprocessed text)

Target:
- seniority_norm (entry, mid)

Notes:
- Internship, leadership, and unknown roles were excluded
- The task focuses on distinguishing entry vs mid roles using language patterns

------------------------------------------------------------

PROBLEM STATEMENT
Can we automatically infer job seniority from a job description using text-based features?

This problem is relevant for:
- Job platforms (role classification)
- Recruiters (posting normalization)
- Career intelligence tools (role matching)

------------------------------------------------------------

METHODOLOGY

1) Text Vectorization
- TF-IDF representation
- Unigrams and bigrams
- English stopword removal
- Sparse high-dimensional feature space

2) Models Evaluated
- Logistic Regression (baseline)
- Logistic Regression with class weighting
- Linear Support Vector Machine (LinearSVC)

3) Evaluation Strategy
- Stratified train/test split
- Metrics: precision, recall, F1-score
- Macro F1-score used due to slight class imbalance

------------------------------------------------------------

MODEL SELECTION & RESULTS
- Linear SVM achieved the best Macro F1-score
- Entry-level recall improved significantly compared to baseline models
- The model balances performance across both classes instead of favoring the majority

Why Macro F1-score:
- Treats entry and mid classes equally
- Penalizes models that ignore minority classes
- Standard metric for imbalanced text classification problems

------------------------------------------------------------

ERROR ANALYSIS
Most misclassifications occurred on:
- Borderline job descriptions
- Roles with mixed responsibilities
- Vague seniority language

These errors are expected and indicate reasonable model behavior rather than overfitting.

------------------------------------------------------------

MODEL ARTIFACTS
The trained artifacts are saved for reuse:
- TF-IDF vectorizer: models/tfidf.joblib
- Seniority classifier: models/seniority_model.joblib

A small inference script is provided to demonstrate real-world usage.

------------------------------------------------------------

PROJECT STRUCTURE

ml-job-seniority-classifier/
├── notebook/
│   └── ml_job_seniority_classifier.ipynb
├── models/
│   ├── tfidf.joblib
│   └── seniority_model.joblib
├── outputs/
│   └── charts/
│       └── confusion_matrix.png
├── predict.py
├── README.txt
├── requirements.txt
└── .gitignore

------------------------------------------------------------

HOW TO RUN

1) Install dependencies:
pip install -r requirements.txt

2) Open and run the notebook:
jupyter notebook

3) Run a quick inference demo:
python predict.py

------------------------------------------------------------

KEY TAKEAWAYS
- Accuracy alone is misleading for imbalanced text problems
- Macro F1-score provides a fairer evaluation
- Linear models perform strongly on TF-IDF features for NLP tasks
- Model explainability and error analysis are as important as raw performance

------------------------------------------------------------

AUTHOR
Med Amine Gnichi
Portfolio Project
