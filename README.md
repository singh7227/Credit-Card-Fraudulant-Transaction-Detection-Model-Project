# Credit-Card-Fraudulant-Transaction-Detection-Model-Project

This project is an end-to-end machine learning solution built to detect fraudulent credit card transactions.  
The goal was to understand how fraud detection systems work in real-world financial datasets and to build a model that performs well on highly imbalanced data.

---

## Project Summary

Credit card fraud datasets are extremely imbalanced — fraudulent transactions are very rare compared to legitimate ones. Because of this, accuracy alone is not a good evaluation metric.

In this project, I focused on:

- Proper data preprocessing
- Handling class imbalance
- Building classification models
- Evaluating performance using precision, recall, and F1-score
- Deploying the trained model using Flask

---

## Tech Stack

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  
- Seaborn  
- Flask  
- Git & GitHub  

---

## Dataset Information

The dataset contains anonymized credit card transactions with the following features:

- `Time`
- `Amount`
- `V1 – V28` (PCA transformed features)
- `Class` (0 = Legitimate, 1 = Fraud)

The dataset is highly imbalanced, with fraudulent transactions making up less than 1% of the total data.

---

## Workflow

1. Data Cleaning and Exploration  
2. Checking Class Distribution  
3. Feature Scaling
4. Train-Test Split  
5. Model Training  
   - Logistic Regression  
   - Random Forest Classifier  
6. Model Evaluation using:
   - Precision
   - Recall
   - F1 Score
   - Confusion Matrix  
7. Flask App Integration for prediction  

---

## Model Performance 

- Accuracy: 99.2%
- Precision (Fraud Class): 0.91
- Recall (Fraud Class): 0.84
- F1 Score: 0.87

Since the dataset is imbalanced, recall and F1-score were given more importance than accuracy.

---
