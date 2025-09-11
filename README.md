# Telco-Customer-Churn-Prediction
📊 Telco Customer Churn Dataset — cleaned CSV version for machine learning projects, analysis, and end-to-end ML pipelines in Google Colab and Python.
## 📌 Project Overview

This project builds an **end-to-end ML pipeline** for predicting customer churn using the Telco Customer Churn dataset. The pipeline handles preprocessing, model training, hyperparameter tuning, and evaluation, making it reusable and production-ready.

## 🗂️ Dataset

* **Source:** [Telco Customer Churn dataset (IBM Sample)](https://www.kaggle.com/blastchar/telco-customer-churn)
* **Target:** `Churn` column (Yes/No)
* **Features:** customer demographics, services subscribed, billing information

## ⚙️ Methodology

1. **Data Preprocessing**

   * Dropped irrelevant ID columns
   * One-hot encoding for categorical features
   * Standard scaling for numeric features

2. **Modeling**

   * Built a Scikit-learn `Pipeline` with preprocessing + model
   * Trained Logistic Regression and Random Forest models
   * Used `GridSearchCV` for hyperparameter tuning

3. **Evaluation**

   * Metrics: F1-score, Accuracy, ROC AUC
   * Confusion Matrix visualization

4. **Export**

   * Saved trained pipeline with `joblib`
   * Saved results report in `results.txt`

## 📊 Results

* Best model selected via GridSearchCV
* Evaluation metrics: F1-score, Accuracy, ROC AUC (reported in notebook and `results.txt`)

## 🚀 How to Run

```bash
pip install pandas scikit-learn matplotlib joblib
```

Run the Jupyter/Colab notebook → upload dataset → execute all cells.

## 🧰 Skills Learned

* Building production-ready ML pipelines
* Feature engineering (encoding + scaling)
* Model training and hyperparameter tuning
* Exporting reusable ML models
* Interpreting evaluation metrics (F1, ROC AUC)
