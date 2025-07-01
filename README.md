# **[IN PROGRESS] **

# Credit Risk Classifier

A machine learning application that predicts the probability of loan default based on customer financial data. Designed for financial institutions like JPMorgan and Capital One to automate risk underwriting with interpretable insights.

---

## Features

- **Exploratory Data Analysis** (EDA) – Understand correlations, outliers, and data health
- **Modeling** – Logistic Regression, Random Forest, and XGBoost with Grid Search
- **Explainability** – SHAP visualizations to satisfy regulatory transparency
- **Streamlit App** – Real-time prediction interface with interpretability toggle

---

## Tech Stack

- Python · Pandas · Scikit-learn · XGBoost  
- Streamlit (UI) · SHAP (model explanations)  
- Matplotlib & Seaborn (visualization)

## Text UML/ Pipeline
```
1. Data Understanding
   ├─ Gather example datasets (e.g., LendingClub, Kaggle credit datasets)
   ├─ Explore feature types: income, credit score, loan amount, etc.
   └─ Identify target variable (loan default = 0/1)

2. Exploratory Data Analysis (EDA)
   ├─ Correlation analysis, outlier detection
   ├─ Missing value imputation
   └─ Visualizations: boxplots, heatmaps, histograms

3. Data Preprocessing
   ├─ Encoding categorical variables
   ├─ Normalization/Standardization
   └─ Train-test split (stratified)

4. Model Development
   ├─ Baseline: Logistic Regression
   ├─ Advanced: Random Forest, XGBoost
   ├─ Cross-validation (e.g., StratifiedKFold)
   └─ Hyperparameter tuning (GridSearchCV / Optuna)

5. Model Evaluation
   ├─ Metrics: ROC AUC, F1, Precision-Recall
   └─ Confusion matrix visualizations

6. Interpretability
   ├─ Feature importance (XGBoost built-in)
   └─ SHAP plots (force, beeswarm, summary)

7. Streamlit App
   ├─ Input form for user financial data
   ├─ Risk prediction output
   └─ Display SHAP explanations

8. Deployment (Optional)
   └─ Streamlit Cloud / Dockerize for local hosting
```

## CreditRiskClassifier 
```
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_model_dev.ipynb
│
├── src/
│   ├── data_prep.py
│   ├── train_model.py
│   ├── evaluate.py
│   └── shap_analysis.py
│
├── app/
│   └── streamlit_app.py
│
├── models/
│   └── xgb_credit_model.pkl
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## How to Run

1. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
