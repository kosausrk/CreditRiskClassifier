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

---

## How to Run

1. **Install Dependencies**

   ```bash
   pip install -r requirements.txt


## DEV WORKFLOW 

# 1. Activate venv
source venv/bin/activate

# 2. If retraining is needed
python run_pipeline.py

# 3. Launch UI
streamlit run app/streamlit.py
