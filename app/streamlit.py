# app/streamlit.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import joblib, shap

from src.data_prep import build_preprocessor

# === Setup ===
st.title("Credit Risk Classifier")

# Load model
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'model.pkl'))
if not os.path.exists(model_path):
    st.error(f"❌ Model file not found at: {model_path}")
    st.stop()
model = joblib.load(model_path)

# Load saved preprocessor (used during training)
preproc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'preprocessor.pkl'))
if not os.path.exists(preproc_path):
    st.error(f"❌ Preprocessor not found at: {preproc_path}")
    st.stop()
preprocessor = joblib.load(preproc_path)

# Load SHAP explainer
explainer = shap.TreeExplainer(model)

# === Manual Input Section ===
st.header("🔧 Manual Input for One Applicant")

input_data = {
    "Age": st.number_input("Age"),
    "Income": st.number_input("Income"),
    "LoanAmount": st.number_input("Loan Amount"),
    "CreditScore": st.number_input("Credit Score"),
    "MonthsEmployed": st.number_input("Months Employed"),
    "NumCreditLines": st.number_input("Number of Credit Lines"),
    "InterestRate": st.number_input("Interest Rate"),
    "LoanTerm": st.number_input("Loan Term (months)"),
    "DTIRatio": st.number_input("Debt-To-Income Ratio"),
    "Education": st.selectbox("Education", ["High School", "Bachelor's", "Master's", "PhD"]),
    "EmploymentType": st.selectbox("Employment Type", ["Full-time", "Part-time", "Self-employed", "Unemployed"]),
    "MaritalStatus": st.selectbox("Marital Status", ["Single", "Married", "Divorced"]),
    "HasMortgage": st.selectbox("Has Mortgage", ["Yes", "No"]),
    "HasDependents": st.selectbox("Has Dependents", ["Yes", "No"]),
    "LoanPurpose": st.selectbox("Loan Purpose", ["Auto", "Business", "Education", "Home", "Other"]),
    "HasCoSigner": st.selectbox("Has Co-Signer", ["Yes", "No"])
}

df = pd.DataFrame([input_data])
df_transformed = preprocessor.transform(df)

pred = model.predict_proba(df_transformed)[0, 1]
st.write(f"🔍 Predicted default probability: **{pred:.2%}**")

if st.checkbox("Show SHAP explanation for this input"):
    shap_values = explainer.shap_values(df_transformed)
    st_shap = st.pyplot()
    shap.force_plot(explainer.expected_value, shap_values[0, :], df.iloc[0, :], matplotlib=True, show=False)
    st_shap.pyplot()

# === CSV Upload Section ===
st.header("📁 Upload CSV to Get Batch Predictions")

uploaded_file = st.file_uploader("Upload a CSV file (same format as Loan_default.csv)", type="csv")

if uploaded_file:
    input_df = pd.read_csv(uploaded_file)
    input_df.columns = input_df.columns.str.strip()

    # Drop target column if included
    if "Default" in input_df.columns:
        input_df = input_df.drop(columns=["Default"])

    input_transformed = preprocessor.transform(input_df)
    pred_probs = model.predict_proba(input_transformed)[:, 1]
    input_df["Predicted_Default_Probability"] = pred_probs

    st.write("📊 Predictions:")
    st.dataframe(input_df)

    high_risk = input_df[input_df["Predicted_Default_Probability"] > 0.5]
    if not high_risk.empty:
        st.warning(f"🚨 {len(high_risk)} high-risk applicants (prob > 50%) detected")

    if st.checkbox("Show SHAP Summary Plot"):
        shap_values = explainer.shap_values(input_transformed)
        shap.summary_plot(shap_values, input_df, show=False)  # ✅ Pass original dataframe for labels

        st.pyplot(bbox_inches='tight')
