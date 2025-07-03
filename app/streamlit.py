# app/streamlit_app.py
import streamlit as st
import pandas as pd
import joblib, shap

import os

from src.data_prep import build_preprocessor #dataprep




# Dynamically resolve model path relative to this script
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'model.pkl'))

if not os.path.exists(model_path):
    st.error(f" Model file not found at: {model_path}")
    st.stop()

model = joblib.load(model_path)



#default 
st.header("📁 Upload CSV to Get Predictions")

uploaded_file = st.file_uploader("Upload a CSV file (same format as Loan_default.csv)", type="csv")

if uploaded_file:
    # Read uploaded data
    input_df = pd.read_csv(uploaded_file)
    input_df.columns = input_df.columns.str.strip()  # Clean up column names

    # Drop any target column if included
    if "Default" in input_df.columns:
        input_df = input_df.drop(columns=["Default"])
    
    # Run model prediction
    pred_probs = model.predict_proba(input_df)[:, 1]
    input_df["Predicted_Default_Probability"] = pred_probs

    # Show table with predictions
    st.write("📊 Predictions:")
    st.dataframe(input_df)

    # Optional: Highlight high-risk loans
    high_risk = input_df[input_df["Predicted_Default_Probability"] > 0.5]
    if not high_risk.empty:
        st.warning(f"🚨 {len(high_risk)} high-risk applicants (prob > 50%) detected")

    # SHAP explanation
    if st.checkbox("Show SHAP Summary Plot"):
        shap_values = explainer.shap_values(input_df)
        shap.summary_plot(shap_values, input_df, show=False)
        st.pyplot(bbox_inches="tight")


explainer = shap.TreeExplainer(model)

st.title("Credit Risk Classifier")
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
pred = model.predict_proba(df)[0,1]
st.write(f"🔍 Predicted default probability: **{pred:.2%}**")

if st.checkbox("Show SHAP explanation"):
    shap_values = explainer.shap_values(df)
    st_shap = st.pyplot()
    shap.force_plot(explainer.expected_value, shap_values[0,:], df.iloc[0,:], matplotlib=True, show=False)
    st_shap.pyplot()
