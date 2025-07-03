# app/streamlit_app.py
import streamlit as st
import pandas as pd
import joblib, shap

model = joblib.load("../models/model.pkl")


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
