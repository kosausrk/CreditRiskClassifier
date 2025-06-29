# src/shap_analysis.py
import shap
import matplotlib.pyplot as plt

def compute_shap(model, X):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return explainer, shap_values

def plot_summary(shap_values, X):
    shap.summary_plot(shap_values, X)



