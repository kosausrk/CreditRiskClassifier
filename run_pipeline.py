# run_pipeline.py
from src.data_prep import load_data, split_data, build_preprocessor
from src.train_model import train_baseline, train_xgb_cv, save_model
from src.evaluate import evaluate_model
from src.shap_analysis import compute_shap, plot_summary

import joblib

# Load data
df = load_data("data/Loan_default.csv")

# Split
train, test = split_data(df)

# Build and fit preprocessor
preprocessor = build_preprocessor()
X_train = preprocessor.fit_transform(train)
X_test = preprocessor.transform(test)

# Save fitted preprocessor
joblib.dump(preprocessor, "models/preprocessor.pkl")

# Target
y_train, y_test = train["Default"], test["Default"]

# Train model
model = train_xgb_cv(X_train, y_train)
save_model(model)

# Evaluate
evaluate_model(model, X_test, y_test)

# SHAP
explainer, shap_values = compute_shap(model, X_train)
plot_summary(shap_values, X_train)
