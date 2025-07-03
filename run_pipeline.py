from src.data_prep import load_data, split_data, build_preprocessor
from src.train_model import train_baseline, train_xgb_cv, save_model
from src.evaluate import evaluate_model
from src.shap_analysis import compute_shap, plot_summary

import joblib
import pandas as pd

# Load data
df = load_data("data/Loan_default.csv")

# Split data
train, test = split_data(df)

# Separate target
y_train = train["Default"]
y_test = test["Default"]

# Drop target column to get raw features
X_train_raw = train.drop(columns=["Default"])
X_test_raw = test.drop(columns=["Default"])

# Build and fit preprocessor
preprocessor = build_preprocessor()
X_train = preprocessor.fit_transform(X_train_raw)
X_test = preprocessor.transform(X_test_raw)

# Save fitted preprocessor
joblib.dump(preprocessor, "models/preprocessor.pkl")

# Train model
model = train_xgb_cv(X_train, y_train)
save_model(model)

# Evaluate model
evaluate_model(model, X_test, y_test)

# SHAP: use raw input for meaningful labels
# SHAP
explainer, shap_values = compute_shap(model, X_train)

# FIXED: use the same transformed matrix for both SHAP values and plot
# but now wrap it in a DataFrame with proper feature names from the preprocessor
import pandas as pd

# Extract column names after transformation (if your preprocessor has .get_feature_names_out)
try:
    feature_names = preprocessor.get_feature_names_out()
except AttributeError:
    feature_names = [f"Feature {i}" for i in range(X_train.shape[1])]

X_train_transformed_df = pd.DataFrame(X_train, columns=feature_names)

# Now plot with proper labels
plot_summary(shap_values, X_train_transformed_df)
