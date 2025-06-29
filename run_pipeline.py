# run_pipeline.py
from src.data_prep import load_data, split_data, build_preprocessor
from src.train_model import train_baseline, train_xgb_cv, save_model
from src.evaluate import evaluate_model
from src.shap_analysis import compute_shap, plot_summary

df = load_data("data/Loan_default.csv")


train, test = split_data(df)
preprocessor = build_preprocessor()

X_train = preprocessor.fit_transform(train)
X_test = preprocessor.transform(test)

y_train, y_test = train["Default"], test["Default"]



model = train_xgb_cv(X_train, y_train)
save_model(model)

evaluate_model(model, X_test, y_test)

explainer, shap_values = compute_shap(model, X_train)
plot_summary(shap_values, X_train)
