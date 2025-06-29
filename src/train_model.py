# src/train_model.py
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

def train_baseline(X, y):
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X, y)
    return lr

def train_xgb_cv(X, y):
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5],
        "learning_rate": [0.01, 0.1]
    }
    xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    grid = GridSearchCV(xgb_clf, param_grid, cv=5, scoring="roc_auc", n_jobs=-1)
    grid.fit(X, y)
    return grid.best_estimator_

def save_model(model, path="models/model.pkl"):
    joblib.dump(model, path)
