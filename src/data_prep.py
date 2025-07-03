import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

NUM_COLS = [
    "Age", "Income", "LoanAmount", "CreditScore", "MonthsEmployed",
    "NumCreditLines", "InterestRate", "LoanTerm", "DTIRatio"
]

CAT_COLS = [
    "Education", "EmploymentType", "MaritalStatus",
    "HasMortgage", "HasDependents", "LoanPurpose", "HasCoSigner"
]

def load_data(path):
    df = pd.read_csv(path, delimiter=",", on_bad_lines='skip', encoding="utf-8")
    df.columns = df.columns.str.strip()  # Remove extra spaces from headers
    print("🧪 Columns in CSV:", df.columns.tolist())  # Debug print
    return df



def split_data(df, target="Default", test_size=0.2, random_state=42):
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    for train_idx, test_idx in splitter.split(df, df[target]):
        return df.iloc[train_idx], df.iloc[test_idx]

def build_preprocessor():
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="MISSING")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, NUM_COLS),
        ("cat", cat_pipeline, CAT_COLS)
    ])

    return preprocessor
