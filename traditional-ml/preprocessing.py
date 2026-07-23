import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, default="/opt/ml/processing/input/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    parser.add_argument("--output-dir", type=str, default="/opt/ml/processing/output")
    args = parser.parse_args()

    df = pd.read_csv(args.input_data)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    df = df.drop(columns=["customerID"])
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)

    y = (df["Churn"] == "Yes").astype(int)
    X = df.drop(columns=["Churn"])

    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    train_out = pd.DataFrame(X_train_t.toarray() if hasattr(X_train_t, "toarray") else X_train_t)
    train_out.insert(0, "target", y_train.values)

    test_out = pd.DataFrame(X_test_t.toarray() if hasattr(X_test_t, "toarray") else X_test_t)
    test_out.insert(0, "target", y_test.values)

    os.makedirs(f"{args.output_dir}/train", exist_ok=True)
    os.makedirs(f"{args.output_dir}/test", exist_ok=True)
    train_out.to_csv(f"{args.output_dir}/train/train.csv", index=False, header=False)
    test_out.to_csv(f"{args.output_dir}/test/test.csv", index=False, header=False)

if __name__ == "__main__":
    main()