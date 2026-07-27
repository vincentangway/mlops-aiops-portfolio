import json
import tarfile
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score

if __name__ == "__main__":
    with tarfile.open("/opt/ml/processing/model/model.tar.gz") as tar:
        tar.extractall(path=".")

    booster = xgb.Booster()
    booster.load_model("xgboost-model")

    test_df = pd.read_csv("/opt/ml/processing/test/test.csv", header=None)
    y_test, X_test = test_df.iloc[:, 0], test_df.iloc[:, 1:]
    dtest = xgb.DMatrix(X_test)

    preds = (booster.predict(dtest) > 0.5).astype(int)
    f1 = f1_score(y_test, preds)

    report = {"classification_metrics": {"f1": {"value": f1}}}
    with open("/opt/ml/processing/evaluation/evaluation.json", "w") as f:
        json.dump(report, f)
