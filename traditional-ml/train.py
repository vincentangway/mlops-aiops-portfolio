import argparse
import os
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--eta", type=float, default=0.2)
    parser.add_argument("--num_round", type=int, default=100)
    parser.add_argument("--objective", type=str, default="binary:logistic")

    parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--validation", type=str, default=os.environ["SM_CHANNEL_VALIDATION"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    args = parser.parse_args()

    train_df = pd.read_csv(f"{args.train}/train.csv", header=None)
    val_df = pd.read_csv(f"{args.validation}/test.csv", header=None)

    y_train, X_train = train_df.iloc[:, 0], train_df.iloc[:, 1:]
    y_val, X_val = val_df.iloc[:, 0], val_df.iloc[:, 1:]

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    params = {"max_depth": args.max_depth, "eta": args.eta, "objective": args.objective, "eval_metric": "auc"}
    booster = xgb.train(params, dtrain, num_boost_round=args.num_round,
                         evals=[(dval, "validation")])

    preds = (booster.predict(dval) > 0.5).astype(int)
    val_f1 = f1_score(y_val, preds)
    print(f"validation:f1={val_f1:.4f}")

    booster.save_model(os.path.join(args.model_dir, "xgboost-model"))

if __name__ == "__main__":
    main()