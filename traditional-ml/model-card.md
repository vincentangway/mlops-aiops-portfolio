# Model Card — Churn Detection (XGBoost)

## Intended use
Predicts whether a telecom customer will churn (cancel service), to support proactive retention outreach (e.g. targeted discounts or check-in calls for high-risk customers). Intended for internal retention-team decision support, not for fully automated customer-facing actions.

## Training data
- Source: Kaggle [`blastchar/telco-customer-churn`](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (Telco Customer Churn dataset), downloaded 2026-07-23.
- 7,043 customer records, 20 features (demographics, account info, subscribed services) + target `Churn`.
- Class balance: 73.5% No churn / 26.5% Churn.
- Split: 80/20 stratified train/test (5,634 / 1,409 rows).

## Model
- Algorithm: XGBoost (binary logistic classification), SageMaker XGBoost framework container v1.7-1.
- Hyperparameters: `max_depth=5`, `eta=0.173`, `num_round=100`, `objective=binary:logistic`, `scale_pos_weight=2.77` (weights the minority/churn class to counter the ~73/27 class imbalance).
- Features: numeric (`tenure`, `MonthlyCharges`, `TotalCharges`) standard-scaled; categorical fields one-hot encoded. `customerID` dropped (identifier, not predictive). `TotalCharges` nulls (11 rows, new customers with `tenure=0`) imputed as 0.

## Key metrics (held-out test set)
- **F1 score: 0.6211** (target metric — chosen over raw accuracy due to class imbalance).
- **AUC: 0.831**.
- Acceptance gate: pipeline's `ConditionStep` requires F1 > 0.60 before registering a model; this run cleared it.

## Pipeline
Built as a SageMaker Pipeline (`churn-detection-pipeline`) with 4 chained steps: `PreprocessChurnData` → `TrainChurnModel` → `EvaluateChurnModel` → `CheckF1Threshold` (conditionally registers to the Model Registry only if F1 > 0.60). Re-running the pipeline reproduces this process end-to-end from raw data to a versioned, registered model.

## Known limitations
- Trained on a static historical snapshot; no mechanism yet to detect concept drift as customer behavior or plans change over time (drift monitoring is planned for Week 2).
- Default classification threshold (probability > 0.5) is not tuned for a specific business cost tradeoff between false positives (wasted retention spend) and false negatives (missed at-risk customers) — worth revisiting once outreach costs/ROI are known.
- No fairness/bias audit performed across demographic segments (e.g. `gender`, `SeniorCitizen`).
- Dataset is a single point-in-time snapshot from one telecom provider; may not generalize to other markets or pricing structures.
- Hyperparameters were lightly tuned (targeted search on `eta`/`max_depth`, then `scale_pos_weight` added specifically to clear the acceptance gate) rather than exhaustively optimized.

## Version history
| Version | F1 | AUC | Status | Notes |
|---------|-----|-----|--------|-------|
| 1 | 0.6211 | 0.831 | **Approved** | First successful pipeline run |
| 2 | 0.6211 | 0.831 | Pending | Duplicate re-run, same config — not approved |
