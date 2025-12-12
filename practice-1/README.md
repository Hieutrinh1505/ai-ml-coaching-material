Predict whether a client will subscribe to a bank term deposit using tree-based classification.

## Features

**Categorical (9):** job, marital, education, default, housing, loan, contact, month, poutcome

**Numerical (7):** age, balance, day, duration, campaign, pdays, previous

## Feature Engineering

- **Label Encoding:** Transformed all features to categorical via LabelEncoder (optimal for tree models)
- **Feature Interactions:** Created 120 feature pairs (C(16,2)) combining encoded features for richer representations

## Models Trained

| Model         | ROC-AUC    | Precision | Recall | F1-Score | Best Params                                         |
| ------------- | ---------- | --------- | ------ | -------- | --------------------------------------------------- |
| Decision Tree | **0.9476** | 0.6921    | 0.6515 | 0.6714   | max_depth=10, min_samples_leaf=8                    |
| Random Forest | **0.9528** | 0.6906    | 0.6689 | 0.6796   | max_depth=10, max_features='sqrt', n_estimators=100 |
| XGBoost       | **0.9650** | 0.7186    | 0.7012 | 0.7098   | learning_rate=0.1, max_leaves=64                    |

## Key Insights

- **XGBoost wins** (ROC-AUC: 96.5%, F1: 0.7098) - Best precision & recall balance
- **Random Forest solid** (ROC-AUC: 95.3%, F1: 0.6796) - Stable and reliable
- **Decision Tree baseline** (ROC-AUC: 94.8%, F1: 0.6714) - Good interpretability but lower performance

## Top 20 Important Features (XGBoost)

1. duration_tf
2. poutcome_tf
3. month_tf_poutcome_tf
4. contact_tf_duration_tf
5. pdays_tf
6. poutcome_tf_duration_tf
7. previous_tf
8. day_tf_duration_tf
9. campaign_tf
10. marital_tf_duration_tf
11. balance_tf
12. job_tf_duration_tf
13. age_tf_duration_tf
14. contact_tf_poutcome_tf
15. poutcome_tf_previous_tf
16. duration_tf_campaign_tf
17. education_tf_duration_tf
18. poutcome_tf_age_tf
19. month_tf
20. poutcome_tf_balance_tf

**Key Finding:** Duration dominates prediction - longer calls correlate strongly with subscription likelihood.

## Possible Improvements

1. **Ensemble Methods:** Combine models via Voting Classifier or Stacking to leverage diverse predictions
2. **CatBoost:** Test gradient boosting optimized for categorical features (skip label encoding)
