Your goal is to predict whether a client will subscribe to a bank term deposit.

Approach: Using Tree algorithms for Classification task

## Dataset Insights (750,000 samples, 18 columns):

- Target class heavily imbalanced: 12.07% positive (subscription=yes), 87.93% negative
- Age range: Mean 40.93 (min 18, max 95) - predominantly 25-60 as demographic
- Marital status: Mostly married clients
- Education: Predominantly secondary level (some unknown values)
- Contact: Primarily cellular (with unknown values to handle)
- Balance: High variance (mean 1,204, std 2,836, range -8,019 to 99,717)
- Duration: Mean 256 seconds, highly correlated with subscription (subscribers: 638s vs non-subscribers: 204s)
- Temporal pattern: Contact peaks in May, August, July, June, November, February
- poutcome: Not sure

## Feature Engineering Applied:

1. **Handled missing values**:
   - Education & contact: Converted 'unknown' to NaN, imputed with most_frequent
   - Numbers: Used median imputation
2. **Categorical encoding**: One-Hot Encoder for all categorical variables (job, marital, education, default, housing, loan, contact, month, poutcome)
3. **New engineered features**:
   - has_loan: Binary feature combining housing + loan status
   - has_contact: Binary feature (pdays != -1, indicating previous contact history)
   - balance_over_age: Ratio capturing financial capacity per year of age
4. **Numerical scaling**: StandardScaler on all numeric features (age, balance, duration, campaign, pdays, previous, etc.)

## Model Performance Results:

| Model                          | Precision | Recall | F1-Score | ROC-AUC | Key Finding                       |
| ------------------------------ | --------- | ------ | -------- | ------- | --------------------------------- |
| Decision Tree (max_depth=5)    | 0.7694    | 0.3980 | 0.5246   | 0.9270  | Conservative, underfitting        |
| Random Forest (100 estimators) | 1.0000    | 0.9999 | 1.0000   | 1.0000  | SEVERE OVERFITTING                |
| XGBoost (20 est., lr=0.001)    | 0.0000    | 0.0000 | 0.0000   | 0.9277  | CONFIG ISSUE - predicts all zeros |

## Detailed Analysis:

**Decision Tree (max_depth=5):**

- ROC-AUC: 92.7% (strong discrimination ability)
- Precision: 77% (low false positives) but Recall: 39.8% (misses 60% of subscriptions)
- Problem: max_depth=5 is too shallow, creating underfitting
- Finding: Conservative model that sacrifices recall for high precision
- Fix needed: Increase max_depth, adjust class_weight for imbalance, reduce min_samples_split

**Random Forest (100 estimators):**

- Perfect metrics (1.0000 across all measures) = RED FLAG for overfitting
- The model has memorized training data instead of learning generalizable patterns
- Finding: This will likely perform much worse on test data
- Fix needed: MAJOR REVISION - reduce n_estimators, enable early stopping, use cross-validation, implement max_depth constraints

**XGBoost (20 estimators, learning_rate=0.001, max_depth=5):**

- ROC-AUC paradox: 92.77% AUC but 0% precision/recall (model predicts class=0 for everything)
- Problem: learning_rate=0.001 too conservative + only 20 estimators = insufficient training
- Finding: Model never learned to predict positive class despite having probabilistic discrimination ability
- Fix needed: Increase learning_rate (0.01-0.1), increase n_estimators (100+), add scale_pos_weight for class imbalance, tune objective function
