# Gradient Boosting

## Core Concepts

### Fundamental Principle
Gradient Boosting sequentially builds shallow trees, where each new tree **fits the residual errors** of all previous trees combined. This "corrects mistakes" one step at a time through incremental improvements.

### How It Works

**Initialization:**
- Start with a constant prediction (usually mean of target values)
- This becomes the baseline for all samples

**Iterative Process (repeat for each tree):**
1. **Calculate residuals**: Error = Actual - Predicted for each sample
2. **Fit new tree**: Build shallow tree (limited depth, ~5 nodes) to predict residuals
3. **Compute leaf values (gamma)**: For each leaf node, calculate optimal constant that minimizes loss
   - Value depends on samples falling into that leaf
4. **Update predictions**: Add tree's predictions scaled by learning rate
   - `new_prediction = old_prediction + learning_rate × tree_prediction`
5. **Loop**: Repeat until stopping criteria (max trees or early stopping)

### Learning Rate Trade-off
- **Small learning rate** (e.g., 0.01): Baby steps → many iterations needed → lower variance → better generalization
- **Large learning rate** (e.g., 0.3): Aggressive steps → fewer iterations → higher variance → risk of overfitting
- **Best practice**: Use small learning rate + more trees (requires early stopping to prevent overfitting)

### Key Differences from Random Forest
- **Sequential vs Parallel**: Gradient Boosting trains trees sequentially; Random Forest trains independently
- **Dependence**: Each tree depends on previous errors; Random Forest trees don't depend on each other
- **Shallow vs Deep**: Gradient Boosting uses shallow trees; Random Forest uses deeper trees
- **Bagging vs Boosting**: Different reduction strategies for variance and bias

## Gradient Boosting for Classification

### Key Difference: Probability-Based Residuals
Classification uses **log-odds** (probabilities) instead of raw continuous values. The core algorithm remains the same, but residuals are computed differently.

### How Classification Works

**Initialization:**
- Start with log-odds: `log(p / (1-p))` where p = proportion of positive class
- Convert to probability: Apply logistic function

**Iterative Process (repeat for each tree):**
1. **Calculate residuals**:
   - If actual = 1 (Yes): `residual = 1 - probability`
   - If actual = 0 (No): `residual = 0 - probability = -probability`
   - Residual always ranges from -1 to +1

2. **Fit new tree**: Build shallow tree (depth 3-5) to predict residuals
   - Find splits that minimize weighted sum of squared residuals

3. **Compute leaf values (gamma)**: For each leaf node:
   - Sum of residuals (for that leaf)
   - Divide by: `sum(probability × (1 - probability))` for samples in leaf
   - Formula: `gamma = Σ(residuals) / Σ(p × (1-p))`
   - This is the Newton step (second-order optimization)

4. **Update predictions**:
   - Convert current log-odds to probability
   - Add scaled tree prediction to log-odds
   - Convert back to probability via logistic function

5. **Loop**: Repeat until max trees or early stopping

### Why This Works
- **`p × (1-p)` term**: Weights samples based on uncertainty
  - When p ≈ 0.5: high weight (uncertain, learn more)
  - When p ≈ 0 or 1: low weight (confident, learn less)
- **Logistic function** naturally bounds predictions to [0,1]
- **Log-odds space** allows additive updates that map back to valid probabilities

### Classification vs Regression Comparison
| Aspect | Regression | Classification |
|--------|-----------|-----------------|
| **Loss function** | Squared error | Log loss (cross-entropy) |
| **Residual** | Actual - Predicted | Actual - Probability |
| **Leaf value denominator** | Count of samples | Σ(p × (1-p)) |
| **Prediction space** | Continuous | Probability [0,1] |
| **Update mechanism** | Direct addition | Through log-odds |

## Key Takeaways

1. **Sequential correction**: Each tree learns from previous tree's mistakes (residuals)
2. **Shallow trees work best**: Limited depth (3-5 splits) prevents overfitting; encourages diversity
3. **Learning rate is critical**: Controls step size and convergence speed; small rates generalize better
4. **Boosting reduces both bias and variance**: Unlike bagging, can reduce both simultaneously through iterative refinement
5. **Requires careful tuning**: More hyperparameters than Random Forest; needs validation to prevent overfitting
6. **Slower training**: Sequential nature means training takes longer than bagging methods
7. **Strong learner result**: Combines weak learners into a very strong, competitive model
8. **Computational cost**: Each tree update requires full dataset evaluation; more expensive than tree-based methods
9. **Classification-specific**: Uses probability-based residuals and Newton step optimization (p × (1-p) weighting)
10. **Uncertainty weighting**: More uncertain predictions (p ≈ 0.5) get higher weight, driving better learning