# Random Forest & Ensemble Methods

## Core Concepts

### Random Forest Fundamentals
Random Forest is an **ensemble method** that combines multiple decision trees to make predictions:
- Train many decision trees on **different subsets** of training data (with replacement - "bagging")
- Each tree is trained independently on a random sample
- For **classification**: use majority voting across all trees
- For **regression**: use average prediction across all trees
- Result: More robust, stable predictions than single decision tree

### Why Ensemble Methods Work: Law of Large Numbers
- Individual weak learners (mediocre predictors) can combine into a strong learner
- Like flipping a coin: odd of getting heads = 50%, but with many flips, average approaches 50% (convergence)
- Ensemble reduces variance through **averaging or voting** across diverse predictions

### Bagging (Bootstrap Aggregating)
- **Sampling with replacement**: Each tree gets a random sample where same data point can appear multiple times
- Different subsets create diversity in trees → different decision boundaries
- All trees are trained in parallel (independent)
- Out-of-bag (OOB) samples: ~37% of data not in a tree's bootstrap sample can be used for validation without cross-validation

### Alternative Sampling Approaches
- **Pasting**: Sampling without replacement (rarely used)
- **Random Patches**: Subsample both rows AND columns (features)
- **Random Subspaces**: Subsample only features, all samples
- These add feature diversity on top of data diversity

### Ensemble Voting Methods
- **Hard Voting**: Each predictor casts one vote, majority wins (used for Random Forest)
- **Soft Voting**: Use predicted probabilities, average them, choose class with highest average probability
- Can combine different algorithms (not just trees) in voting classifiers

### Boosting Methods (Alternative Ensemble Strategy)
**Concept**: Combine weak learners sequentially, each new learner focuses on errors of previous ones
- **AdaBoost/SAMME**: Assign higher weights to misclassified samples, train next classifier on weighted data
- **Gradient Boosting**: Don't reweight; instead, train next predictor to fit residual errors (predictions - actual values)
  - Lower learning rate → smaller step size → need more trees to fit well
  - More trees + low learning rate = risk of overfitting
  - Use cross-validation or early stopping (n_iter_no_change) to find optimal number of trees

### Gradient Boosting Variants
- **Standard Gradient Boosting**: Sequential, slower training
- **Histogram-Based Gradient Boosting**: Faster on large datasets, uses histogram approximations

### Stacking (Advanced Ensemble)
- Multiple base predictors make predictions on training data (via cross-validation)
- Predictions become input features to a meta-learner (blender)
- Meta-learner learns how to best combine predictions
- More complex but potentially better performance

### Feature Importance in Random Forest
- Measure how much each feature contributes to reducing impurity across all trees
- More reliable than single tree (averaged over many trees)
- Helps identify most influential features

### Extra Trees (Extremely Randomized Trees)
- **Trade-off**: Accept slightly more bias for significantly lower variance
- At each split, use random thresholds instead of optimized thresholds
- Faster training than Random Forest
- Often produces similar performance

## Key Takeaways

1. **Reduces overfitting**: Bootstrap sampling and averaging reduce variance from high-variance trees
2. **Captures non-linear relationships**: Multiple trees can model complex patterns
3. **Robust to outliers**: Single outlier less likely to affect ensemble prediction
4. **Handles mixed data types**: Inherits benefits of decision trees
5. **Feature importance**: Identifies which features matter most
6. **Fast parallelization**: Bagging trees can train in parallel (unlike boosting)
7. **No feature scaling needed**: Trees are scale-invariant
8. **Boosting vs Bagging trade-off**:
   - **Bagging (Random Forest)**: Fast training, reduces variance, good baseline
   - **Boosting**: Sequential training (slower), can reduce both bias and variance, requires careful tuning
9. **Hyperparameter tuning matters**: Number of trees, depth, samples per split affect performance
10. **Out-of-Bag evaluation**: Can estimate model performance without separate validation set