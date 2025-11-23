# Decision Trees

## Core Concepts

### Fundamental Principle
Decision trees recursively split data into subsets based on feature values to create a tree structure. Each node represents a decision point (split), and leaf nodes represent final predictions. The goal is to create **pure splits** where each partition contains mostly one class.

### How It Works
- Start with all data at the root node
- At each node, find the feature and threshold that creates the best split
- Measure split quality using **impurity measures**: Gini impurity or Entropy
- Recursively split child nodes until stopping criteria is met
- Make predictions by following splits down to a leaf node

### Impurity Measures
- **Gini Impurity**: Probability of incorrectly classifying a randomly picked item from a node
  - Range: 0 (pure) to 0.5 (for binary classification, maximally mixed)
  - Lower Gini = better split
- **Entropy**: Information-theoretic measure of disorder in a node
  - Range: 0 (pure) to log(n_classes) (maximally mixed)
  - 0 entropy = only one class present (perfect purity)
- **Choose best split**: One that maximizes information gain (reduction in impurity)

### Handling Overfitting (Regularization)
Decision trees tend to overfit because they can grow very deep and create overly specific rules:
- **Max depth**: Limit tree height (most effective parameter)
- **Min samples split**: Minimum samples required to split a node
- **Min samples leaf**: Minimum samples required at leaf nodes
- Increasing min_samples or reducing max_depth → **more regularization** → simpler model

### For Regression
Decision trees can also perform regression:
- Split data to minimize mean squared error (MSE) instead of impurity
- Leaf nodes contain average values instead of class predictions
- Same regularization principles apply

## Key Takeaways

1. **Interpretable and visual**: Easy to explain decisions made by the model (white-box)
2. **Non-linear splits**: Can capture complex non-linear relationships
3. **Axis-aligned splits**: Always splits perpendicular to axes; **sensitive to axis rotation** (flipped data changes model)
4. **High variance**: Small changes in data or hyperparameters → different tree structure
5. **Greedy algorithm**: Makes locally optimal splits; doesn't guarantee globally optimal tree
6. **Both classification and regression**: Versatile for different problem types
7. **Works with mixed data types**: Can handle both categorical and numerical features naturally
8. **Prone to overfitting**: Needs careful hyperparameter tuning to generalize well