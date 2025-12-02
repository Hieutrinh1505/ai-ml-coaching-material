# K-Nearest Neighbors (KNN)

## Core Concepts

### Fundamental Principle
KNN operates on a simple premise: **data points with similar features tend to have similar labels**. Given a new data point, we find the K nearest neighbors in the training set and use them to make a prediction.

### How It Works
- Find K closest data points (usually measured by Euclidean distance)
- For **classification**: use majority voting among K neighbors
- For **regression**: use average value of K neighbors
- Supervised learning algorithm that makes no assumptions about data distribution (non-parametric)

### K Parameter Trade-off
- **Low K (e.g., K=1,3)**: More flexible, responsive to local patterns, but **sensitive to noise and outliers**
- **High K**: Smoother predictions, more robust to noise, but **may miss local patterns** and underfit
- **Golden zone**: Balance between capturing patterns and avoiding overfitting. Avoid K so large that minority classes get permanently voted out.

### Algorithm Properties
- **Non-linear algorithm**: No linear boundary separates classes
- **Works for both classification and regression**
- **Lazy learner**: No training phase, computation happens at prediction time
- **Instance-based learning**: Each prediction depends on actual training examples

## Key Takeaways

1. **Simple yet effective**: Intuitive algorithm that often serves as a good baseline
2. **No training overhead**: Fast to implement, but slow at prediction time (must compute distances to all training samples)
3. **Sensitive to feature scaling**: Need to normalize/standardize features; distance is meaningless if one feature dominates
4. **Limited complexity capture**: Struggles with complex non-linear relationships and high-dimensional data (curse of dimensionality)
5. **Choose K wisely**: Typically K=3-5 for classification, but validate with cross-validation
6. **Memory intensive**: Stores entire training dataset; impractical for very large datasets


elbow method for k
minkowski
