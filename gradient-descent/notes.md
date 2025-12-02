# Gradient Descent Notes

## Core Concept
Gradient descent is an optimization algorithm that iteratively updates parameters to minimize a loss function by moving in the direction of the steepest descent.

## Key Components

- **Objective**: Minimize a loss function (e.g., sum of squared residuals) by adjusting model parameters
- **Gradient**: The derivative of the loss function with respect to each parameter, indicating the direction of steepest increase
- **Update Rule**: Move parameters in the opposite direction of the gradient (negative gradient direction)
- **Step Size**: Controlled by the learning rate - determines how far to move with each update
  - Large steps when far from the optimal point (large gradient magnitude)
  - Small steps when close to the optimal point (small gradient magnitude)

## Convergence
- Gradient descent converges when the step size approaches zero
- This occurs when the gradient is near zero, indicating a local (or global) minimum has been reached
- Convergence criterion: gradient magnitude or parameter change falls below a specified threshold