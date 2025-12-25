# Logistic Regression

## What does this code do?
Implements logistic regression from scratch - a linear classification algorithm that uses the sigmoid function to output probabilities. Despite the name, it's a classification algorithm, not regression.

## Key Components
- **Sigmoid Function**: Maps outputs to probability range [0, 1]
- **Log-Loss (Cross-Entropy)**: Classification error metric
- **Gradient Descent**: Optimizes weights to minimize log-loss
- **Decision Boundary**: Threshold (typically 0.5) for classification

## Learning Concepts
- Binary classification fundamentals
- Sigmoid activation function and why it's useful
- Log-loss vs MSE for classification
- Probability interpretation of outputs

## Model Equation
```
z = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ
ŷ = sigmoid(z) = 1 / (1 + e^-z)
```

## Classification Steps
```
Raw input → Linear combination → Sigmoid function 
→ Probability (0-1) → Threshold (0.5) → Class prediction
```

## Related Files
- `linear_regression.py` - Foundational linear model
- `classification_evaluation_metrics_scratch.ipynb` - Evaluate classifier
- `regularizations.py` - Add L1/L2 regularization

## Further Reading
- Why sigmoid function for binary classification
- Maximum likelihood estimation connection
- Multi-class extension (softmax regression)
