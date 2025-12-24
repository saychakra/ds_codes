import numpy as np


class LinearRegression:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = None

    def fit(self, X, y, epochs=1000):
        # Initialize parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Training loop
        for _ in range(epochs):
            # Forward pass (predictions)
            y_predicted = np.dot(X, self.weights) + self.bias

            # Calculate gradients
            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted - y)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

    def mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)

    # Create and train model
    model = LinearRegression(learning_rate=0.01)
    model.fit(X, y.flatten())

    # Make predictions
    predictions = model.predict(X)

    # Calculate error
    error = model.mse(y.flatten(), predictions)
    print(f"Mean Squared Error: {error:.4f}")
    print(f"Weights: {model.weights}, Bias: {model.bias:.4f}")
