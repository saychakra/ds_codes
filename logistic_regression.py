import numpy as np


class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        # Convert z to numpy array if it isn't already
        z = np.array(z)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Convert y to numpy array and ensure correct shape
        y = np.array(y).reshape(-1)

        # Gradient descent
        for _ in range(self.num_iterations):
            # Forward pass
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        return [1 if i > 0.5 else 0 for i in y_predicted]

    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_model)

# Generate sample data
X = np.random.randn(100, 2)    # 100 samples, 2 features
y = np.random.randint(0, 2, 100)   # 100 binary labels

# Use the model
model = LogisticRegression(learning_rate=0.01, num_iterations=1000)

# Print shapes before fitting
print("Initial X shape:", X.shape)
print("Initial y shape:", y.shape)

model.fit(X, y)

predictions = model.predict(X)
probabilities = model.predict_proba(X)

print("X shape:", X.shape)
print("y shape:", y.shape)

print(f"Predictions: {predictions}\nProbabilities: {probabilities}")
