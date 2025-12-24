import numpy as np


class Regularization:
    @staticmethod
    def l1_regularization(weights, lambda_param):
        """
        Compute L1 regularization term and its gradient
        L1 = λ * Σ|w|
        """
        reg_term = lambda_param * np.sum(np.abs(weights))
        gradient = lambda_param * np.sign(weights)
        return reg_term, gradient

    @staticmethod
    def l2_regularization(weights, lambda_param):
        """
        Compute L2 regularization term and its gradient
        L2 = λ * Σ(w²)
        """
        reg_term = 0.5 * lambda_param * np.sum(weights ** 2)  # 0.5 for easier derivative
        gradient = lambda_param * weights
        return reg_term, gradient

    @staticmethod
    def elastic_net(weights, lambda_param, l1_ratio=0.5):
        """
        Compute Elastic Net regularization (combination of L1 and L2)
        Elastic Net = α * L1 + (1-α) * L2
        """
        l1_term, l1_grad = Regularization.l1_regularization(weights, lambda_param * l1_ratio)
        l2_term, l2_grad = Regularization.l2_regularization(weights, lambda_param * (1 - l1_ratio))
        return l1_term + l2_term, l1_grad + l2_grad

######################### training a linear regression model with regularization #########################
class LinearRegressionWithRegularization:
    def __init__(self, lambda_param=0.1, regularization_type='l2'):
        self.lambda_param = lambda_param
        self.regularization_type = regularization_type
        self.weights = None
        self.bias = None

    def initialize_parameters(self, n_features):
        """Initialize weights and bias"""
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0

    def compute_regularization(self):
        """Compute regularization term and gradient based on specified type"""
        if self.regularization_type == 'l1':
            return Regularization.l1_regularization(self.weights, self.lambda_param)
        elif self.regularization_type == 'l2':
            return Regularization.l2_regularization(self.weights, self.lambda_param)
        elif self.regularization_type == 'elastic_net':
            return Regularization.elastic_net(self.weights, self.lambda_param)
        else:
            return 0, np.zeros_like(self.weights)

    def compute_cost(self, X, y):
        """
        Compute cost function with regularization
        J = MSE + regularization_term
        """
        predictions = np.dot(X, self.weights) + self.bias
        mse = np.mean((predictions - y) ** 2)

        # Add regularization term
        reg_term, _ = self.compute_regularization()

        return mse + reg_term

    def compute_gradients(self, X, y):
        """
        Compute gradients of cost function with regularization
        """
        m = len(y)
        predictions = np.dot(X, self.weights) + self.bias

        # Compute gradients for MSE
        dw = (2/m) * np.dot(X.T, (predictions - y))
        db = (2/m) * np.sum(predictions - y)

        # Add regularization gradient
        _, reg_gradient = self.compute_regularization()
        dw += reg_gradient

        return dw, db

    def train(self, X, y, learning_rate=0.01, n_iterations=1000):
        """
        Train the model using gradient descent
        """
        if self.weights is None:
            self.initialize_parameters(X.shape[1])

        cost_history = []

        for i in range(n_iterations):
            # Compute gradients
            dw, db = self.compute_gradients(X, y)

            # Update parameters
            self.weights -= learning_rate * dw
            self.bias -= learning_rate * db

            # Compute cost and store
            if i % 100 == 0:
                cost = self.compute_cost(X, y)
                cost_history.append(cost)
                print(f"Iteration {i}: Cost = {cost:.4f}")

        return cost_history

    def predict(self, X):
        """Make predictions"""
        return np.dot(X, self.weights) + self.bias

def generate_sample_data(n_samples=100, n_features=2, noise=0.1):
    """Generate sample data for testing"""
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    true_weights = np.array([1.0, -2.0])
    y = np.dot(X, true_weights) + 0.5 + noise * np.random.randn(n_samples)
    return X, y

def compare_regularizations():
    """Compare different regularization methods"""
    # Generate data
    X, y = generate_sample_data(n_samples=100, n_features=2)

    # List of regularization types to test
    reg_types = ['none', 'l1', 'l2', 'elastic_net']

    results = {}

    for reg_type in reg_types:
        # Create and train model
        model = LinearRegressionWithRegularization(
            lambda_param=0.1,
            regularization_type=reg_type
        )

        # Train model
        cost_history = model.train(X, y, learning_rate=0.01, n_iterations=1000)

        # Store results
        results[reg_type] = {
            'weights': model.weights,
            'bias': model.bias,
            'final_cost': cost_history[-1]
        }

    # Print results
    print("\nComparison of Regularization Methods:")
    print("-" * 50)
    for reg_type, result in results.items():
        print(f"\n{reg_type.upper()} Regularization:")
        print(f"Weights: {result['weights']}")
        print(f"Bias: {result['bias']:.4f}")
        print(f"Final Cost: {result['final_cost']:.4f}")

def visualize_regularization_effects():
    """Visualize how different lambda values affect weights"""
    import matplotlib.pyplot as plt

    # Generate data
    X, y = generate_sample_data(n_samples=100, n_features=2)

    # Test different lambda values
    lambda_values = [0, 0.01, 0.1, 1.0, 10.0]
    reg_types = ['l1', 'l2']

    weights = {reg_type: [] for reg_type in reg_types}

    for reg_type in reg_types:
        for lambda_param in lambda_values:
            model = LinearRegressionWithRegularization(
                lambda_param=lambda_param,
                regularization_type=reg_type
            )
            model.train(X, y)
            weights[reg_type].append(model.weights)

    # Plot results
    plt.figure(figsize=(12, 5))

    for i, reg_type in enumerate(reg_types):
        plt.subplot(1, 2, i+1)
        weights_array = np.array(weights[reg_type])
        plt.plot(lambda_values, weights_array[:, 0], 'b-', label='Weight 1')
        plt.plot(lambda_values, weights_array[:, 1], 'r-', label='Weight 2')
        plt.xscale('log')
        plt.xlabel('Lambda')
        plt.ylabel('Weight Value')
        plt.title(f'{reg_type.upper()} Regularization')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Compare different regularization methods
    compare_regularizations()

    # Visualize regularization effects
    visualize_regularization_effects()
