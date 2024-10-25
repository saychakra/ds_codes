import numpy as np

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        
    def _initialize_weights_and_bias(self, X):
        n_features = X.shape[1]
        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0
        
    def _get_hyperplane_distance(self, X):
        return np.dot(X, self.weights) - self.bias
    
    def _compute_cost(self, distance, y):
        # Compute hinge loss
        hinge_loss = np.maximum(0, 1 - y * distance)
        # Add L2 regularization term
        return np.mean(hinge_loss) + self.lambda_param * np.sum(self.weights ** 2)
    
    def fit(self, X, y):
        """
        Train the SVM classifier
        
        Parameters:
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values (1 or -1)
        """
        self._initialize_weights_and_bias(X)
        
        # Gradient descent
        for _ in range(self.n_iters):
            distances = self._get_hyperplane_distance(X)
            
            # Compute gradients
            dw = np.zeros_like(self.weights)
            db = 0
            
            for idx, distance in enumerate(distances):
                if y[idx] * distance <= 1:
                    dw += -y[idx] * X[idx]
                    db += y[idx]
            
            # Add regularization term gradient
            dw = dw / len(y) + 2 * self.lambda_param * self.weights
            db = db / len(y)
            
            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    
    def predict(self, X):
        """
        Predict class labels for samples in X
        
        Parameters:
        X : array-like of shape (n_samples, n_features)
            The data matrix for which we want to predict the targets.
            
        Returns:
        array-like of shape (n_samples,) : Predicted class labels
        """
        distances = self._get_hyperplane_distance(X)
        return np.sign(distances)
    
    def score(self, X, y):
        """
        Calculate accuracy score
        
        Parameters:
        X : array-like of shape (n_samples, n_features)
            Test samples
        y : array-like of shape (n_samples,)
            True labels for X
            
        Returns:
        float : Accuracy of the model
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)

# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = np.where(X[:, 0] + X[:, 1] > 0, 1, -1)
    
    # Create and train the model
    svm = SVM(learning_rate=0.01, lambda_param=0.01, n_iters=1000)
    svm.fit(X, y)
    
    # Make predictions
    predictions = svm.predict(X)
    
    # Calculate accuracy
    accuracy = svm.score(X, y)
    print(f"Accuracy: {accuracy:.2f}")