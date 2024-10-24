import numpy as np
from collections import defaultdict

class XGBoostTree:
    def __init__(self, max_depth=3, min_samples_split=2, learning_rate=0.1, min_child_weight=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.learning_rate = learning_rate
        self.min_child_weight = min_child_weight
        self.root = None
        
    class Node:
        def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
            self.feature = feature
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value
            
    def _calculate_gain(self, g, h, g_l, h_l, g_r, h_r, lambda_=1.0):
        """Calculate gain for split using the XGBoost formula"""
        def calc_term(g_sum, h_sum):
            return (g_sum * g_sum) / (h_sum + lambda_)
        
        gain = 0.5 * (calc_term(g_l, h_l) + calc_term(g_r, h_r) - calc_term(g, h))
        return gain
    
    def _calculate_leaf_value(self, g, h, lambda_=1.0):
        """Calculate leaf value using the XGBoost formula"""
        return -g / (h + lambda_)
    
    def _find_best_split(self, X, g, h):
        """Find the best split for a node"""
        m = len(X)
        best_gain = -np.inf
        best_feature = None
        best_threshold = None
        
        if m < self.min_samples_split:
            return None, None
        
        total_g = np.sum(g)
        total_h = np.sum(h)
        
        for feature in range(X.shape[1]):
            # Sort feature values and gradients
            sorted_idx = np.argsort(X[:, feature])
            sorted_x = X[sorted_idx, feature]
            sorted_g = g[sorted_idx]
            sorted_h = h[sorted_idx]
            
            # Try all possible splits
            g_left = 0
            h_left = 0
            
            for i in range(0, m - 1):
                g_left += sorted_g[i]
                h_left += sorted_h[i]
                
                # Skip if this split doesn't satisfy minimum child weight
                if h_left < self.min_child_weight or \
                   (total_h - h_left) < self.min_child_weight:
                    continue
                
                # Skip duplicate values
                if i < m - 1 and sorted_x[i] == sorted_x[i + 1]:
                    continue
                
                g_right = total_g - g_left
                h_right = total_h - h_left
                
                gain = self._calculate_gain(total_g, total_h, g_left, h_left, 
                                         g_right, h_right)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = (sorted_x[i] + sorted_x[i + 1]) / 2
                    
        return best_feature, best_threshold
    
    def _build_tree(self, X, g, h, depth=0):
        """Recursively build the tree"""
        if depth >= self.max_depth:
            return self.Node(value=self._calculate_leaf_value(np.sum(g), np.sum(h)))
        
        feature, threshold = self._find_best_split(X, g, h)
        
        if feature is None:
            return self.Node(value=self._calculate_leaf_value(np.sum(g), np.sum(h)))
        
        # Split the data
        left_mask = X[:, feature] < threshold
        right_mask = ~left_mask
        
        # Create child nodes
        left_node = self._build_tree(X[left_mask], g[left_mask], h[left_mask], depth + 1)
        right_node = self._build_tree(X[right_mask], g[right_mask], h[right_mask], depth + 1)
        
        return self.Node(feature=feature, threshold=threshold, left=left_node, right=right_node)
    
    def fit(self, X, g, h):
        """Build the tree using gradients and hessians"""
        self.root = self._build_tree(X, g, h)
        return self
    
    def predict(self, X):
        """Make predictions for samples in X"""
        return np.array([self._predict_sample(x, self.root) for x in X])
    
    def _predict_sample(self, x, node):
        """Predict single sample"""
        if node.value is not None:
            return node.value
        
        if x[node.feature] < node.threshold:
            return self._predict_sample(x, node.left)
        return self._predict_sample(x, node.right)

class XGBoost:
    def __init__(self, n_estimators=100, max_depth=3, learning_rate=0.1, 
                 min_samples_split=2, min_child_weight=1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_samples_split = min_samples_split
        self.min_child_weight = min_child_weight
        self.trees = []
        self.base_prediction = None
    
    def _logistic(self, x):
        """Logistic (sigmoid) function"""
        return 1 / (1 + np.exp(-x))
    
    def _gradient(self, y_true, y_pred):
        """Calculate gradients for logistic loss"""
        pred = self._logistic(y_pred)
        return pred - y_true
    
    def _hessian(self, y_true, y_pred):
        """Calculate hessians for logistic loss"""
        pred = self._logistic(y_pred)
        return pred * (1 - pred)
    
    def fit(self, X, y):
        """Fit the XGBoost model"""
        # Initialize predictions with base value
        self.base_prediction = np.log(np.mean(y) / (1 - np.mean(y)))
        y_pred = np.full_like(y, self.base_prediction, dtype=float)
        
        # Build trees iteratively
        for _ in range(self.n_estimators):
            # Calculate gradients and hessians
            gradients = self._gradient(y, y_pred)
            hessians = self._hessian(y, y_pred)
            
            # Create and train tree
            tree = XGBoostTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                learning_rate=self.learning_rate,
                min_child_weight=self.min_child_weight
            )
            tree.fit(X, gradients, hessians)
            
            # Update predictions
            predictions = tree.predict(X)
            y_pred += self.learning_rate * predictions
            
            # Store tree
            self.trees.append(tree)
            
        return self
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        # Start with base prediction
        y_pred = np.full(X.shape[0], self.base_prediction, dtype=float)
        
        # Add predictions from all trees
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)
            
        # Convert to probabilities
        return self._logistic(y_pred)
    
    def predict(self, X, threshold=0.5):
        """Predict class labels"""
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)

# Example usage
def example_usage():
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(1000, 5)
    y = (X[:, 0] * X[:, 1] + X[:, 2] > 0).astype(int)
    
    # Create and train model
    model = XGBoost(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        min_samples_split=2,
        min_child_weight=1
    )
    model.fit(X, y)
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Calculate accuracy
    accuracy = np.mean(y_pred == y)
    print(f"Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    example_usage()