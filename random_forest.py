import numpy as np
from collections import Counter

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, n_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.root = None

    class Node:
        def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
            self.feature = feature  # Index of feature to split on
            self.threshold = threshold  # Threshold value for the split
            self.left = left  # Left child node
            self.right = right  # Right child node
            self.value = value  # For leaf nodes, stores the predicted class

    def fit(self, X, y):
        self.n_classes = len(set(y))
        if self.n_features is None:
            self.n_features = X.shape[1]
        else:
            self.n_features = min(self.n_features, X.shape[1])
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split or \
           n_labels == 1:
            leaf_value = self._most_common_label(y)
            return self.Node(value=leaf_value)

        # Select random features for consideration
        feat_idxs = np.random.choice(n_features, self.n_features, replace=False)

        # Find the best split
        best_feature, best_threshold = self._best_split(X, y, feat_idxs)
        
        if best_feature is None:  # No valid split found
            leaf_value = self._most_common_label(y)
            return self.Node(value=leaf_value)

        # Create child splits
        left_idxs = X[:, best_feature] < best_threshold
        right_idxs = ~left_idxs
        left = self._grow_tree(X[left_idxs], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs], y[right_idxs], depth + 1)

        return self.Node(best_feature, best_threshold, left, right)

    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        best_feature = None
        best_threshold = None

        # Calculate current entropy
        current_entropy = self._entropy(y)

        # Try each feature
        for feature in feat_idxs:
            thresholds = np.unique(X[:, feature])
            
            # Try each threshold
            for threshold in thresholds:
                # Split the data
                left_mask = X[:, feature] < threshold
                right_mask = ~left_mask
                
                if len(y[left_mask]) == 0 or len(y[right_mask]) == 0:
                    continue

                # Calculate information gain
                left_entropy = self._entropy(y[left_mask])
                right_entropy = self._entropy(y[right_mask])
                n = len(y)
                n_l = sum(left_mask)
                n_r = sum(right_mask)
                gain = current_entropy - (n_l/n * left_entropy + n_r/n * right_entropy)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        ps = ps[ps > 0]  # Remove zero probabilities
        return -np.sum(ps * np.log2(ps))

    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value

        if x[node.feature] < node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

class RandomForest:
    def __init__(self, n_trees=100, max_depth=None, min_samples_split=2, n_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            # Bootstrap sampling
            idxs = np.random.choice(len(X), size=len(X), replace=True)
            bootstrap_X = X[idxs]
            bootstrap_y = y[idxs]
            
            # Create and train tree
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_features=self.n_features
            )
            tree.fit(bootstrap_X, bootstrap_y)
            self.trees.append(tree)

    def predict(self, X):
        # Get predictions from all trees
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        # Return majority vote for each sample
        return np.array([Counter(predictions).most_common(1)[0][0] 
                        for predictions in tree_predictions.T])

# Example usage
def example_usage():
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(100, 5)  # 100 samples, 5 features
    y = (X[:, 0] + X[:, 1] > 0).astype(int)  # Simple binary classification

    # Create and train random forest
    rf = RandomForest(n_trees=10, max_depth=5)
    rf.fit(X, y)

    # Make predictions
    y_pred = rf.predict(X)
    
    # Calculate accuracy
    accuracy = np.mean(y_pred == y)
    print(f"Accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    example_usage()