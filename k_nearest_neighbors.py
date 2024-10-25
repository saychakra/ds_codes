import numpy as np
from collections import Counter

class KNearestNeighbors:
    def __init__(self, k=3):
        """
        Initialize KNN classifier
        
        Parameters:
        k (int): Number of nearest neighbors to use for prediction
        """
        self.k = k
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        """
        Fit the model using training data
        
        Parameters:
        X (array-like): Training features of shape (n_samples, n_features)
        y (array-like): Target values of shape (n_samples,)
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        
    def euclidean_distance(self, x1, x2):
        """
        Calculate Euclidean distance between two points
        
        Parameters:
        x1, x2 (array-like): Points to calculate distance between
        
        Returns:
        float: Euclidean distance
        """
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def predict(self, X):
        """
        Predict class labels for samples in X
        
        Parameters:
        X (array-like): Samples to predict of shape (n_samples, n_features)
        
        Returns:
        array: Predicted class labels
        """
        X = np.array(X)
        predictions = []
        
        for x in X:
            # Calculate distances between x and all examples in the training set
            distances = []
            for x_train in self.X_train:
                distance = self.euclidean_distance(x, x_train)
                distances.append(distance)
            
            # Get k nearest samples, labels
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = self.y_train[k_indices]
            
            # Majority vote
            most_common = Counter(k_nearest_labels).most_common(1)
            predictions.append(most_common[0][0])
            
        return np.array(predictions)
    
    def score(self, X_test, y_test):
        """
        Calculate prediction accuracy
        
        Parameters:
        X_test (array-like): Test features
        y_test (array-like): True labels for X_test
        
        Returns:
        float: Prediction accuracy between 0.0 and 1.0
        """
        predictions = self.predict(X_test)
        return np.mean(predictions == y_test)