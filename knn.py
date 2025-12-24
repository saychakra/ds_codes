import matplotlib.pyplot as plt
import numpy as np


class KMeansClustering:
    def __init__(self, k=3, max_iters=100, random_state=None):
        """
        Initialize KMeans clustering algorithm

        Parameters:
        k (int): Number of clusters
        max_iters (int): Maximum number of iterations
        random_state (int): Random seed for reproducibility
        """
        self.k = k
        self.max_iters = max_iters
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        self.inertia_ = None  # Sum of squared distances to closest centroid

    def fit(self, X):
        """
        Fit KMeans clustering to the data

        Parameters:
        X (array-like): Training data of shape (n_samples, n_features)
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        n_samples, n_features = X.shape

        # Initialize centroids randomly
        random_indices = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[random_indices]

        for _ in range(self.max_iters):
            # Store old centroids
            old_centroids = self.centroids.copy()

            # Assign points to nearest centroid
            self.labels = self._assign_clusters(X)

            # Update centroids
            self._update_centroids(X)

            # Check convergence
            if self._has_converged(old_centroids):
                break

        # Calculate inertia
        self.inertia_ = self._calculate_inertia(X)

        return self

    def _assign_clusters(self, X):
        """
        Assign each point to the nearest centroid

        Parameters:
        X (array-like): Data points

        Returns:
        array: Cluster labels for each point
        """
        distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)

    def _update_centroids(self, X):
        """
        Update centroid positions based on mean of assigned points

        Parameters:
        X (array-like): Data points
        """
        for k in range(self.k):
            if np.sum(self.labels == k) > 0:  # Avoid empty clusters
                self.centroids[k] = np.mean(X[self.labels == k], axis=0)

    def _has_converged(self, old_centroids, tol=1e-4):
        """
        Check if the algorithm has converged

        Parameters:
        old_centroids (array-like): Centroids from previous iteration
        tol (float): Tolerance for convergence

        Returns:
        bool: True if converged, False otherwise
        """
        return np.all(np.abs(old_centroids - self.centroids) < tol)

    def _calculate_inertia(self, X):
        """
        Calculate sum of squared distances to nearest centroid

        Parameters:
        X (array-like): Data points

        Returns:
        float: Inertia value
        """
        distances = np.sqrt(((X - self.centroids[self.labels])**2).sum(axis=1))
        return np.sum(distances**2)

    def predict(self, X):
        """
        Predict cluster labels for new data

        Parameters:
        X (array-like): New data points

        Returns:
        array: Predicted cluster labels
        """
        return self._assign_clusters(X)

    def plot_clusters(self, X, title="K-Means Clustering Results"):
        """
        Plot the clusters and centroids (works for 2D data)

        Parameters:
        X (array-like): Data points
        title (str): Plot title
        """
        if X.shape[1] != 2:
            raise ValueError("Plotting is only supported for 2D data")

        plt.figure(figsize=(8, 6))
        plt.scatter(X[:, 0], X[:, 1], c=self.labels, cmap='viridis')
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1],
                   c='red', marker='x', s=200, linewidths=3,
                   label='Centroids')
        plt.title(title)
        plt.legend()
        plt.show()
