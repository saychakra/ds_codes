import numpy as np
from scipy.stats import multivariate_normal


class GaussianMixture:
    def __init__(self, n_components, max_iter=100, tol=1e-4):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol

    def initialize_parameters(self, X):
        n_samples, n_features = X.shape

        # Initialize mixing coefficients
        self.weights = np.ones(self.n_components) / self.n_components

        # Initialize means randomly
        random_idx = np.random.choice(n_samples, self.n_components, replace=False)
        self.means = X[random_idx]

        # Initialize covariance matrices
        self.covs = np.array([np.eye(n_features) for _ in range(self.n_components)])

    def gaussian_pdf(self, X, mean, cov):
        return multivariate_normal.pdf(X, mean=mean, cov=cov)

    def e_step(self, X):
        n_samples = X.shape[0]
        responsibilities = np.zeros((n_samples, self.n_components))

        # Calculate responsibilities
        for k in range(self.n_components):
            responsibilities[:, k] = self.weights[k] * self.gaussian_pdf(
                X, self.means[k], self.covs[k]
            )

        # Normalize responsibilities
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)
        return responsibilities

    def m_step(self, X, responsibilities):
        n_samples = X.shape[0]

        # Update weights (mixing coefficients)
        Nk = responsibilities.sum(axis=0)
        self.weights = Nk / n_samples

        # Update means
        self.means = np.dot(responsibilities.T, X) / Nk[:, np.newaxis]

        # Update covariances
        for k in range(self.n_components):
            diff = X - self.means[k]
            self.covs[k] = np.dot(responsibilities[:, k] * diff.T, diff) / Nk[k]

    def fit(self, X):
        self.initialize_parameters(X)

        log_likelihood_old = -np.inf

        for _iteration in range(self.max_iter):
            # E-step
            responsibilities = self.e_step(X)

            # M-step
            self.m_step(X, responsibilities)

            # Compute log likelihood
            log_likelihood = 0
            for k in range(self.n_components):
                log_likelihood += self.weights[k] * self.gaussian_pdf(
                    X, self.means[k], self.covs[k]
                )
            log_likelihood = np.sum(np.log(log_likelihood))

            # Check convergence
            if abs(log_likelihood - log_likelihood_old) < self.tol:
                break

            log_likelihood_old = log_likelihood

    def predict(self, X):
        responsibilities = self.e_step(X)
        return np.argmax(responsibilities, axis=1)


# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    n_samples = 300

    # Generate three clusters
    cluster1 = np.random.normal(0, 1, (n_samples // 3, 2))
    cluster2 = np.random.normal(4, 1.5, (n_samples // 3, 2))
    cluster3 = np.random.normal(-3, 1, (n_samples // 3, 2))

    X = np.vstack([cluster1, cluster2, cluster3])

    # Fit GMM
    gmm = GaussianMixture(n_components=3)
    gmm.fit(X)

    # Get cluster assignments
    labels = gmm.predict(X)
    print("Cluster centers:", gmm.means)
