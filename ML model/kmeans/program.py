import numpy as np
from sklearn.cluster import KMeans

# Generate a small dataset
data = np.array([
    [1, 1], [1, 4], [1, 3],
    [0, 2], [1, 1], [10, 6]
])

# K-Means implementation with custom distance metrics


class KMeansNumpy:
    def __init__(self, n_clusters=2, max_iters=10, tol=1e-4, distance_metric="euclidean"):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.distance_metric = distance_metric
        self.centroids = None
        self.labels = None

    def _compute_distance(self, data, centroid):
        if self.distance_metric == "euclidean":
            return np.sqrt(np.sum((data - centroid) ** 2, axis=1))
        elif self.distance_metric == "manhattan":
            return np.sum(np.abs(data - centroid), axis=1)
        else:
            raise ValueError(
                f"Unsupported distance metric: {self.distance_metric}")

    def fit(self, data):
        # Initialize centroids randomly
        self.centroids = data[np.random.choice(
            data.shape[0], self.n_clusters, replace=False)]
        for _ in range(self.max_iters):
            # Assign clusters
            distances = np.zeros((data.shape[0], self.n_clusters))
            for i, centroid in enumerate(self.centroids):
                distances[:, i] = self._compute_distance(data, centroid)

            self.labels = np.argmin(distances, axis=1)

            # Recompute centroids
            new_centroids = np.zeros((self.n_clusters, data.shape[1]))

            for i in range(self.n_clusters):
                cluster_points = data[self.labels == i]
                new_centroids[i] = cluster_points.mean(axis=0)

            if np.all(np.abs(new_centroids - self.centroids) < self.tol):
                break
            self.centroids = new_centroids
        return self

    def predict(self, data):
        # Assign clusters for new data
        distances = np.zeros((data.shape[0], self.n_clusters))
        for i, centroid in enumerate(self.centroids):
            distances[:, i] = self._compute_distance(data, centroid)
        return np.argmin(distances, axis=1)


# Run K-Means with NumPy
kmeans_numpy = KMeansNumpy(n_clusters=2, distance_metric="manhattan")
kmeans_numpy.fit(data)

# Run K-Means with scikit-learn
kmeans_sklearn = KMeans(n_clusters=2, random_state=42)
kmeans_sklearn.fit(data)

# Prediction on new data
new_data = np.array([[0, 0], [12, 3]])
numpy_predictions = kmeans_numpy.predict(new_data)
sklearn_predictions = kmeans_sklearn.predict(new_data)

# Print results
print("NumPy Centroids:\n", kmeans_numpy.centroids)
print("\nScikit-learn Centroids:\n", kmeans_sklearn.cluster_centers_)
print("\nNumPy Predictions for new data:\n", numpy_predictions)
print("\nScikit-learn Predictions for new data:\n", sklearn_predictions)
