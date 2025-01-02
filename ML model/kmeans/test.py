import numpy as np


class kmeans_numpy:

    def predict(self, data):
        self.centroids = data[np.random.choice(
            data.shape[0], self.n_clusters, replace=False)]

        for _ in range(self.max_iters):
            distances = np.zeros((data.shape[0], self.n_clusters))
            for i, centroid in enumerate(self.centroids):
                distances[:, i] = self._compute_distance(data, centroid)

            self.labels = np.argmin(distances, axis=1)

            new_centroids = np.zeros((data.shape[0], self.n_clusters))

            for i in range(self.cluster):
                datax = data[self.labels == i]
                new_centroids[:, i] = data.mean(axis=0)

            self.centroids = new_centroids
