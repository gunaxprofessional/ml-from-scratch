import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter

# Custom KNN Class


class KNearestNeighborsCustom:
    def __init__(self, k=3, metric="euclidean"):
        self.k = k
        self.metric = metric

    def _calculate_distance(self, point1, point2):
        if self.metric == "euclidean":
            return np.sqrt(np.sum((point1 - point2) ** 2))
        elif self.metric == "manhattan":
            return np.sum(np.abs(point1 - point2))
        else:
            raise ValueError("Unsupported distance metric!")

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for test_point in X_test:
            distances = [self._calculate_distance(
                test_point, train_point) for train_point in self.X_train]
            k_indices = np.argsort(distances)[:self.k]
            k_labels = self.y_train[k_indices]
            most_common = Counter(k_labels).most_common(1)
            predictions.append(most_common[0][0])
        return np.array(predictions)


# Dataset
X_train = np.array([[1, 2], [2, 3], [3, 4], [5, 6]])
y_train = np.array([0, 0, 1, 1])
X_test = np.array([[1.5, 2.5], [4, 5]])

# Custom KNN
custom_knn = KNearestNeighborsCustom(k=3, metric="euclidean")
custom_knn.fit(X_train, y_train)
custom_predictions = custom_knn.predict(X_test)
print("Custom KNN Predictions:", custom_predictions)

# Scikit-Learn KNN
sklearn_knn = KNeighborsClassifier(n_neighbors=3, metric="euclidean")
sklearn_knn.fit(X_train, y_train)
sklearn_predictions = sklearn_knn.predict(X_test)
print("Scikit-Learn KNN Predictions:", sklearn_predictions)
