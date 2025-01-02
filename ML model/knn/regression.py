import numpy as np
from sklearn.neighbors import KNeighborsRegressor


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
            k_values = self.y_train[k_indices]
            predictions.append(np.mean(k_values))
        return np.array(predictions)


# Example Dataset (Regression)
X_train = np.array([[1], [2], [3], [4], [5]])
y_train = np.array([2.0, 4.0, 6.0, 8.0, 10.0])  # A simple linear relationship
X_test = np.array([[1.5], [3.5]])

# Custom KNN for Regression
knn_regressor = KNearestNeighborsCustom(k=2, metric="euclidean")
knn_regressor.fit(X_train, y_train)
custom_regression_predictions = knn_regressor.predict(X_test)
print("Custom KNN Regression Predictions:", custom_regression_predictions)

# Scikit-Learn KNN for Regression
sklearn_knn_regressor = KNeighborsRegressor(n_neighbors=2, metric="euclidean")
sklearn_knn_regressor.fit(X_train, y_train)
sklearn_regression_predictions = sklearn_knn_regressor.predict(X_test)
print("Scikit-Learn KNN Regression Predictions:", sklearn_regression_predictions)
