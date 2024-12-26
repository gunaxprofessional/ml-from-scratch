import numpy as np


class DecisionTreeRegressor:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def mse(self, y):
        """Calculate Mean Squared Error for a split."""
        if len(y) == 0:
            return 0
        mean_y = np.mean(y)
        return np.mean((y - mean_y) ** 2)

    def split(self, X, y, feature, threshold):
        """Split the dataset based on a feature and threshold."""
        left_idx = X[:, feature] <= threshold
        right_idx = X[:, feature] > threshold
        return X[left_idx], X[right_idx], y[left_idx], y[right_idx]

    def best_split(self, X, y):
        """Find the best feature and threshold for splitting."""
        n_samples, n_features = X.shape
        best_metric = float("inf")
        best_feature = None
        best_threshold = None
        best_splits = None

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                X_left, X_right, y_left, y_right = self.split(
                    X, y, feature, threshold
                )
                if len(y_left) < self.min_samples_split or len(y_right) < self.min_samples_split:
                    continue

                metric = (len(y_left) / len(y)) * self.mse(y_left) + \
                         (len(y_right) / len(y)) * self.mse(y_right)

                if metric < best_metric:
                    best_metric = metric
                    best_feature = feature
                    best_threshold = threshold
                    best_splits = (X_left, X_right, y_left, y_right)

        return best_feature, best_threshold, best_splits

    def build_tree(self, X, y, depth=0):
        """Recursively build the decision tree."""
        n_samples, n_features = X.shape

        if len(y) < self.min_samples_split or (self.max_depth and depth >= self.max_depth):
            # Leaf node
            return {"type": "leaf", "value": np.mean(y)}

        feature, threshold, splits = self.best_split(X, y)
        if not splits:
            # If no split is found
            return {"type": "leaf", "value": np.mean(y)}

        X_left, X_right, y_left, y_right = splits
        left_child = self.build_tree(X_left, y_left, depth + 1)
        right_child = self.build_tree(X_right, y_right, depth + 1)

        return {
            "type": "node",
            "feature": feature,
            "threshold": threshold,
            "left": left_child,
            "right": right_child,
        }

    def fit(self, X, y):
        """Fit the decision tree to the dataset."""
        self.tree = self.build_tree(X, y)

    def predict_one(self, x, tree):
        """Predict the value for a single sample."""
        if tree["type"] == "leaf":
            return tree["value"]
        feature = tree["feature"]
        threshold = tree["threshold"]
        if x[feature] <= threshold:
            return self.predict_one(x, tree["left"])
        else:
            return self.predict_one(x, tree["right"])

    def predict(self, X):
        """Predict the value for each sample in the dataset."""
        predictions = []
        for x in X:
            predictions.append(self.predict_one(x, self.tree))
        return np.array(predictions)


# Example Usage
if __name__ == "__main__":
    # Training Dataset
    X_train = np.array([[1.0, 2.1],
                        [2.0, 1.1],
                        [1.3, 3.0],
                        [3.0, 3.1],
                        [2.5, 0.5]])
    y_train = np.array([10.0, 20.0, 15.0, 25.0, 30.0])

    # Test Dataset
    X_test = np.array([[1.5, 2.0],  # Expected output ~ 10
                       [2.5, 1.0],  # Expected output ~ 25
                       [3.0, 3.0]])  # Expected output ~ 25

    # Train Decision Tree Regressor
    tree_regressor = DecisionTreeRegressor(max_depth=3, min_samples_split=2)
    tree_regressor.fit(X_train, y_train)
    preds = tree_regressor.predict(X_test)
    print("Predictions:", preds)
