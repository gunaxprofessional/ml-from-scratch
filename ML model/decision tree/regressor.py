import numpy as np


class DecisionTreeRegressor:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None

    def _mse(self, y):
        """Calculate Mean Squared Error."""
        if len(y) == 0:
            return 0
        mean = np.mean(y)
        return np.mean((y - mean) ** 2)

    def _best_split(self, X, y):
        """
        Find the best feature and value to split on.
        Returns: (best_feature, best_value, best_mse, left_mask, right_mask)
        """
        best_mse = float("inf")
        best_feature = None
        best_value = None
        best_split = None

        for feature_idx in range(X.shape[1]):
            feature_values = X[:, feature_idx]
            unique_values = np.unique(feature_values)

            for value in unique_values:
                left_mask = feature_values <= value
                right_mask = ~left_mask

                if len(y[left_mask]) < self.min_samples_leaf or len(y[right_mask]) < self.min_samples_leaf:
                    continue

                mse_left = self._mse(y[left_mask])
                mse_right = self._mse(y[right_mask])

                mse_split = (len(y[left_mask]) * mse_left +
                             len(y[right_mask]) * mse_right) / len(y)

                if mse_split < best_mse:
                    best_mse = mse_split
                    best_feature = feature_idx
                    best_value = value
                    best_split = (left_mask, right_mask)

        return best_feature, best_value, best_mse, best_split

    def _build_tree(self, X, y, depth=0):
        """Recursively build the tree."""
        if len(y) == 0:
            return None

        # Stopping conditions
        if (self.max_depth is not None and depth >= self.max_depth) or len(y) < self.min_samples_split:
            return {"value": np.mean(y)}

        # Find the best split
        best_feature, best_value, best_mse, best_split = self._best_split(X, y)
        if best_feature is None:
            return {"value": np.mean(y)}

        left_mask, right_mask = best_split

        return {
            "feature": best_feature,
            "value": best_value,
            "left": self._build_tree(X[left_mask], y[left_mask], depth + 1),
            "right": self._build_tree(X[right_mask], y[right_mask], depth + 1),
        }

    def fit(self, X, y):
        """Fit the model to the data."""
        self.tree = self._build_tree(X, y)

    def _predict_sample(self, sample, tree):
        """Predict for a single sample."""
        if "value" in tree:
            return tree["value"]

        if sample[tree["feature"]] <= tree["value"]:
            return self._predict_sample(sample, tree["left"])
        else:
            return self._predict_sample(sample, tree["right"])

    def predict(self, X):
        """Predict for a set of samples."""
        return np.array([self._predict_sample(sample, self.tree) for sample in X])


# Example Usage
if __name__ == "__main__":
    # Training Data (Regression)
    X_train = np.array([
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 4.0],
        [6.0, 8.0],
        [7.0, 8.0],
        [8.0, 9.0]
    ])
    y_train = np.array([1.5, 2.5, 3.5, 7.5, 8.5, 9.5])

    # Test Data
    X_test = np.array([
        [1.5, 2.5],
        [6.5, 7.5]
    ])

    # Train Decision Tree Regressor
    regressor = DecisionTreeRegressor(
        max_depth=3, min_samples_split=2, min_samples_leaf=1)
    regressor.fit(X_train, y_train)

    # Predict
    predictions = regressor.predict(X_test)
    print("Predictions:", predictions)
