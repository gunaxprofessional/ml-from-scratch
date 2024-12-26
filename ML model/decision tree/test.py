import numpy as np


class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=2, criterion='Gini'):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.criterion = criterion
        self.tree = None

    def Gini(self, y):
        gini = 1
        Classes = np.unique(y)
        for Class in Classes:
            p = np.sum(y == Class)/len(y)
            p = p**2
            gini -= p
        return gini

    def entropy(self, y):
        Classes = np.unique(y)
        entropy = 0
        for Class in Classes:
            p = np.sum(y == Class)/len(y)
            entropy -= p*np.log(p)
        return entropy

    def information_gain(self, y, y_left, y_right):
        """Calculate Information Gain for a split."""
        parent_entropy = self.entropy(y)
        left_entropy = self.entropy(y_left)
        right_entropy = self.entropy(y_right)

        n = len(y)
        n_left = len(y_left)
        n_right = len(y_right)

        weighted_entropy = (n_left / n) * left_entropy + \
            (n_right / n) * right_entropy
        return parent_entropy - weighted_entropy

    def split(self, x, y, feature, threshold):
        left = y[:, feature] <= threshold
        right = y[:, feature] > threshold

        return x[left], x[right], y[left], [right]

    def best_split(self, x, y):
        best_threshold = best_split = best_feature = best_metric = None
        metric = (float('inf') if self.criterion == 'Gini' else float('-inf'))
        n_features = x.shape[0]

        for feature in n_features:
            thresholds = np.unique(x[:feature])
            for threshold in thresholds:
                X_left, X_right, y_left, y_right = self.split(
                    x, y, feature, threshold)

                if len(y_left) < self.min_samples_split or len(y_right) < self.min_samples_split:
                    continue

                if self.criterion == 'Gini':
                    metric = (len(y_left)/len(y)) * self.Gini(y_left) + \
                        (len(y_right)/len(y)) * self.Gini(y_right)

                    if metric < best_metric:
                        best_metric = metric
                        best_threshold = threshold
                        best_feature = feature
                        best_split = (X_left, X_right, y_left, y_right)

        return best_threshold, best_feature, best_split

    def build_tree(self, x, y, depth=0):
        n_features = x.shape[0]

        classes = np.unique(y)

        if len(classes) == 1 or (dept and depth > self.max_depth) or len(y) < self.min_samples_split:
            return {'type': 'leaf',
                    'class': classes[0]}

        best_threshold, best_feature, best_split = self.best_split(x, y)

        if best_split is None:
            return {'type': 'leaf',
                    'class': np.bincount(y).argmax()}

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
