import numpy as np
from sklearn.naive_bayes import GaussianNB

# Dummy dataset
X = np.array([[1.5, 2.3], [1.1, 1.9], [2.0, 2.5], [2.2, 2.8],
              [5.0, 7.0], [5.5, 6.5], [6.0, 7.5], [5.8, 7.2]])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1])  # Labels

# Custom Naive Bayes Implementation


class NaiveBayes:
    def __init__(self):
        self.class_priors = {}
        self.feature_likelihoods = {}
        self.classes = None

    def fit(self, X, y):
        """Train the Naive Bayes classifier"""
        self.classes = np.unique(y)
        n_samples, n_features = X.shape

        # Compute prior probabilities
        for c in self.classes:
            class_samples = X[y == c]
            self.class_priors[c] = len(class_samples) / n_samples

            # Compute likelihood for each feature
            self.feature_likelihoods[c] = {}  # Ensure class entry exists
            for feature_idx in range(n_features):
                feature_values = class_samples[:, feature_idx]
                print(feature_values)
                mean = np.mean(feature_values)
                var = np.var(feature_values)
                self.feature_likelihoods[c][feature_idx] = (mean, var)

    def _gaussian_pdf(self, x, mean, var):
        """Compute Gaussian Probability Density Function"""
        eps = 1e-6  # Avoid division by zero
        coeff = 1 / np.sqrt(2 * np.pi * var + eps)
        exponent = np.exp(-(x - mean)**2 / (2 * var + eps))
        return coeff * exponent

    def predict(self, X):
        """Make predictions for the input data"""
        predictions = []
        for sample in X:
            class_posteriors = {}
            for c in self.classes:
                # Log to prevent underflow
                prior = np.log(self.class_priors[c])
                likelihood = 0
                for feature_idx, x in enumerate(sample):
                    mean, var = self.feature_likelihoods[c][feature_idx]
                    likelihood += np.log(self._gaussian_pdf(x, mean, var))

                # Posterior = log(prior) + log(likelihood)
                class_posteriors[c] = prior + likelihood

            # Predict class with highest posterior
            predictions.append(max(class_posteriors, key=class_posteriors.get))
        return np.array(predictions)


# Custom Implementation
custom_model = NaiveBayes()
custom_model.fit(X, y)
custom_predictions = custom_model.predict(X)
print("Custom Naive Bayes Predictions:", custom_predictions)

# Scikit-learn Gaussian Naive Bayes
sklearn_model = GaussianNB()
sklearn_model.fit(X, y)
sklearn_predictions = sklearn_model.predict(X)
print("Sklearn Naive Bayes Predictions:", sklearn_predictions)

# Check if the results match
print("Predictions Match:", np.array_equal(
    custom_predictions, sklearn_predictions))
