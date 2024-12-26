import numpy as np


class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Training loop
        for i in range(self.num_iterations):
            # Forward pass
            z = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(z)

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if i % 200 == 0:
                print(f"Iteration {i}, Cost: {self.compute_cost(y, y_pred)}")

    def compute_cost(self, y, y_pred):
        # Binary cross-entropy cost
        n_samples = y.shape[0]
        cost = -(1 / n_samples) * np.sum(y * np.log(y_pred + 1e-15) +
                                         (1 - y) * np.log(1 - y_pred + 1e-15))
        return cost

    def predict_proba(self, X):
        # Probability predictions
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)

    def predict(self, X, threshold=0.5):
        # Binary predictions based on threshold
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)


# Create tiny dataset (4 records)
X = np.array([
    [2, 3],    # Class 1 (positive)
    [1, 2],    # Class 1 (positive)
    [-1, 0],   # Class 0 (negative)
    [-2, -1]   # Class 0 (negative)
])

y = np.array([1, 1, 0, 0])

# Train model
model = LogisticRegression(learning_rate=0.1, num_iterations=1000)
model.fit(X, y)

# Show final results
print("\nFinal Parameters:")
print("Weights:", model.weights.round(3))
print("Bias:", round(model.bias, 3))

# Show predictions for each point
final_probabilities = model.predict_proba(X)
final_predictions = model.predict(X)

print("\nFinal Predictions for each point:")
for i, (point, prob, pred) in enumerate(zip(X, final_probabilities, final_predictions)):
    print(
        f"Point {point}: Probability = {prob.round(3)}, Predicted = {pred}, True label = {y[i]}")
