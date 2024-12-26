import numpy as np


class MultipleLinearRegression:
    def __init__(self, epochs=100, learning_rate=0.01):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.coefficients = None

    def compute_loss(self, x, y):
        y_pred = np.dot(x, self.coefficients[1:]) + self.coefficients[0]
        return np.mean((y - y_pred) ** 2)

    def fit(self, x, y):
        n_samples, n_features = x.shape
        self.coefficients = np.zeros(n_features + 1)  # +1 for the intercept

        for epoch in range(self.epochs):
            # Calculate predictions
            y_pred = np.dot(x, self.coefficients[1:]) + self.coefficients[0]

            # Compute gradients
            # Partial derivatives for weights
            jw = -(2 / n_samples) * np.dot(x.T, (y - y_pred))
            # Partial derivative for bias
            jb = -(2 / n_samples) * np.sum(y - y_pred)

            # Update coefficients
            self.coefficients[1:] -= self.learning_rate * jw  # Update weights
            self.coefficients[0] -= self.learning_rate * jb   # Update bias

            # Logging loss every 100 epochs
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {self.compute_loss(x, y):.4f}")

    def predict(self, x):
        return np.dot(x, self.coefficients[1:]) + self.coefficients[0]


if __name__ == "__main__":
    # Example dataset with multiple features
    X = np.array([
        [1, 2],  # Feature 1, Feature 2
        [2, 3],
        [3, 4],
        [4, 5],
        [5, 6]
    ])
    y = np.array([5, 7, 9, 11, 13])  # Target values

    # Create and train the model
    model = MultipleLinearRegression(learning_rate=0.01, epochs=1000)
    model.fit(X, y)

    # Print results
    print("\nModel Results:")
    print(f"Intercept: {model.coefficients[0]:.4f}")
    print(f"Coefficients: {model.coefficients[1:]}")

    # Make predictions
    y_pred = model.predict(X)
    print(f"\nPredicted values: {y_pred}")
