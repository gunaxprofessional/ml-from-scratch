import numpy as np


class MultipleLinearRegression:
    def __init__(self, epochs=100, learning_rate=0.01):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.coefficients = 0

    def compute_loss(self, x, y):
        y_pred = np.dot(x, self.coefficients[1:]) + self.coefficients[0]
        return np.mean((y-y_pred)**2)

    def fit(self, x, y):
        n_features = x.shape[1]
        self.coefficients = np.zeros(n_features+1)
        for epoch in range(self.epochs):
            y_pred = np.dot(x, self.coefficients[1:]) + self.coefficients[0]
            jw = -(2/len(x))*(np.dot(x.T, (y-y_pred)))
            jb = -(2/len(x))*(np.sum(y-y_pred))

            self.coefficients[1:] = self.coefficients[1:] - \
                self.learning_rate * jw

            self.coefficients[0] = self.coefficients[0] - \
                self.learning_rate * jb

            if epoch % 100 == 0:
                print(f'{epoch} loss {self.compute_loss(x,y)}')

    def predict(self, X):
        """
        Make predictions using the trained model.

        Parameters:
        X (numpy array): Feature matrix

        Returns:
        numpy array: Predicted values
        """
        return np.dot(X, self.coefficients[1:]) + self.coefficients[0]


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
    model = MultipleLinearRegression(
        learning_rate=0.01, epochs=1000)
    model.fit(X, y)

    # Print results
    print("\nModel Results:")
    print(f"Intercept: {model.coefficients[0]:.4f}")
    print(f"Coefficients: {model.coefficients[1:]}")

    # Make predictions
    y_pred = model.predict(X)
    print(f"\nPredicted values: {y_pred}")
