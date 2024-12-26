import numpy as np
import matplotlib.pyplot as plt


class LinearRegressionGD:
    def __init__(self, learning_rate=0.01, epochs=1000, tolerance=1e-6):
        """
        Initialize the Linear Regression model using Gradient Descent.

        Parameters:
        learning_rate (float): Learning rate for gradient descent.
        epochs (int): Number of iterations for gradient descent.
        tolerance (float): Threshold for early stopping based on loss change.
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.tolerance = tolerance
        self.slope = 0
        self.intercept = 0

    def compute_loss(self, X, y):
        """
        Compute the Mean Squared Error loss.

        Parameters:
        X (numpy array): Feature values (independent variable).
        y (numpy array): Target values (dependent variable).

        Returns:
        float: The Mean Squared Error loss.
        """
        y_pred = self.slope * X + self.intercept
        return np.mean((y - y_pred) ** 2)

    def fit(self, X, y):
        """
        Train the model using gradient descent.

        Parameters:
        X (numpy array): Feature values.
        y (numpy array): Target values.
        """
        n = len(X)  # Number of data points
        previous_loss = float('inf')

        for epoch in range(self.epochs):
            # Calculate predictions
            y_pred = self.slope * X + self.intercept

            # Compute gradients
            dm = (-2 / n) * np.sum(X * (y - y_pred))
            db = (-2 / n) * np.sum(y - y_pred)

            # Update parameters
            self.slope -= self.learning_rate * dm
            self.intercept -= self.learning_rate * db

            # Compute the current loss
            loss = self.compute_loss(X, y)

            # Print the loss every 100 epochs
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {loss:.6f}")

            # Check for convergence
            if abs(previous_loss - loss) < self.tolerance:
                print(f"Converged at epoch {epoch}")
                break

            previous_loss = loss

    def predict(self, X):
        """
        Make predictions using the trained model.

        Parameters:
        X (numpy array): Feature values.

        Returns:
        numpy array: Predicted values.
        """
        return self.slope * X + self.intercept

    def plot_results(self, X, y):
        """
        Plot the dataset and the regression line.

        Parameters:
        X (numpy array): Feature values.
        y (numpy array): Target values.
        """
        plt.scatter(X, y, color="blue", label="Data points")
        plt.plot(X, self.predict(X), color="red", label="Regression line")
        plt.xlabel("X")
        plt.ylabel("y")
        plt.legend()
        plt.title("Linear Regression")
        plt.show()


# Example usage:
if __name__ == "__main__":
    # Example dataset
    X = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 5, 4, 5])

    # Initialize and train the model
    model = LinearRegressionGD(learning_rate=0.01, epochs=1000, tolerance=1e-6)
    model.fit(X, y)

    # Print results
    print(f"Slope (m): {model.slope}")
    print(f"Intercept (b): {model.intercept}")

    # Make predictions
    y_pred = model.predict(X)
    print(f"Predicted values: {y_pred}")

    # Plot the results
    model.plot_results(X, y)
