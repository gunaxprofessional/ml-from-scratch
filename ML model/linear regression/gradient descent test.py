import numpy as np


class LinearReg:
    def __init__(self, learning_rate=0.001, epochs=100):
        self.learning_rate = learning_rate
        self.slope = 0
        self.intercept = 0
        self.epochs = epochs

    def compute_loss(self, x, y):
        y_pred = self.slope * x + self.intercept
        return np.mean((y-y_pred) ** 2)

    def fit(self, x, y):
        n = len(x)
        for epoch in range(self.epochs):
            y_pred = self.slope*x + self.intercept

            jm = -(2/n)*np.sum(x*(y-y_pred))
            jb = -(2/n)*np.sum((y-y_pred))

            self.slope = self.slope - (self.learning_rate*jm)
            self.intercept = self.intercept - (self.learning_rate*jb)

            if epoch % 100 == 0:
                print(f'{epoch} loss {self.compute_loss(x,y)}')

    def predict(self, X):
        """
        Make predictions using the trained model.

        Parameters:
        X (numpy array): Feature values.

        Returns:
        numpy array: Predicted values.
        """
        return self.slope * X + self.intercept


if __name__ == "__main__":
    # Example dataset
    X = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 5, 4, 5])

    # Initialize and train the model
    model = LinearReg(learning_rate=0.01, epochs=1000)
    model.fit(X, y)

    # Print results
    print(f"Slope (m): {model.slope}")
    print(f"Intercept (b): {model.intercept}")

    # Make predictions
    y_pred = model.predict(X)
    print(f"Predicted values: {y_pred}")
