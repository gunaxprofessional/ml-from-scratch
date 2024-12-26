import numpy as np
import matplotlib.pyplot as plt


def least_squares(X, y):
    """
    Perform simple linear regression using the least squares method.

    Parameters:
    X (numpy array): Feature values (independent variable).
    y (numpy array): Target values (dependent variable).

    Returns:
    tuple: Slope (m) and intercept (b).
    """
    n = len(X)

    # Calculate the required sums
    sum_x = np.sum(X)
    sum_y = np.sum(y)
    sum_x_squared = np.sum(X ** 2)
    sum_x_y = np.sum(X * y)

    # Calculate the slope (m)
    m = (n * sum_x_y - sum_x * sum_y) / (n * sum_x_squared - sum_x ** 2)

    # Calculate the intercept (b)
    b = (sum_y - m * sum_x) / n

    return m, b


def fit(X, y):
    """
    Fit the linear regression model using least squares.

    Parameters:
    X (numpy array): Feature values (independent variable).
    y (numpy array): Target values (dependent variable).

    Returns:
    tuple: Slope (m) and intercept (b) of the fitted model.
    """
    return least_squares(X, y)


def predict(X, m, b):
    """
    Make predictions using the linear regression model.

    Parameters:
    X (numpy array): Feature values.
    m (float): Slope of the line.
    b (float): Intercept of the line.

    Returns:
    numpy array: Predicted values.
    """
    return m * X + b


# Example usage:
if __name__ == "__main__":
    # Example dataset
    X = np.array([1, 2, 3, 4, 5])  # Feature values
    y = np.array([2, 4, 5, 4, 5])  # Target values

    # Fit the model to the data
    m, b = fit(X, y)

    # Print the results
    print(f"Slope (m): {m}")
    print(f"Intercept (b): {b}")

    # Make predictions using the model
    y_pred = predict(X, m, b)
    print(f"Predicted values: {y_pred}")

    # Plot the results
    plt.scatter(X, y, color="blue", label="Data points")
    plt.plot(X, y_pred, color="red", label="Regression line")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.title("Simple Linear Regression using Least Squares")
    plt.show()
