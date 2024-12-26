import numpy as np
from sklearn.svm import SVC

# Sample data (2D points)
X = np.array([[2, 3], [3, 3], [4, 6], [5, 6]])
y = np.array([1, 1, -1, -1])

# Custom SVM implementation (from the previous code)


class SVM:
    def __init__(self, C=1.0):
        self.C = C
        self.w = None
        self.b = None

    def fit(self, X, y, epochs=1000, learning_rate=0.001):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(epochs):
            for i in range(n_samples):
                if y[i] * (np.dot(X[i], self.w) + self.b) < 1:
                    self.w -= learning_rate * \
                        (2 * self.w - self.C * y[i] * X[i])
                    self.b -= learning_rate * (-self.C * y[i])
                else:
                    self.w -= learning_rate * (2 * self.w)

    def predict(self, X):
        decision_function = np.dot(X, self.w) + self.b
        return np.sign(decision_function)

    def compute_loss(self, X, y):
        n_samples = len(y)
        hinge_loss = np.maximum(0, 1 - y * (np.dot(X, self.w) + self.b))
        regularization_loss = 0.5 * np.dot(self.w, self.w)
        total_loss = (1 / n_samples) * np.sum(hinge_loss) + regularization_loss
        return total_loss


# Train Custom SVM Model
svm_custom = SVM(C=1.0)
svm_custom.fit(X, y, epochs=1000, learning_rate=0.01)
predictions_custom = svm_custom.predict(X)
loss_custom = svm_custom.compute_loss(X, y)

# Train Sklearn SVM Model
svm_sklearn = SVC(C=1.0, kernel='linear')
svm_sklearn.fit(X, y)
predictions_sklearn = svm_sklearn.predict(X)
# Sklearn SVM loss is not directly available, so we will use the decision_function to calculate the hinge loss
decision_function_sklearn = svm_sklearn.decision_function(X)
hinge_loss_sklearn = np.maximum(0, 1 - y * decision_function_sklearn)
regularization_loss_sklearn = 0.5 * \
    np.dot(svm_sklearn.coef_.flatten(), svm_sklearn.coef_.flatten())
loss_sklearn = np.mean(hinge_loss_sklearn) + regularization_loss_sklearn

# Print results
print("Custom SVM Predictions:", predictions_custom)
print("Sklearn SVM Predictions:", predictions_sklearn)

print("\nCustom SVM Loss:", loss_custom)
print("Sklearn SVM Loss:", loss_sklearn)
