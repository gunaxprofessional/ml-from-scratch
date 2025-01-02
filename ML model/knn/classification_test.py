import numpy as np


def predict(self, X_test):
    distance = np.array(self._compute_distance())
