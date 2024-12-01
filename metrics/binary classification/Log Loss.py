import numpy as np
import math
from sklearn import metrics


y_true = [1, 0, 1, 1, 1]
prob = [0.9, 0.3, 0.4, 0.8, 0.6]
brier = 0


for p, y in zip(prob, y_true):
    brier += (y * math.log(p) + (1-y) * math.log(1-p))

n = -(1/len(prob))

print(n*brier)

print(metrics.log_loss(y_true=y_true, y_pred=prob))
