import numpy as np
from sklearn import metrics


y_pred = [1, 0, 1, 1, 1]
prob = [0.9, 0.3, 0.4, 0.8, 0.6]
brier = 0


for i, j in zip(prob, y_pred):
    brier += ((i - j) * (i - j))

print("Brier Score: ", brier/len(y_pred))

print(metrics.brier_score_loss(y_prob=prob, y_true=y_pred))
