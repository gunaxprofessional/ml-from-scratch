import numpy as np
from sklearn import metrics


y_true = [1, 0, 1, 1, 1]
y_pred = [0, 1, 1, 0, 0]


TN = TP = FN = FP = 0


for true, pred in zip(y_true, y_pred):
    if true == 1 and pred == 1:
        TP += 1
    elif true == 0 and pred == 0:
        TN += 1
    elif true == 0 and pred == 1:
        FP += 1
    elif true == 1 and pred == 0:
        FN += 1


print("manual specificity score", (TN) / (TN + FP))
