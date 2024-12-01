
import math
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


print(((TP * TN) - (FP * FN)) / math.sqrt((TP + FP)
      * (TP + FN) * (TN + FP) * (TN + FN)))

print(metrics.matthews_corrcoef(y_true=y_true, y_pred=y_pred))
