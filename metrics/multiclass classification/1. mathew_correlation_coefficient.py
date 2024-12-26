from sklearn import metrics
import numpy as np

y_pred = [0, 0, 0, 2, 0, 1, 0, 1, 2, 2]
y_true = [1, 0, 1, 2, 0, 1, 0, 0, 1, 2]

n = len(set(y_true))

precision = 0

TN_total = TP_total = FN_total = FP_total = 0

for classLabel in set(y_true):
    tn = tp = fp = fn = 0
    for p, t in zip(y_pred, y_true):
        if t == classLabel and p == classLabel:
            tp += 1
        elif t == classLabel and p != classLabel:
            fn += 1
        elif t != classLabel and p != classLabel:
            tn += 1
        elif t != classLabel and p == classLabel:
            fp += 1

    TN_total += tn
    TP_total += tp
    FN_total += fn
    FP_total += fp


mcc = ((TP_total * TN_total) - (FP_total * FN_total)) / np.sqrt((TP_total + FP_total)
                                                                * (TP_total + FN_total) * (TN_total + FP_total) * (TN_total + FN_total))

print("Manual Mathew Correlation Coefficient: ", mcc)
print("sklearn Mathew Correlation Coefficient: ",
      metrics.matthews_corrcoef(y_true, y_pred))
