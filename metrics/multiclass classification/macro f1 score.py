from sklearn import metrics

y_pred = [0, 0, 0, 2, 0, 1, 0, 1, 2, 2]
y_true = [1, 0, 1, 2, 0, 1, 0, 0, 1, 2]

n = len(set(y_true))

precision = recall = f1 = 0


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

    precision = (tp/(tp+fp)) if (tp+fp) > 0 else 0
    recall = (tp/(tp+fn)) if (tp+fn) > 0 else 0
    f1 += (2 * (precision*recall)/(precision +
                                   recall)) if precision+recall > 0 else 0
f1 /= n

print("manual f1 score ", f1)

print("sklearn f1 score ", metrics.f1_score(
    y_true=y_true, y_pred=y_pred, average='macro'))
