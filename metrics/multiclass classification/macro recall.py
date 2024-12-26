from sklearn import metrics

y_pred = [0, 0, 0, 2, 0, 1, 0, 1, 2, 2]
y_true = [1, 0, 1, 2, 0, 1, 0, 0, 1, 2]

n = len(set(y_true))

recall = 0


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

    recall += (tp/(tp+fn)) if (tp+fn) > 0 else 0


print("manual recall score ", recall/n)

print("sklearn recall score ", metrics.recall_score(
    y_true=y_true, y_pred=y_pred, average='macro'))
