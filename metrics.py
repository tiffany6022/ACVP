from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef, roc_auc_score


def get_metrics(y_true, y_prob):
    return {
        "acc": accuracy(y_true, y_prob),
        "auc": auc(y_true, y_prob),
        "mcc": mcc(y_true, y_prob),
        "sen": sensitivity(y_true, y_prob),
        "spec": specificity(y_true, y_prob)
    }


def accuracy(y_true, y_prob):
    return accuracy_score(y_true, (y_prob >= 0.5).astype(int))


def auc(y_true, y_prob):
    return roc_auc_score(y_true, y_prob)


def mcc(y_true, y_prob):
    return matthews_corrcoef(y_true, (y_prob >= 0.5).astype(int))


def sensitivity(y_true, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, (y_prob >= 0.5).astype(int)).ravel()
    return tp / (tp + fn)


def specificity(y_true, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, (y_prob >= 0.5).astype(int)).ravel()
    return tn / (tn + fp)
