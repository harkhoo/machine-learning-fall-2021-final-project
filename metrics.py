from sklearn.metrics import confusion_matrix

def metrics(lbl, pred):
    tn, fp, fn, tp = confusion_matrix(lbl, pred)
    accuracy = (tn + tp) / (tn + tp + fn + fp)
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = (2 * precision * recall) / (precision + recall)
    return accuracy, specificity, sensitivity, precision, recall, f1_score