from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
import numpy as np

def accuracies(y_true, y_pred, adaptive=True):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    auc = roc_auc_score(y_true, y_pred)

    if adaptive == True:
        fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
        dist = fpr ** 2 + (1 - tpr) ** 2
        best_thres = thresholds[np.argmin(dist)]
    else:
        best_thres = 0.5

    y_pred_val = np.where(np.array(y_pred).flatten() >= best_thres, 1, 0)
    cm = confusion_matrix(y_true, y_pred_val)
    tn, fp, fn, tp = cm.ravel()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    accuracy = (tp + tn) / (tn + fp + fn + tp)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    print('AUC score:', np.round(auc, 4))
    print('Accuracy:', np.round(accuracy, 4))
    print('Sensitivity:', np.round(sensitivity, 4))
    print('Specificity:', np.round(specificity, 4))