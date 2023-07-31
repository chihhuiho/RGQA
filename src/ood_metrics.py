from sklearn.metrics import auc, roc_curve, precision_recall_curve
from accfpr.accfpr import accfpr_curve
import numpy as np

# https://github.com/tayden/ood-metrics/blob/master/ood_metrics/metrics.py
def auroc(preds, labels):
    """Calculate and return the area under the ROC curve using unthresholded predictions on the data and a binary true label.
    
    preds: array, shape = [n_samples]
           Target scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions.
           
    labels: array, shape = [n_samples]
            True binary labels in range {0, 1} or {-1, 1}.
    """
    fpr, tpr, _ = roc_curve(labels, preds)
    return auc(fpr, tpr)


def aupr(preds, labels):
    """Calculate and return the area under the Precision Recall curve using unthresholded predictions on the data and a binary true label.
    
    preds: array, shape = [n_samples]
           Target scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions.
           
    labels: array, shape = [n_samples]
            True binary labels in range {0, 1} or {-1, 1}.
    """
    precision, recall, _ = precision_recall_curve(labels, preds)
    return auc(recall, precision)


def fpr_at_95_tpr(preds, labels):
    """Return the FPR when TPR is at minimum 95%.
        
    preds: array, shape = [n_samples]
           Target scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions.
           
    labels: array, shape = [n_samples]
            True binary labels in range {0, 1} or {-1, 1}.
    """
    fpr, tpr, _ = roc_curve(labels, preds)
    
    if all(tpr < 0.95):
        # No threshold allows TPR >= 0.95
        return 0
    elif all(tpr >= 0.95):    
        # All thresholds allow TPR >= 0.95, so find lowest possible FPR
        idxs = [i for i, x in enumerate(tpr) if x>=0.95]
        return min(map(lambda idx: fpr[idx], idxs))
    else:
        # Linear interp between values to get FPR at TPR == 0.95
        return np.interp(0.95, tpr, fpr)

def fpr_at_any_tpr(preds, labels, min_tpr=0.95):
    """Return the FPR when TPR is at minimum min_tpr.
        
    preds: array, shape = [n_samples]
           Target scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions.
           
    labels: array, shape = [n_samples]
            True binary labels in range {0, 1} or {-1, 1}.
    """
    fpr, tpr, _ = roc_curve(labels, preds)
    
    if all(tpr < min_tpr):
        # No threshold allows TPR >= 0.95
        return 0
    elif all(tpr >= min_tpr):    
        # All thresholds allow TPR >= 0.95, so find lowest possible FPR
        idxs = [i for i, x in enumerate(tpr) if x>=min_tpr]
        return min(map(lambda idx: fpr[idx], idxs))
    else:
        # Linear interp between values to get FPR at TPR == 0.95
        return np.interp(min_tpr, tpr, fpr)

def detection_error(preds, labels):
    """Return the misclassification probability when TPR is 95%.
        
    preds: array, shape = [n_samples]
           Target scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions.
           
    labels: array, shape = [n_samples]
            True binary labels in range {0, 1} or {-1, 1}.
            Negatives are assumed to be labelled as 1
    """
    fpr, tpr, _ = roc_curve(labels, preds)

    # Get ratios of positives to negatives
    neg_ratio = sum(np.array(labels) == 1) / len(labels)
    pos_ratio = 1 - neg_ratio

    # Get indexes of all TPR >= 95%
    idxs = [i for i, x in enumerate(tpr) if x>=0.95]
    
    # Calc error for a given threshold (i.e. idx)
    # Calc is the (# of negatives * FNR) + (# of positives * FPR)
    _detection_error = lambda idx: neg_ratio * (1 - tpr[idx]) + pos_ratio * fpr[idx]
    
    # Return the minimum detection error such that TPR >= 0.95
    return min(map(_detection_error, idxs))


def ood_performance(gts: np.ndarray, scores: np.ndarray):
    measures = {}
    # measures['fpr95'] = fpr_at_95_tpr(scores, gts)
    measures['fpr95'] = fpr_at_any_tpr(scores, gts)
    measures['fpr81.4'] = fpr_at_any_tpr(scores, gts, min_tpr=0.814)
    measures['fpr88.9'] = fpr_at_any_tpr(scores, gts, min_tpr=0.889)
    measures['auroc'] = auroc(scores, gts)
    measures['aupr_in'] = aupr(scores, gts)
    measures['aupr_out'] = aupr([-s for s in scores], [1 - g for g in gts])
    try:
        measures['detection_error'] = detection_error(scores, gts)
    except:
        measures['detection_error'] = np.nan
    return measures

def accfpr_metrics(scores, preds, labels, neg_label, target_acc_ratio=0.95, target_acc=None):
    """
    Return auc, fpr when acc is 95% maxmimum.

    scores: array, shape = [n_samples]
            Target scores, how much confidence to believe it is an in-domain instance.

    preds: array, shape=[n_samples]
           Prediction class of classification model.
    
    labels: array, shape=[n_samples]
            True multi-class labels, should include neg_label.
    
    neg_label: int or str,
               Negative label.
    
    target_acc_ratio: float,
                Ratio of maximum accuracy you are expecting, will threshold at
                target_acc_ratio * maxmimum accuracy
    """
    fpr, acc, _ = accfpr_curve(labels, preds, scores, neg_label)

    if target_acc is None:
        target_acc = acc[-1] * target_acc_ratio
    else:
        target_acc_ratio = round(target_acc/acc[-1],2)
    if all(acc >= target_acc):
        idxs = [i for i, x in enumerate(acc) if x>=target_acc]
        min_idx = np.argmin(list(map(lambda idx: fpr[idx], idxs)))
        # return {'auaf': auc(fpr, acc), f'fpr@{target_acc_ratio:.2f}acc': fpr[min_idx], f'{target_acc_ratio:.2f}acc': acc[min_idx]}
        return {'auaf': auc(fpr, acc), f'fpr@{target_acc_ratio:.2f}acc': fpr[min_idx], 'full_acc': acc[-1]}
    else:
        # return {'auaf': auc(fpr, acc), f'fpr@{target_acc_ratio:.2f}acc': np.interp(target_acc, acc, fpr), f'{target_acc_ratio:.2f}acc': target_acc}
        return {'auaf': auc(fpr, acc), f'fpr@{target_acc_ratio:.2f}acc': np.interp(target_acc, acc, fpr), 'full_acc': acc[-1]}