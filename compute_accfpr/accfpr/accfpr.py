"""
Implementation of area under acc-fpr curve. An acc-fpr curve is designed for selective classification task
where the testing data might contain some proportion of out-of-domain/unanswerable instances. Acc is measured
on all in-domain examples, and those rejected will be viewed as wrong classification. When the threshold
tau decreases, more instances are accepted and therefore fpr increases, and meawhile acc will also increase.
"""
import numpy as np
import warnings

from sklearn.utils import check_consistent_length, column_or_1d, assert_all_finite
from sklearn.utils.validation import _check_sample_weight
from sklearn.utils.extmath import stable_cumsum
from sklearn.exceptions import UndefinedMetricWarning

# https://github.com/scikit-learn/scikit-learn/blob/36958fb240fbe435673a9e3c52e769f01f36bec0/sklearn/metrics/_ranking.py#L703
def _multi_clf_curve(y_true, y_pred, y_score, neg_label, sample_weight=None):
    """Calculate true and false positives per binary classification threshold.
    Parameters
    ----------
    y_true : ndarray of shape (n_samples,)
        True targets of classification.
    y_pred: ndarray of shape (n_samples,)
        Predictions from the model.
    y_score : ndarray of shape (n_samples,)
        Estimated probabilities or output of a decision function.
    neg_label : int or str
        The label of the negative class, need to be specified.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
    Returns
    -------
    fps : ndarray of shape (n_thresholds,)
        A count of false positives, at index i being the number of negative
        samples assigned a score >= thresholds[i]. The total number of
        negative samples is equal to fps[-1] (thus true negatives are given by
        fps[-1] - fps).
    tps : ndarray of shape (n_thresholds,)
        An increasing count of true positives, at index i being the number
        of positive samples assigned a score >= thresholds[i]. The total
        number of positive samples is equal to tps[-1] (thus false negatives
        are given by tps[-1] - tps).
    thresholds : ndarray of shape (n_thresholds,)
        Decreasing score values.
    """
    check_consistent_length(y_true, y_pred, y_score, sample_weight)
    y_true = column_or_1d(y_true)
    y_pred = column_or_1d(y_pred)
    y_score = column_or_1d(y_score)
    assert_all_finite(y_true)
    assert_all_finite(y_pred)
    assert_all_finite(y_score)

    # Filter out zero-weighted samples, as they should not impact the result
    if sample_weight is not None:
        sample_weight = column_or_1d(sample_weight)
        sample_weight = _check_sample_weight(sample_weight, y_true)
        nonzero_weight_mask = sample_weight != 0
        y_true = y_true[nonzero_weight_mask]
        y_pred = y_pred[nonzero_weight_mask]
        y_score = y_score[nonzero_weight_mask]
        sample_weight = sample_weight[nonzero_weight_mask]
    
    assert neg_label is not None, print("You need to specify a negative label!")

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    y_pred = y_pred[desc_score_indices]
    if sample_weight is not None:
        weight = sample_weight[desc_score_indices]
    else:
        weight = 1.0
    
    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    accs = \
        stable_cumsum(np.logical_and(y_true != neg_label, y_true == y_pred) * weight)[threshold_idxs]
    fps = stable_cumsum((y_true == neg_label) * weight)[threshold_idxs]
    return fps, accs, y_score[threshold_idxs]

# https://github.com/scikit-learn/scikit-learn/blob/36958fb240fbe435673a9e3c52e769f01f36bec0/sklearn/metrics/_ranking.py#L892
def accfpr_curve(
    y_true, y_pred, y_score, neg_label, *, sample_weight=None, drop_intermediate=True
):
    """Compute accfpr curve.
    Parameters
    ----------
    y_true : ndarray of shape (n_samples,)
        True binary labels. If labels are not either {-1, 1} or {0, 1}, then
        pos_label should be explicitly given.
    y_score : ndarray of shape (n_samples,)
        Target scores, can either be probability estimates of the positive
        class, confidence values, or non-thresholded measure of decisions
        (as returned by "decision_function" on some classifiers).
    neg_label : int or str
        The label of the negative class.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
    drop_intermediate : bool, default=True
        Whether to drop some suboptimal thresholds which would not appear
        on a plotted ROC curve. This is useful in order to create lighter
        ROC curves.
    Returns
    -------
    acc : ndarray of shape (>2,)
        Increasing false positive rates such that element i is the false
        positive rate of predictions with score >= `thresholds[i]`.
    tpr : ndarray of shape (>2,)
        Increasing true positive rates such that element `i` is the true
        positive rate of predictions with score >= `thresholds[i]`.
    thresholds : ndarray of shape = (n_thresholds,)
        Decreasing thresholds on the decision function used to compute
        fpr and tpr. `thresholds[0]` represents no instances being predicted
        and is arbitrarily set to `max(y_score) + 1`.
    """
    fps, accs, thresholds = _multi_clf_curve(
        y_true, y_pred, y_score, neg_label, sample_weight=sample_weight
    )

    # Attempt to drop thresholds corresponding to points in between and
    # collinear with other points. These are always suboptimal and do not
    # appear on a plotted ROC curve (and thus do not affect the AUC).
    # Here np.diff(_, 2) is used as a "second derivative" to tell if there
    # is a corner at the point. Both fps and tps must be tested to handle
    # thresholds with multiple data points (which are combined in
    # _binary_clf_curve). This keeps all cases where the point should be kept,
    # but does not drop more complicated cases like fps = [1, 3, 7],
    # tps = [1, 2, 4]; there is no harm in keeping too many thresholds.
    if drop_intermediate and len(fps) > 2:
        optimal_idxs = np.where(
            np.r_[True, np.logical_or(np.diff(fps, 2), np.diff(accs, 2)), True]
        )[0]
        fps = fps[optimal_idxs]
        accs = accs[optimal_idxs]
        thresholds = thresholds[optimal_idxs]
    
    # Add an extra threshold position
    # to make sure that the curve starts at (0, 0)
    accs = np.r_[0, accs]
    fps = np.r_[0, fps]
    thresholds = np.r_[thresholds[0] + 1, thresholds]

    if fps[-1] <= 0:
        warnings.warn(
            "No negative samples in y_true, false positive value should be meaningless",
            UndefinedMetricWarning,
        )
        fpr = np.repeat(np.nan, fps.shape)
    else:
        fpr = fps / fps[-1]
    
    num_pos = (y_true != neg_label).sum()
    if num_pos <= 0:
        warnings.warn(
            "No positive samples in y_true, accuracy should be meaningless",
            UndefinedMetricWarning,
        )
        acc = np.repeat(np.nan, accs.shape)
    else:
        acc = accs / num_pos
    
    return fpr, acc, thresholds