import sklearn.metrics
import numpy as np

# Evaluation function for classifiers


def accuracy(y_true, y_pred, **args):
    """Accuracy score based on the `sklearn.metrics.accuracy_score`_ function.

    More details here : `Accuracy score`_

    .. _Accuracy score: https://scikit-learn.org/stable/modules/model_evaluation.html#accuracy-score
    .. _sklearn.metrics.accuracy_score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
    """
    return sklearn.metrics.accuracy_score(y_true, y_pred, **args)


def balanced_accuracy(y_true, y_pred, **args):
    """Balanced accuracy score based on the `sklearn.metrics.balanced_accuracy_score`_ function.

    More details here : `Balanced accuracy score`_

    .. _sklearn.metrics.balanced_accuracy_score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html
    .. _Balanced accuracy score: https://scikit-learn.org/stable/modules/model_evaluation.html#balanced-accuracy-score
    """
    return sklearn.metrics.balanced_accuracy_score(y_true, y_pred, **args)


def average_precision(y_true, y_prob, **args):
    """Average prevision score based on the `sklearn.metrics.average_precision_score`_ function.

    More details here : `Precision, recall and F-measures`_

    .. _sklearn.metrics.average_precision_score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html#sklearn.metrics.average_precision_score
    .. _Precision, recall and F-measures: https://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall-f-measure-metrics
    """
    return sklearn.metrics.average_precision_score(y_true, y_prob, **args)


def brier_score(y_true, y_prob, **args):
    """Brier score based on the `sklearn.metrics.brier_score_loss`_ function.

    More details here : `Probability calibration`_

    .. _sklearn.metrics.brier_score_loss: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html#sklearn.metrics.brier_score_loss
    .. _Probability calibration: https://scikit-learn.org/stable/modules/calibration.html#calibration
    """
    return sklearn.metrics.brier_score_loss(y_true, y_prob, **args)


def f1(y_true, y_pred, **args):
    """F1 score based on the `sklearn.metrics.f1_score`_ function.

    More details here : `Precision, recall and F-measures`_

    .. _sklearn.metrics.f1_score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    .. _Precision, recall and F-measures: https://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall-f-measure-metrics
    """
    return sklearn.metrics.f1_score(y_true, y_pred, **args)


def f1_micro(y_true, y_pred, **args):
    """F1 score based on the `sklearn.metrics.f1_score`_ function.

    Average argument set to 'micro'.

    More details here : `Precision, recall and F-measures`_

    .. _sklearn.metrics.f1_score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    .. _Precision, recall and F-measures: https://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall-f-measure-metrics
    """
    return sklearn.metrics.f1_score(y_true, y_pred, average='micro', **args)


def f1_macro(y_true, y_pred, **args):
    """F1 score based on the `sklearn.metrics.f1_score`_ function.

    Average argument set to 'macro'.

    More details here : `Precision, recall and F-measures`_

    .. _sklearn.metrics.f1_score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    .. _Precision, recall and F-measures: https://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall-f-measure-metrics
    """
    return sklearn.metrics.f1_score(y_true, y_pred, average='macro', **args)


def f1_weighted(y_true, y_pred, **args):
    """F1 score based on the `sklearn.metrics.f1_score`_ function.

    Average argument set to 'weighted'.

    More details here : `Precision, recall and F-measures`_

    .. _sklearn.metrics.f1_score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    .. _Precision, recall and F-measures: https://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall-f-measure-metrics
    """
    return sklearn.metrics.f1_score(y_true, y_pred, average='weighted', **args)


def f1_samples(y_true, y_pred, **args):
    """F1 score based on the `sklearn.metrics.f1_score`_ function.

    Average argument set to 'samples'.

    More details here : `Precision, recall and F-measures`_

    .. _sklearn.metrics.f1_score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    .. _Precision, recall and F-measures: https://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall-f-measure-metrics
    """
    return sklearn.metrics.f1_score(y_true, y_pred, average='samples', **args)


def log_loss(y_true, y_prob, **args):
    """Log loss based on the `sklearn.metrics.log_loss`_ function.

    More details here : `Log loss`_

    .. _sklearn.metrics.log_loss: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html#sklearn.metrics.log_loss
    .. _Log loss: https://scikit-learn.org/stable/modules/model_evaluation.html#log-loss
    """
    return sklearn.metrics.log_loss(y_true, y_prob, **args)


def precision(y_true, y_pred, **args):
    """Precision score based on the `sklearn.metrics.precision_score`_ function.

    More details here : `Precision, recall and F-measures`_

    .. _sklearn.metrics.precision_score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html#sklearn.metrics.precision_score
    .. _Precision, recall and F-measures: https://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall-f-measure-metrics
    """
    return sklearn.metrics.precision_score(y_true, y_pred, **args)


def precision_micro(y_true, y_pred, **args):
    """Precision score based on the `sklearn.metrics.precision_score`_ function.

    Average argument set to 'micro'.

    More details here : `Precision, recall and F-measures`_

    .. _sklearn.metrics.precision_score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html#sklearn.metrics.precision_score
    .. _Precision, recall and F-measures: https://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall-f-measure-metrics
    """
    return sklearn.metrics.precision_score(y_true, y_pred, average='micro', **args)


def recall(y_true, y_pred, **args):
    """Recall score based on the `sklearn.metrics.recall_score`_ function.

    More details here : `Precision, recall and F-measures`_

    .. _sklearn.metrics.recall_score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html#sklearn.metrics.recall_score
    .. _Precision, recall and F-measures: https://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall-f-measure-metrics
    """
    return sklearn.metrics.recall_score(y_true, y_pred, **args)


def recall_micro(y_true, y_pred, **args):
    """Recall score based on the `sklearn.metrics.recall_score`_ function.

    Average argument set to 'micro'.

    More details here : `Precision, recall and F-measures`_

    .. _sklearn.metrics.recall_score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html#sklearn.metrics.recall_score
    .. _Precision, recall and F-measures: https://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall-f-measure-metrics
    """
    return sklearn.metrics.recall_score(y_true, y_pred, average='micro', **args)


def jaccard(y_true, y_pred, **args):
    """Jaccard score based on the `sklearn.metrics.jaccard_score`_ function.

    More details here : `Jaccard similarity coefficient score`_

    .. _sklearn.metrics.jaccard_score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.jaccard_score.html#sklearn.metrics.jaccard_score
    .. _Jaccard similarity coefficient score: https://scikit-learn.org/stable/modules/model_evaluation.html#jaccard-similarity-score
    """
    return sklearn.metrics.jaccard_score(y_true, y_pred, **args)


def matthews_corrcoef(y_true, y_pred, **args):
    """Matthews correlation coefficient based on the `sklearn.metrics.matthews_corrcoef`_ function.

    More details here : `Matthews correlation coefficient`_

    .. _sklearn.metrics.matthews_corrcoef: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html#sklearn.metrics.matthews_corrcoef
    .. _Matthews correlation coefficient: https://scikit-learn.org/stable/modules/model_evaluation.html#matthews-corrcoef
    """
    return sklearn.metrics.matthews_corrcoef(y_true, y_pred, **args)


def confusion_matrix(y_true, y_pred, **args):
    """Confusion matrix based on the `sklearn.metrics.confusion_matrix`_ function.

    More details here : `Confusion matrix`_

    .. _sklearn.metrics.confusion_matrix: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html#sklearn.metrics.confusion_matrix
    .. _Confusion matrix: https://scikit-learn.org/stable/modules/model_evaluation.html#confusion-matrix
    """
    return sklearn.metrics.confusion_matrix(y_true, y_pred, **args)


def roc_curve(y_true, y_prob, **args):
    """Area Under the Receiver Operating Characteristic Curve (ROC AUC)
    score based on the `sklearn.metrics.roc_auc_score`_ function.

    More details here : `Receiver operating characteristic (ROC)`_

    .. _sklearn.metrics.roc_auc_score: https://scikit-learn.org/0.15/modules/generated/sklearn.metrics.roc_curve.html#sklearn.metrics.roc_curve
    .. _Receiver operating characteristic (ROC): https://scikit-learn.org/stable/modules/model_evaluation.html#roc-metrics
    """
    return sklearn.metrics.roc_curve(y_true, y_prob, **args)


def roc_auc(y_true, y_prob, **args):
    """Area Under the Receiver Operating Characteristic Curve (ROC AUC)
    score based on the `sklearn.metrics.roc_auc_score`_ function.

    More details here : `Receiver operating characteristic (ROC)`_

    .. _Receiver operating characteristic (ROC): https://scikit-learn.org/stable/modules/model_evaluation.html#roc-metrics
    """
    return sklearn.metrics.roc_auc_score(y_true, y_prob, **args)


def roc_auc_ovr(y_true, y_prob, **args):
    """Area Under the Receiver Operating Characteristic Curve (ROC AUC)
    score based on the `sklearn.metrics.roc_auc_score`_ function.

    multi_class argument is set to 'ovr'.

    More details here : `Receiver operating characteristic (ROC)`_

    .. _Receiver operating characteristic (ROC): https://scikit-learn.org/stable/modules/model_evaluation.html#roc-metrics
    """
    return sklearn.metrics.roc_auc_score(y_true, y_prob, multi_class='ovr', **args)


def roc_auc_ovo(y_true, y_prob, **args):
    """Area Under the Receiver Operating Characteristic Curve (ROC AUC)
    score based on the `sklearn.metrics.roc_auc_score`_ function.

    multi_class argument is set to 'ovo'.

    More details here : `Receiver operating characteristic (ROC)`_

    .. _Receiver operating characteristic (ROC): https://scikit-learn.org/stable/modules/model_evaluation.html#roc-metrics
    """
    return sklearn.metrics.roc_auc_score(y_true, y_prob, multi_class='ovo', **args)


def roc_auc_ovr_weighted(y_true, y_prob, **args):
    """Area Under the Receiver Operating Characteristic Curve (ROC AUC)
    score based on the `sklearn.metrics.roc_auc_score`_ function.

    Average argument set to 'weighted' and multi_class to 'ovr'.

    More details here : `Receiver operating characteristic (ROC)`_

    .. _Receiver operating characteristic (ROC): https://scikit-learn.org/stable/modules/model_evaluation.html#roc-metrics
    """
    return sklearn.metrics.roc_auc_score(y_true, y_prob,
                                         average='weighted',
                                         multi_class='ovr', **args)


def roc_auc_ovo_weighted(y_true, y_prob, **args):
    """Area Under the Receiver Operating Characteristic Curve (ROC AUC)
    score based on the `sklearn.metrics.roc_auc_score`_ function.

    Average argument set to 'weighted' and multi_class to 'ovo'.

    More details here : `Receiver operating characteristic (ROC)`_

    .. _Receiver operating characteristic (ROC): https://scikit-learn.org/stable/modules/model_evaluation.html#roc-metrics
    """
    return sklearn.metrics.roc_auc_score(y_true, y_prob,
                                         average='weighted',
                                         multi_class='ovo', **args)


def true_positives(y_true, y_pred, pos_label=1):
    """Returns the number of true positives given a class number.

    .. math::

        TP = \sum_{i}^{n} (y_i = 1) \& (\hat{y}_i = 1)

    Parameters
    ----------
    y_true: array like
        True labels
    y_pred: array like
        Predicted labels
    pos_label: int (default 1)
        Label class number (if binary classification then it's 1)

    Returns
    -------
    int:
        Number of true positives
    """
    if type(y_true) == list:
        y_true = np.array(y_true)
    if type(y_pred) == list:
        y_pred = np.array(y_pred)

    return np.sum((y_true == pos_label) & (y_pred == pos_label))


def false_positives(y_true, y_pred, pos_label=1):
    """Returns the number of false positives given a class number.

    .. math::

        FP = \sum_{i}^{n} (y_i \\ne 1) \& (\hat{y}_i = 1)

    Parameters
    ----------
    y_true: array like
        True labels
    y_pred: array like
        Predicted labels
    pos_label: int (default 1)
        Label class number (if binary classification then it's 1)

    Returns
    -------
    int:
        Number of false positives
    """
    if type(y_true) == list:
        y_true = np.array(y_true)
    if type(y_pred) == list:
        y_pred = np.array(y_pred)

    return np.sum((y_true != pos_label) & (y_pred == pos_label))


def false_negatives(y_true, y_pred, pos_label=1):
    """Returns the number of false negatives given a class number.

    .. math::

        FN = \\sum_{i}^n (y_i = 1) \& (\hat{y}_i \\ne 1)

    Parameters
    ----------
    y_true: array like
        True labels
    y_pred: array like
        Predicted labels
    pos_label: int (default 1)
        Label class number (if binary classification then it's 1)

    Returns
    -------
    int:
        Number of false negatives
    """
    if type(y_true) == list:
        y_true = np.array(y_true)
    if type(y_pred) == list:
        y_pred = np.array(y_pred)

    return np.sum((y_true == pos_label) & (y_pred != pos_label))


def true_negatives(y_true, y_pred, pos_label=1):
    """Returns the number of true negatives given a class number.

    .. math::

        TN = \sum_{i}^{n} (y_i \\ne 1) \& (\hat{y}_i \\ne 1)

    Parameters
    ----------
    y_true: array like
        True labels
    y_pred: array like
        Predicted labels
    pos_label: int (default 1)
        Label class number (if binary classification then it's 1)

    Returns
    -------
    int:
        Number of true negatives
    """
    if type(y_true) == list:
        y_true = np.array(y_true)
    if type(y_pred) == list:
        y_pred = np.array(y_pred)

    return np.sum((y_true != pos_label) & (y_pred != pos_label))

def true_positive_rate(y_true, y_pred, pos_label=1):
    """
    """
    TP = true_positives(y_true, y_pred, pos_label)
    FN = false_negatives(y_true, y_pred, pos_label)
    if TP + FN > 0:
        return TP / (TP + FN)
    else:
        return 1.

def false_positive_rate(y_true, y_pred, pos_label=1):
    """
    """
    FP = false_positives(y_true, y_pred, pos_label)
    TN = true_negatives(y_true, y_pred, pos_label)
    if FP + TN > 0:
        return FP / (FP + TN)
    else:
        return 1.