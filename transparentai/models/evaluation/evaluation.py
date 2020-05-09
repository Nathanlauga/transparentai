
from .classification import *
from .regression import *
import warnings

from transparentai import utils

__all__ = [
    'compute_metrics'
]

# Inspired from https://scikit-learn.org/stable/modules/model_evaluation.html
EVALUATION_METRICS = {
    # CLASSIFICATION METRICS
    'accuracy': accuracy,
    'balanced_accuracy': balanced_accuracy,
    'average_precision': average_precision,
    'brier_score': brier_score,
    'f1': f1,
    'f1_micro': f1_micro,
    'f1_macro': f1_macro,
    'f1_weighted': f1_weighted,
    'f1_samples': f1_samples,
    'log_loss': log_loss,
    'precision': precision,
    'precision_micro': precision_micro,
    'recall': recall,
    'recall_micro': recall_micro,
    'TPR': true_positive_rate,
    'true_positive_rate': true_positive_rate,
    'sensitivity': true_positive_rate,
    'FPR': false_positive_rate,
    'false_positive_rate': false_positive_rate,
    'jaccard': jaccard,
    'matthews_corrcoef': matthews_corrcoef,
    'roc_curve': roc_curve,
    'roc_auc': roc_auc,
    'roc_auc_ovr': roc_auc_ovr,
    'roc_auc_ovo': roc_auc_ovo,
    'roc_auc_ovr_weighted': roc_auc_ovr_weighted,
    'roc_auc_ovo_weighted': roc_auc_ovo_weighted,
    'true_positives': true_positives,
    'TP': true_positives,
    'false_positives': false_positives,
    'FP': false_positives,
    'false_negatives': false_negatives,
    'FN': false_negatives,
    'true_negatives': true_negatives,
    'TN': true_negatives,
    'confusion_matrix': confusion_matrix,

    # REGRESSION METRICS
    'max_error': max_error,
    'mean_absolute_error': mean_absolute_error,
    'MAE': mean_absolute_error,
    'mean_squared_error': mean_squared_error,
    'MSE': mean_squared_error,
    'root_mean_squared_error': root_mean_squared_error,
    'RMSE': root_mean_squared_error,
    'mean_squared_log_error': mean_squared_log_error,
    'median_absolute_error': median_absolute_error,
    'r2': r2,
    'mean_poisson_deviance': mean_poisson_deviance,
    'mean_gamma_deviance': mean_gamma_deviance
}


def score_function_need_prob(fn):
    """
    """
    return 'y_prob' in EVALUATION_METRICS[fn].__code__.co_varnames


def compute_metrics(y_true, y_pred, metrics, classification=True):
    """Computes the inputed metrics.

    metrics can have str or function. If it's a string
    then it has to be a key from METRICS global variable dict.

    Returns a dictionnary with metric's name as key and 
    metric function's result as value

    Parameters
    ----------
    y_true: array like
        True labels
    y_pred: array like
        Predicted labels
    metrics: list
        List of metrics to compute
    classification: bool (default True)
        Whether the ML task is a classification or not

    Returns
    -------
    dict:
        Dictionnary with metric's name as key and 
        metric function's result as value

    Raises
    ------
    TypeError:
        metrics must be a list
    """
    if type(metrics) != list:
        raise TypeError('metrics must be a list')

    if type(y_true) == list:
        y_true = np.array(y_true)
    if type(y_pred) == list:
        y_pred = np.array(y_pred)

    metrics = utils.preprocess_metrics(input_metrics=metrics,
                                       metrics_dict=EVALUATION_METRICS)

    if classification:
        y_prob = y_pred

        if len(y_pred.shape) > 1:
            n_classes = y_pred.shape[1]
            y_pred = np.argmax(y_pred, axis=1)
        else:
            y_pred = np.round(y_pred, 0)

    res = {}
    for name, fn in metrics.items():
        need_prob = 'y_prob' in fn.__code__.co_varnames

        if need_prob & classification:
            if (len(y_prob.shape) == 1) | ('_ov' in name):
                res[name] = fn(y_true, y_prob)
            elif len(y_prob.shape) == 2:
                res[name] = fn(y_true, y_prob[:, 1])
            else:
                res[name] = list()
                for c in range(n_classes):
                    res[name].append(fn(y_true, y_prob[:, c]))
        else:
            res[name] = fn(y_true, y_pred)

    return res
