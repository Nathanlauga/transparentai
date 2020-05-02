
from .classification import *
from .regression import *
import warnings

# Inspired from https://scikit-learn.org/stable/modules/model_evaluation.html
METRICS = {
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
    'recall': recall,
    'jaccard': jaccard,
    'matthews_corrcoef': matthews_corrcoef,
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

def preprocess_metrics(input_metrics):
    """Preprocess the inputed metrics so that it maps
    with the appropriate function in METRICS global variable.

    input_metrics can have str or function. If it's a string
    then it has to be a key from METRICS global variable dict

    Returns a dictionnary with metric's name as key and 
    metric function as value

    Parameters
    ----------
    input_metrics: list
        List of metrics to compute

    Returns
    -------
    dict:
        Dictionnary with metric's name as key and 
        metric function as value

    Raises
    ------
    TypeError:
        input_metrics must be a list
    """
    if type(input_metrics) != list:
        raise TypeError('input_metrics must be a list')

    fn_dict = {}
    cnt_custom = 1

    for fn in input_metrics:
        if type(fn) == str:
            if fn in METRICS:
                fn_dict[fn] = METRICS[fn]
            else:
                warnings.warn('%s function not found' % fn)
        else:
            fn_dict['custom_'+str(cnt_custom)] = fn
            cnt_custom += 1

    if len(fn_dict.keys()) == 0:
        raise ValueError('No valid metrics found')

    return fn_dict


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

    metrics = preprocess_metrics(input_metrics=metrics)

    if classification:
        y_prob = y_pred
        y_pred = np.round(y_pred, 0)

    res = {}
    for name, fn in metrics.items():
        need_prob = 'y_prob' in fn.__code__.co_varnames

        if need_prob & classification:
            res[name] = fn(y_true, y_prob)
        else:
            res[name] = fn(y_true, y_pred)

    return res
