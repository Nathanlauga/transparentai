import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from ..classification import compute_metrics
from ..evaluation import evaluation

from transparentai import plots


def plot_confusion_matrix(confusion_matrix):
    """
    Show confusion matrix.

    Parameters
    ----------
    confusion_matrix: array
        confusion_matrix metrics result
    """
    n_classes = len(confusion_matrix)

    sns.heatmap(confusion_matrix,
                cmap='Blues',
                square=True,
                fmt='d',
                annot=True)
    plt.xlabel('Predicted')
    plt.ylabel('Real')
    plt.title('Confusion Matrix')


def plot_roc_curve(roc_curve, roc_auc):
    """
    Show a roc curve plot with roc_auc score on the legend.

    Parameters
    ----------
    roc_curve: array
        roc_curve metrics result for each class
    roc_auc: array
        roc_auc metrics result for each class      
    """
    fpr, tpr = dict(), dict()
    n_classes = len(roc_curve)

    for v in range(n_classes):
        fpr[v] = roc_curve[v][0]
        tpr[v] = roc_curve[v][1]
    lw = 2

    colors = sns.color_palette("colorblind", n_classes)
    for v, color in zip(range(n_classes), colors):
        n = 1 if n_classes == 1 else v
        plt.plot(fpr[v], tpr[v], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(n, roc_auc[v]))
        if n_classes <= 2:
            break

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic curve')
    plt.legend(loc="lower right")


SCORE_PLOT_FUNCTION = {
    'confusion_matrix': plot_confusion_matrix,
    'roc_auc': plot_roc_curve
}


def plot_table_score_clf(perf):
    """Insert a table of scores on a
    matplotlib graphic for a classifier

    Parameters
    ----------
    perf: dict
        Dictionnary with computed score  
    """
    perf_clf = {}
    for k, v in perf.items():
        if k in SCORE_PLOT_FUNCTION:
            continue
        perf_clf[k] = v
    return plots.plot_table_score(perf_clf)


def plot_score_function(perf, perf_prob, metric):
    """Plots score with a specific function.

    E.g. confusion_matrix or roc_auc

    Parameters
    ----------
    perf: dict
        Dictionnary with computed score  
    perf_prob: dict
        Dictionnary with computed score (using probabilities)
    metric: str
        name of the metric

    Raises
    ------
    ValueError:
        metric does not have a plot function
    """

    if metric not in SCORE_PLOT_FUNCTION:
        raise ValueError('metric does not have a plot function')

    fn = SCORE_PLOT_FUNCTION[metric]

    if metric == 'roc_auc':
        fn(perf_prob['roc_curve'], perf_prob['roc_auc'])
    elif metric in perf:
        fn(perf[metric])
    else:
        fn(perf_prob[metric])


def compute_prob_performance(y_true, y_prob, metrics):
    """Computes performance that require probabilities

    Parameters
    ----------
    y_true: array like
        True labels
    y_pred: array like
        Predicted labels
    metrics: list
        List of metrics to compute

    Returns
    -------
    dict:
        Dictionnary of metrics computed that requires
        probabilities. If no metrics need those then
        it returns None

    Raises
    ------
    TypeError:
        metrics must be a list
    """
    if type(metrics) != list:
        raise TypeError('metrics must be a list')
    if type(y_prob) in [list, pd.Series, pd.DataFrame]:
        y_prob = np.array(y_prob)

    for m in metrics:
        if not evaluation.score_function_need_prob(m):
            metrics = [m1 for m1 in metrics if m1 != m]

    if len(metrics) == 0:
        return None

    if len(y_prob.shape) > 1:
        n_classes = y_prob.shape[1]
        if n_classes == 1:
            n_classes = 2
    else:
        n_classes = 2

    perf_prob = {}
    for m in metrics:
        perf_prob[m] = list()
        for c in range(n_classes):
            # If binary classifier then default class is 1
            if n_classes == 2:
                c = 1
                pred = y_prob
            else:
                pred = y_prob[:, c]
            y_true_c = np.array(y_true == c).astype(int)
            score = compute_metrics(y_true_c, pred, [m])[m]
            perf_prob[m].append(score)

    return perf_prob


def preprocess_scores(y_pred):
    """Preprocess y_pred for plot_performance function.

    if y_pred is probabilities then y_pred become predicted class,
    y_prob is the probabilities else, y_prob is None

    Parameters
    ----------
    y_pred: array like (1D or 2D)
        if 1D array Predicted labels, 
        if 2D array probabilities (returns of a predict_proba function)

    Returns
    -------
    np.ndarray
        array with predicted labels
    np.ndarray
        array with probabilities if available else None
    int:
        number of classes
    """
    if type(y_pred) in [list, pd.Series, pd.DataFrame]:
        y_pred = np.array(y_pred)

    y_prob = y_pred
    if len(y_pred.shape) > 1:
        n_classes = y_pred.shape[1]
        y_pred = np.argmax(y_pred, axis=1)
    else:
        max, min, uniq = np.max(y_pred), np.min(y_pred), len(np.unique(y_pred))
        # If List of predicted class 
        if (max == int(max)) & (min == int(min)) & (uniq <= max+1):
            n_classes = max
        # Else : list of probabilities
        else:
            n_classes = 2

    return y_pred, y_prob, n_classes


def plot_performance(y_true, y_pred, y_true_valid=None, y_pred_valid=None, metrics=None, **kwargs):
    """Plots the performance of a classifier. 
    You can use the metrics of your choice with the metrics argument

    Can compare train and validation set.

    Parameters
    ----------
    y_true: array like
        True labels
    y_pred: array like (1D or 2D)
        if 1D array Predicted labels, 
        if 2D array probabilities (returns of a predict_proba function)
    y_true_valid: array like (default None)
        True labels
    y_pred_valid: array like (1D or 2D) (default None)
        if 1D array Predicted labels, 
        if 2D array probabilities (returns of a predict_proba function)
    metrics: list (default None)
        List of metrics to plots

    Raises
    ------
    TypeError:
        if metrics is set it must be a list
    """

    y_pred, y_prob, n_classes = preprocess_scores(y_pred)

    validation = (y_true_valid is not None) & (y_pred_valid is not None)
    if validation:
        y_pred_valid, y_prob_valid, _ = preprocess_scores(y_pred_valid)

    if metrics is None:
        metrics = ['accuracy', 'confusion_matrix', 'roc_auc', 'roc_curve']
        if n_classes <= 2:
            metrics += ['f1', 'recall', 'precision']
        else:
            metrics += ['f1_micro', 'recall_micro', 'precision_micro']

    elif type(metrics) != list:
        raise TypeError('metrics must be a list')
    elif ('roc_auc' in metrics) & ('roc_curve' not in metrics):
        metrics.append('roc_curve')
    elif ('roc_auc' not in metrics) & ('roc_curve' in metrics):
        metrics.append('roc_auc')

    metrics_plot = [m for m in metrics if m in SCORE_PLOT_FUNCTION]

    # 1. Compute scores
    perf_prob = compute_prob_performance(y_true, y_prob, metrics)
    if validation:
        perf_prob_valid = compute_prob_performance(
            y_true_valid, y_prob_valid, metrics)

    if perf_prob is not None:
        metrics = [m for m in metrics if m not in perf_prob]

    if len(metrics) == 0:
        perf = None
    else:
        perf = compute_metrics(y_true, y_pred, metrics)
        if validation:
            perf_valid = compute_metrics(y_true_valid, y_pred_valid, metrics)

    # 2. If some metrics can be plot plot them
    if len(metrics_plot) > 0:
        n_fn = len(metrics_plot)
        n_rows = n_fn if validation else n_fn + n_fn % 2
        row_size = 6

        # Init figure
        fig = plt.figure(figsize=(15, row_size*n_rows+1))
        widths = [1, 1]
        heights = [2] + [row_size]*n_rows
        gs = fig.add_gridspec(ncols=2, nrows=n_rows+1, wspace=0.7,
                              width_ratios=widths,
                              height_ratios=heights)
        r, c = 1, 0

        # Header with scores
        if not validation:
            ax = fig.add_subplot(gs[0, :])
            plot_table_score_clf(perf)
        else:
            ax = fig.add_subplot(gs[0, 0])
            plot_table_score_clf(perf)
            ax = fig.add_subplot(gs[0, 1])
            plot_table_score_clf(perf_valid)

        for m in metrics_plot:
            ax = fig.add_subplot(gs[r:r+1, c])
            plot_score_function(perf, perf_prob, m)
            c += 1

            if validation:
                ax = fig.add_subplot(gs[r:r+1, c])
                plot_score_function(perf_valid, perf_prob_valid, m)
                c += 1

            if c == 2:
                r += 1
                c = 0

        title = 'Model performance plot'
        if validation:
            title += ' train set (left) and test set (right)'
        fig.suptitle(title, fontsize=18)
        plt.show()
        return plots.plot_or_figure(fig, **kwargs)
