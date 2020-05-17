import matplotlib.pyplot as plt
import numpy as np

from ..regression import metrics as regression
from transparentai import plots


def plot_error_distribution(errors):
    """Plots the error distribution with standard deviation,
    mean and median.

    The error is calculated by the following formula :

    .. math::

        error = y - \hat{y}

    Parameters
    ----------
    errors: array like
        Errors of a regressor   
    """
    mean   = np.mean(errors)
    median = np.median(errors)
    std    = np.std(errors)

    y, x, _ = plt.hist(errors, bins=50, color='#3498db',
                       label='std = %.4f' % std)

    plt.vlines(mean, ymin=0, ymax=y.max(), color='#e74c3c', linewidths=3,
               label='mean = %.4f' % mean)
    plt.vlines(median, ymin=0, ymax=y.max(), color='#e67e22', linewidths=3,
               label='median = %.4f' % median)

    plt.legend()
    plt.title('Error distribution (bins=50)')


def plot_performance(y_true, y_pred, y_true_valid=None, y_pred_valid=None, metrics=None, **kwargs):
    """Plots the performance of a regressor. 
    You can use the metrics of your choice with the metrics argument

    Can compare train and validation set.

    Parameters
    ----------
    y_true: array like
        True target values
    y_pred: array like
        Predicted values
    y_true_valid: array like (default None)
        True target values for validation set
    y_pred_valid: array like (1D or 2D) (default None)
        Predicted  values for validation set
    metrics: list
        List of metrics to plots

    Raises
    ------
    TypeError:
        if metrics is set it must be a list
    """

    validation = (y_true_valid is not None) & (y_pred_valid is not None)

    if metrics is None:
        metrics = ['MAE', 'mean_squared_error',
                   'root_mean_squared_error', 'r2']
    elif type(metrics) != list:
        raise TypeError('metrics must be a list')

    # 1. compute metrics
    perf = regression.compute_metrics(y_true, y_pred, metrics)
    errors = y_true - y_pred

    if validation:
        perf_valid = regression.compute_metrics(
            y_true_valid, y_pred_valid, metrics)
        errors_valid = y_true_valid - y_pred_valid

    # 2. Plot figure with score values and error difference
    # Init figure
    fig = plt.figure(figsize=(15, 8))
    n_cols = int(validation)+1

    widths = [1] * n_cols
    heights = [1, 2]
    gs = fig.add_gridspec(ncols=n_cols, nrows=2, wspace=0.2,
                          width_ratios=widths,
                          height_ratios=heights)

    ax = fig.add_subplot(gs[0, 0])
    plots.plot_table_score(perf)
    ax = fig.add_subplot(gs[1, 0])
    plot_error_distribution(errors)

    if validation:
        ax = fig.add_subplot(gs[0, 1])
        plots.plot_table_score(perf_valid)
        ax = fig.add_subplot(gs[1, 1])
        plot_error_distribution(errors_valid)

    title = 'Model performance plot'
    if validation:
        title += ' train set (left) and test set (right)'
    fig.suptitle(title, fontsize=18)

    # plt.show()
    return plots.plot_or_figure(fig, **kwargs)