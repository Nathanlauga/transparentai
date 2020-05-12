
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from transparentai import plots
from ..monitoring import monitoring


def plot_monitoring(y_true, y_pred, timestamp=None, interval='month',
                    metrics=None, classification=False, **kwargs):
    """Plots model performance over a timestamp array which represent
    the date or timestamp of the prediction.

    If timestamp is None or interval then it just compute the metrics
    on all the predictions.

    If interval is not None it can be one of the following : 'year', 'month', 
    'day' or 'hour'. 

    - 'year' : format '%Y'
    - 'month' : format '%Y-%m'
    - 'day' : format '%Y-%m-%d'
    - 'hour' : format '%Y-%m-%d-%r'

    If it's for a classification and you're using y_pred as probabilities
    don't forget to pass the classification=True argument !

    You can use your choosing metrics. for that refer to the `evaluation metrics`_
    documentation.

    .. _evaluation metrics: #


    Parameters
    ----------
    y_true: array like
        True labels
    y_pred: array like (1D or 2D)
        if 1D array Predicted labels, 
        if 2D array probabilities (returns of a predict_proba function)
    timestamp: array like or None (default None)
        Array of datetime when the prediction occured
    interval: str or None (default 'month')
        interval to format the timestamp with
    metrics: list (default None)
        List of metrics to compute
    classification: bool (default True)
        Whether the ML task is a classification or not

    """
    scores = monitoring.monitor_model(y_true, y_pred, timestamp,
                                      interval, metrics, classification)

    scores = scores[['count'] + scores.columns.values.tolist()[:-1]]

    n_rows = int(len(scores.columns) / 2) + 1
    fig = plt.figure(figsize=(15, 5*n_rows))
    gs = fig.add_gridspec(n_rows, 2, hspace=0.3)

    dates = mdates.num2date(mdates.datestr2num(scores.index))
    fig.autofmt_xdate()

    for i, (name, score) in enumerate(scores.iteritems()):
        ax = fig.add_subplot(gs[int(i/2), i % 2])
        if name == 'count':
            plt.bar(dates, score, width=7)
        else:
            plt.plot(dates, score)

        ymin, ymax = ax.get_ylim()
        ax.set_ylim((ymin*0.9, ymax*1.1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=18)

        ax.set_title(name)

    fig.suptitle('Model performance', fontsize=16)
    # plt.show()
    return plots.plot_or_figure(fig, **kwargs)
