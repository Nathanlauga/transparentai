import numpy as np
import pandas as pd

from transparentai import utils
from transparentai.models import evaluation

def compute_metrics_groupby(y_true, y_pred, groupby, metrics, classification):
    """Computes metrics groupby an array.
     
    Parameters
    ----------
    y_true: array like
        True labels
    y_pred: array like (1D or 2D)
        if 1D array Predicted labels, 
        if 2D array probabilities (returns of a predict_proba function)
    groupby: array like
        Array of values to groupby the computed metrics by
    metrics: list
        List of metrics to compute
    classification: bool
        Whether the ML task is a classification or not
    
    Returns
    -------
    pd.DataFrame:
        DataFrame with groubpy values as indexes and 
        computed metrics as columns
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    res = list()
    for group in np.unique(groupby):
        cond = groupby == group
        y_true_group = y_true[cond]
        y_pred_group = y_pred[cond]
        score = evaluation.compute_metrics(y_true_group, y_pred_group, 
                                              metrics, classification)
        score['count'] = np.sum(cond)
        res.append(score)
        
    res = pd.DataFrame(res, index=np.unique(groupby))
    return res.sort_index()
    

def monitor_model(y_true, y_pred, timestamp=None, interval='month', 
                   metrics=None, classification=False):
    """Monitor model over a timestamp array which represent
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
        
    Returns
    -------
    pd.DataFrame:
        DataFrame with datetime interval as indexes and 
        computed metrics as columns

    Raises
    ------
    ValueError:
        interval must be 'year', 'month', 'day' or 'hour'
    TypeError:
        y_true must be an array like
    TypeError:
        timestamp must be an array like
    """
    if interval not in ['year','month','day','hour',None]:
        raise ValueError("interval must be 'year', 'month', 'day' or 'hour'")
    if not utils.is_array_like(y_true):
        raise TypeError('y_true must be an array like')
    if timestamp is not None:
        if not utils.is_array_like(timestamp):
            raise TypeError('timestamp must be an array like')
    
    if (interval is None) | (timestamp is None):
        timestamp = np.repeat(['no date'], len(y_true))
    elif interval == 'year':
        timestamp = timestamp.dt.year.astype(str)
    elif interval == 'month':
        timestamp = timestamp.dt.strftime('%Y-%m')
    elif interval == 'day':
        timestamp = timestamp.dt.strftime('%Y-%m-%d')
    else:
        timestamp = timestamp.dt.strftime('%Y-%m-%d-%r')
        
    if metrics is None:
        if (classification):
            n_classes = 2 if len(y_pred.shape) < 2 else y_pred.shape[1]
            if n_classes > 2:
                metrics = ['accuracy', 'f1_micro', 'precision_micro', 'recall_micro']
            else:
                metrics = ['accuracy', 'f1', 'precision', 'recall']
        else:
            metrics = ['MAE', 'mean_squared_error', 'root_mean_squared_error', 'r2']
        
    
    scores = compute_metrics_groupby(y_true, y_pred, timestamp, 
                                     metrics, classification)
    
    return scores
