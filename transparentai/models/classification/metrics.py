from ..evaluation import evaluation

def compute_metrics(y_true, y_pred, metrics):
    """Computes the inputed metrics for a
    classification problem.
    
    Use metrics.compute_metrics function.
    
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
        Dictionnary with metric's name as key and 
        metric function's result as value
    """
    return evaluation.compute_metrics(y_true, y_pred, metrics, classification=True)