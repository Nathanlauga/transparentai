from transparentai.models import evaluation
import numpy as np

__all__ = [
    'average_odds_difference',
    'disparate_impact',
    'equal_opportunity_difference',
    'statistical_parity_difference',
    'theil_index'
]


def preprocess_y(y, pos_label):
    """Preprocess Y prediction if it's probabilities.
    Returns a numpy array with 0 and 1, 0 if it's not the selected
    label and 1 if it is.

    Parameters
    ----------
    y: array like
        list of predicted labels
    pos_label: int
        number of the positive label

    Returns
    -------
    np.ndarray
        y array preprocessed
    """
    y = np.array(y)

    if len(y.shape) > 1:
        y = np.argmax(y, axis=1)
    else:
        y = np.round(y, 0)

    return (y == pos_label).astype(int)


def base_rate(y, prot_attr, pos_label=1, privileged=True):
    """Computes the base rate of a privileged group (or not 
    depending on the argument passed).

    Parameters
    ----------
    y: array like
        list of predicted labels
    prot_attr: array like
        Array of 0 and 1 same length as y which
        indicates if the row is member of a privileged
        group or not
    pos_label: int (default 1)
        number of the positive label
    privileged: bool (default True)
        Boolean prescribing whether to
        condition this metric on the `privileged_groups`, if `True`, or
        the `unprivileged_groups`, if `False`. Defaults to `None`
        meaning this metric is computed over the entire dataset.

    Returns
    -------
    float:
        Base rate of privileged or 
        unprivileged group

    Raises
    ------
    TypeError:
        y and prot_attr must have the same length
    """
    prot_attr = np.array(prot_attr)
    y = preprocess_y(y, pos_label)

    if len(y) != len(prot_attr):
        raise ValueError('y and prot_attr must have the same length')

    priv_cond = prot_attr == int(privileged)
    n_priv = np.sum(priv_cond)
    n_pos = np.sum(y[priv_cond] == 1)

    if n_priv > 0:
        return n_pos / n_priv
    return 1.


def model_metrics_priv(metrics_fun, *args, privileged=None):
    """Computes a metric function 
    (e.g. transparentai.evaluation.classification.true_positive_rate)
    for a priviliged or unprivileged group 

    Parameters
    ----------
    metrics_fun: function
        metrics function to compute
    privileged: bool (default None)
        Boolean prescribing whether to
        condition this metric on the `privileged_groups`, if `True`, or
        the `unprivileged_groups`, if `False`. Defaults to `None`
        meaning this metric is computed over the entire dataset.

    Returns
    -------
    float:
        Result of the function
    """
    y_true, y_pred = args[0], args[1]
    prot_attr, pos_label = args[2], args[3]

    y_true = preprocess_y(y_true, pos_label)
    y_pred = preprocess_y(y_pred, pos_label)

    if privileged is not None:
        y_true = y_true[prot_attr == int(privileged)]
        y_pred = y_pred[prot_attr == int(privileged)]

    return metrics_fun(y_true, y_pred)


def tpr_privileged(*args, privileged=True):
    """Computes true positives rate for a
    priviliged or unprivileged group 
    """
    metrics_fun = evaluation.classification.true_positive_rate
    return model_metrics_priv(metrics_fun, *args, privileged=privileged)


def fpr_privileged(*args, privileged=True):
    """Computes false positives rate for a
    priviliged or unprivileged group 
    """
    metrics_fun = evaluation.classification.false_positive_rate
    return model_metrics_priv(metrics_fun, *args, privileged=privileged)


def difference(metric_fun, *args):
    """Computes difference of the metric for 
    unprivileged and privileged groups.

    Parameters
    ----------
    metric_fun: function
        metric function that returns a number

    Returns
    -------
    float:
        Difference of a metric for 
        unprivileged and privileged groups.
    """
    return (metric_fun(*args, privileged=False)
            - metric_fun(*args, privileged=True))


def ratio(metric_fun, *args):
    """Computes ratio of the metric for 
    unprivileged and privileged groups.

    Parameters
    ----------
    metric_fun: function
        metric function that returns a number

    Returns
    -------
    float:
        Ratio of a metric for 
        unprivileged and privileged groups.
    """
    return (metric_fun(*args, privileged=False)
            / metric_fun(*args, privileged=True))


def statistical_parity_difference(y, prot_attr, pos_label=1):
    """Computes the statistical parity difference 
    for a protected attribute and a specified label

    Computed as the difference of the rate of 
    favorable outcomes received by the unprivileged group 
    to the privileged group.

    The ideal value of this metric is 0 A value < 0 implies 
    higher benefit for the privileged group and a value > 0 
    implies a higher benefit for the unprivileged group.

    Fairness for this metric is between -0.1 and 0.1

    .. math::

       Pr(\\hat{Y} = v | D = \\text{unprivileged})
       - Pr(\\hat{Y} = v | D = \\text{privileged})

    code source inspired from `aif360 statistical_parity_difference`_

    .. _aif360 statistical_parity_difference: https://aif360.readthedocs.io/en/latest/modules/generated/aif360.sklearn.metrics.statistical_parity_difference.html?highlight=Statistical%20Parity%20Difference#aif360.sklearn.metrics.statistical_parity_difference

    Parameters
    ----------
    y: array like
        list of predicted labels
    prot_attr: array like
        Array of 0 and 1 same length as y which
        indicates if the row is member of a privileged
        group or not
    pos_label: int (default 1)
        number of the positive label

    Returns
    -------
    float:
        Statistical parity difference bias metric

    Raises
    ------
    ValueError:
        y and prot_attr must have the same length
    """
    if len(y) != len(prot_attr):
        raise ValueError('y and prot_attr must have the same length')

    return difference(base_rate, y, prot_attr, pos_label)


def disparate_impact(y, prot_attr, pos_label=1):
    """Computes the Disparate impact 
    for a protected attribute and a specified label

    Computed as the ratio of rate of favorable outcome for 
    the unprivileged group to that of the privileged group.

    The ideal value of this metric is 1.0 A value < 1 implies 
    higher benefit for the privileged group and a value > 1 
    implies a higher benefit for the unprivileged group.

    Fairness for this metric is between 0.8 and 1.2

    .. math::

       \\frac{Pr(\\hat{Y} = v | D = \\text{unprivileged})}
       {Pr(\\hat{Y} = v | D = \\text{privileged})}

    code source inspired from `aif360 disparate_impact`_ 

    .. _aif360 disparate_impact: https://aif360.readthedocs.io/en/latest/modules/generated/aif360.sklearn.metrics.disparate_impact_ratio.html?highlight=Disparate%20Impact#aif360.sklearn.metrics.disparate_impact_ratio

    Parameters
    ----------
    y: array like
        list of predicted labels
    prot_attr: array like
        Array of 0 and 1 same length as y which
        indicates if the row is member of a privileged
        group or not
    pos_label: int (default 1)
        number of the positive label

    Returns
    -------
    float:
        Disparate impact bias metric

    Raises
    ------
    ValueError:
        y and prot_attr must have the same length
    """
    if len(y) != len(prot_attr):
        raise ValueError('y and prot_attr must have the same length')

    return ratio(base_rate, y, prot_attr, pos_label)


def equal_opportunity_difference(y_true, y_pred, prot_attr, pos_label=1):
    """Computes the equal opportunity difference 
    for a protected attribute and a specified label    

    This metric is computed as the difference of 
    true positive rates between the unprivileged and 
    the privileged groups. The true positive rate is 
    the ratio of true positives to the total number 
    of actual positives for a given group.

    The ideal value is 0. A value of < 0 implies higher 
    benefit for the privileged group and a value > 0 implies 
    higher benefit for the unprivileged group.

    Fairness for this metric is between -0.1 and 0.1

    :math:`TPR_{D = \\text{unprivileged}} - TPR_{D = \\text{privileged}}`

    code source from `aif360 equal_opportunity_difference`_

    .. _aif360 equal_opportunity_difference: https://aif360.readthedocs.io/en/latest/modules/generated/aif360.sklearn.metrics.equal_opportunity_difference.html?highlight=Equal%20Opportunity%20Difference

    Parameters
    ----------
    y_true: array like
        True labels
    y_pred: array like
        Predicted labels
    prot_attr: array like
        Array of 0 and 1 same length as y which
        indicates if the row is member of a privileged
        group or not
    pos_label: int (default 1)
        number of the positive label

    Returns
    -------
    float:
        Equal opportunity difference bias metric

    Raises
    ------
    ValueError:
        y_true and y_pred must have the same length
    ValueError:
        y_true and prot_attr must have the same length
    """
    if len(y_true) != len(y_pred):
        raise ValueError('y_true and y_pred must have the same length')
    if len(y_true) != len(prot_attr):
        raise ValueError('y_true and prot_attr must have the same length')

    return difference(tpr_privileged, y_true, y_pred, prot_attr, pos_label)


def average_odds_difference(y_true, y_pred, prot_attr, pos_label=1):
    """    
    Average odds difference 
    ***********************

    Computes the average odds difference 
    for a protected attribute and a specified label 

    Computed as average difference of false positive rate 
    (false positives / negatives) and true positive rate 
    (true positives / positives) between unprivileged and 
    privileged groups.

    The ideal value of this metric is 0. A value of < 0 implies
    higher benefit for the privileged group and a value > 0
    implies higher benefit for the unprivileged group.

    Fairness for this metric is between -0.1 and 0.1

    .. math::

       \\frac{1}{2}\\left[|FPR_{D = \\text{unprivileged}} - FPR_{D = \\text{privileged}}|
       + |TPR_{D = \\text{unprivileged}} - TPR_{D = \\text{privileged}}|\\right]

    A value of 0 indicates equality of odds.

    code source from `aif360 average_odds_difference`_

    .. _aif360 average_odds_difference: https://aif360.readthedocs.io/en/latest/modules/generated/aif360.sklearn.metrics.average_odds_difference.html?highlight=Average%20Odds%20Difference#aif360.sklearn.metrics.average_odds_difference

    Parameters
    ----------
    y_true: array like
        True labels
    y_pred: array like
        Predicted labels
    prot_attr: array like
        Array of 0 and 1 same length as y which
        indicates if the row is member of a privileged
        group or not
    pos_label: int (default 1)
        number of the positive label

    Returns
    -------
    float:
        Average of absolute difference bias metric

    Raises
    ------
    ValueError:
        y_true and y_pred must have the same length
    ValueError:
        y_true and prot_attr must have the same length
    """
    if len(y_true) != len(y_pred):
        raise ValueError('y_true and y_pred must have the same length')
    if len(y_true) != len(prot_attr):
        raise ValueError('y_true and prot_attr must have the same length')

    args = [y_true, y_pred, prot_attr, pos_label]
    return (1/2) * (
        difference(fpr_privileged, *args) + difference(tpr_privileged, *args)
    )


def theil_index(y_true, y_pred, prot_attr, pos_label=1):
    """Computes the theil index 
    for a protected attribute and a specified label 

    Computed as the generalized entropy of benefit 
    for all individuals in the dataset, with alpha = 1. 
    It measures the inequality in benefit allocation for individuals.

    A value of 0 implies perfect fairness.

    Fairness is indicated by lower scores, higher scores are problematic

    With :math:`b_i = \\hat{y}_i - y_i + 1`:

    .. math:: 

        \\frac{1}{n}\sum_{i=1}^n\\frac{b_{i}}{\mu}\ln\\frac{b_{i}}{\mu}

    code source from `aif360 theil_index`_

    .. _aif360 theil_index: https://aif360.readthedocs.io/en/latest/modules/generated/aif360.metrics.ClassificationMetric.html?highlight=Theil%20Index#aif360.metrics.ClassificationMetric.generalized_entropy_index

    Parameters
    ----------
    y_true: array like
        True labels
    y_pred: array like
        Predicted labels
    prot_attr: array like
        Array of 0 and 1 same length as y which
        indicates if the row is member of a privileged
        group or not
    pos_label: int (default 1)
        number of the positive label

    Returns
    -------
    float:
        Theil index bias metric

    Raises
    ------
    ValueError:
        y_true and y_pred must have the same length
    ValueError:
        y_true and prot_attr must have the same length
    """
    if len(y_true) != len(y_pred):
        raise ValueError('y_true and y_pred must have the same length')
    if len(y_true) != len(prot_attr):
        raise ValueError('y_true and prot_attr must have the same length')

    y_true = preprocess_y(y_true, pos_label)
    y_pred = preprocess_y(y_pred, pos_label)

    b = y_pred - y_true + 1

    return np.mean(np.log((b / np.mean(b))**b) / np.mean(b))
