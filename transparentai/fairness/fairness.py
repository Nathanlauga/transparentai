import warnings

import pandas as pd
import numpy as np

from transparentai import utils
from ..fairness import metrics


FAIRNESS_METRICS = {
    'statistical_parity_difference': metrics.statistical_parity_difference,
    'disparate_impact': metrics.disparate_impact,
    'equal_opportunity_difference': metrics.equal_opportunity_difference,
    'average_odds_difference': metrics.average_odds_difference,
    'theil_index': metrics.theil_index
}


def create_privilieged_df(df, privileged_group):
    """Returns a formated dataframe with protected attribute columns
    and whether the row is privileged (1) or not (0).

    example of a privileged_group dictionnary :
    >>> privileged_group = {
    >>>     'gender':['Male'],                # privileged group is man for gender attribute
    >>>     'age': lambda x: x > 30 & x < 55  # privileged group aged between 30 and 55 years old
    >>> }

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe to extract privilieged group from.
    privileged_group: dict
        Dictionnary with protected attribute as key (e.g. age or gender)
        and a list of favorable value (like ['Male']) or a function
        returning a boolean corresponding to a privileged group

    Returns
    -------
    pd.DataFrame
        DataFrame with protected attribute columns
        and whether the row is privileged (1) or not (0)

    Raises
    ------
    TypeError:
        df is not a pandas.DataFrame
    TypeError:
        privileged_group is not a dictionnary
    ValueError:
        privileged_group has not valid keys (in df columns)
    """

    if type(df) != pd.DataFrame:
        raise TypeError('df is not a pandas.DataFrame')
    if type(privileged_group) != dict:
        raise TypeError('privileged_group is not a dictionnary')
    for k in privileged_group:
        if k not in df.columns:
            warnings.warn('%s variable is not in df columns' % k)
    if all([k not in df.columns for k in privileged_group]):
        raise ValueError('privileged_group has not valid keys (in df columns)')

    res = list()
    for k, e in privileged_group.items():
        if k not in df.columns:
            continue
        if type(e) == list:
            tmp = df[k].isin(e)
        else:
            tmp = df[k].apply(e)
        tmp = tmp.astype(int).values
        res.append(pd.Series(tmp, name=k))

        del tmp

    return pd.concat(res, axis=1)


def compute_fairness_metrics(y_true, y_pred, df, privileged_group,
                             metrics=None, pos_label=1, regr_split=None):
    """Computes the fairness metrics for one attribute

    metrics can have str or function. If it's a string
    then it has to be a key from FAIRNESS_METRICS global variable dict.
    By default it uses the 5 fairness function :

    - statistical_parity_difference
    - disparate_impact
    - equal_opportunity_difference
    - average_odds_difference
    - theil_index

    You can also use it for a regression problem. You can set a value
    in the regr_split argument so it converts it to a binary classification problem.
    To use the mean use 'mean'.
    If the favorable label is more than the split value 
    set pos_label argument to 1 else to 0.

    Example :
    >>> from transparentai.datasets import load_boston
    >>> from sklearn.linear_model import LinearRegression

    >>> data = load_boston()
    >>> X, y = data.drop(columns='MEDV'), data['MEDV']
    >>> regr = LinearRegression().fit(X, y)

    >>> privileged_group = {
        'AGE': lambda x: (x > 30) & (x < 55)
    }
    >>> y_true, y_pred = y, regr.predict(X)
    >>> compute_fairness_metrics(y_true, y_pred, data,
                                 privileged_group, regr_split='mean')
    {'AGE': {'statistical_parity_difference': -0.2041836536594836,
      'disparate_impact': 0.674582301980198,
      'equal_opportunity_difference': 0.018181818181818188,
      'average_odds_difference': -0.0884835589941973,
      'theil_index': 0.06976073748626294}}

    Returns a dictionnary with metric's name as key and 
    metric function's result as value

    Parameters
    ----------
    y_true: array like
        True labels
    y_pred: array like
        Predicted labels
    df: pd.DataFrame
        Dataframe to extract privilieged group from.
    privileged_group: dict
        Dictionnary with protected attribute as key (e.g. age or gender)
        and a list of favorable value (like ['Male']) or a function
        returning a boolean corresponding to a privileged group
    metrics: list (default None)
        List of metrics to compute, if None then it
        uses the 5 default Fairness function
    pos_label: number
        The label of the positive class.
    regr_split: 'mean' or number (default None)
        If it's a regression problem then you can convert result to a
        binary classification using 'mean' or a choosen number.
        both y_true and y_pred become 0 and 1 : 0 if it's equal or less
        than the split value (the average if 'mean') and 1 if more.
        If the favorable label is more than the split value set pos_label=1
        else pos_label=0
    Returns
    -------
    dict:
        Dictionnary with metric's name as key and 
        metric function's result as value

    Raises
    ------
    ValueError:
        y_true and y_pred must have the same length
    ValueError:
        y_true and df must have the same length
    TypeError:
        metrics must be a list
    """

    if len(y_true) != len(y_pred):
        raise ValueError('y_true and y_pred must have the same length')
    if len(y_true) != len(df):
        raise ValueError('y_true and df must have the same length')
    if metrics is None:
        metrics = [
            'statistical_parity_difference',
            'disparate_impact',
            'equal_opportunity_difference',
            'average_odds_difference',
            'theil_index',
        ]
    elif type(metrics) != list:
        raise TypeError('metrics must be a list')

    privileged_df = create_privilieged_df(df, privileged_group)

    if type(privileged_df) == pd.Series:
        prot_attr = prot_attr.to_frame()

    if regr_split is not None:
        if regr_split == 'mean':
            split_val = np.mean(y_true)
        elif type(regr_split) not in [int, float]:
            raise ValueError('regr_split has to be \'mean\' or a scalar value')
        else:
            split_val = regr_split

        y_true = (y_true > split_val).astype(int)
        y_pred = (y_pred > split_val).astype(int)

    if type(y_true) == list:
        y_true = np.array(y_true)
    if type(y_pred) == list:
        y_pred = np.array(y_pred)

    metrics = utils.preprocess_metrics(input_metrics=metrics,
                                       metrics_dict=FAIRNESS_METRICS)
    res = {}
    args = []
    for name, fn in metrics.items():
        need_both = 'y_true' in fn.__code__.co_varnames

        for attr, values in privileged_df.iteritems():
            if attr not in res:
                res[attr] = {}
            if need_both:
                res[attr][name] = fn(y_true, y_pred, values, pos_label)
            else:
                res[attr][name] = fn(y_pred, values, pos_label)

    return res
