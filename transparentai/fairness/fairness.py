import warnings

import pandas as pd
import numpy as np

from transparentai import utils
from ..fairness import metrics
from ..datasets import variable

__all__ = [
    'create_privilieged_df', 
    'compute_fairness_metrics', 
    'FAIRNESS_METRICS', 
    'model_bias', 
    'find_correlated_feature'
]

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
         'gender':['Male'],                # privileged group is man for gender attribute
         'age': lambda x: x > 30 & x < 55  # privileged group aged between 30 and 55 years old
     }

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

    Example
    =======

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

    Returns a dictionnary with protected attributes name's 
    as key containing a dictionnary with metric's
    name as key and metric function's result as value

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
        Dictionnary with protected attributes name's 
        as key containing a dictionnary with metric's
        name as key and metric function's result as value

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


def fairness_metrics_goal_threshold(metric):
    """Returns metric goal and threshold values.

    Parameters
    ----------
    metric: str
        The name of the metric

    Returns
    -------
    int:
        goal value
    float:
        threshold (+ and -) of the metric
    """
    metrics_goal_1 = [
        'disparate_impact'
    ]

    if metric in metrics_goal_1:
        return 1, 0.2
    elif metric == 'theil_index':
        return 0, 0.2
    return 0, 0.1


def is_metric_fair(score, metric):
    """Whether the given metric is fair or not.

    Parameters
    ----------
    score: float:
        Score value of the metric
    metric: str
        The name of the metric

    Returns
    -------
    bool:
        Whether the metric is fair or not.
    """
    goal, threshold = fairness_metrics_goal_threshold(metric)
    return np.abs(score - goal) <= threshold


def fairness_metrics_text(score, metric):
    """Returns a explanation text for the following metrics :

    - statistical_parity_difference
    - disparate_impact
    - equal_opportunity_difference
    - average_odds_difference

    returns '' if none of the above metrics.

    Parameters
    ----------
    score: float:
        Score value of the metric
    metric: str
        The name of the metric

    Returns
    -------
    str:
        Text explaining what the score means.
    """
    score = round(score, 4)

    if metric == 'statistical_parity_difference':
        g1, g2 = ('un', '') if (score) > 0 else ('', 'un')

        return 'The %sprivileged group is predicted ' % g1 + '\
with the positive output %.2f%% more often than the %sprivileged group.' % (abs(score)*100, g2)

    elif metric == 'disparate_impact':
        g1, g2 = ('un', '') if (score) > 1 else ('', 'un')
        score = np.reciprocal(score) if score < 1 else score

        return 'The %sprivileged group is predicted ' % g1 + '\
with the positive output %.2f times more often than the %sprivileged group.' % (score, g2)

    elif metric == 'equal_opportunity_difference':
        g1, g2 = ('un', '') if (score) > 0 else ('', 'un')

        return 'For a person in the %sprivileged group, ' % g1 + '\
the model predict a correct positive output %.2f%% more often than a person in the %sprivileged group.' % (abs(score)*100, g2)

    elif metric == 'average_odds_difference':
        g1, g2 = ('un', '') if (score) > 0 else ('', 'un')

        return 'For a person in the %sprivileged group, ' % g1 + '\
the model predict a correct positive output or a correct negative output %.2f%% more often ' % (abs(score)*100) + '\
than a person in the %sprivileged group.' % (g2)

    return ''


def model_bias(y_true, y_pred, df, privileged_group,
               pos_label=1, regr_split=None, returns_text=False):
    """Computes the fairness metrics for protected attributes
    refered in the privileged_group argument.

    It uses the 4 fairness function :

    - statistical_parity_difference
    - disparate_impact
    - equal_opportunity_difference
    - average_odds_difference

    You can also use it for a regression problem. You can set a value
    in the regr_split argument so it converts it to a binary classification problem.
    To use the mean use 'mean'.
    If the favorable label is more than the split value 
    set pos_label argument to 1 else to 0.

    This function is using the fairness.compute_metrics function.
    So if returns_text is False then it's the same output.

    Example
    =======

    >>> from transparentai.datasets import load_boston
    >>> from sklearn.linear_model import LinearRegression

    >>> data = load_boston()
    >>> X, y = data.drop(columns='MEDV'), data['MEDV']
    >>> regr = LinearRegression().fit(X, y)

    >>> privileged_group = {
        'AGE': lambda x: (x > 30) & (x < 55)
    }
    >>> y_true, y_pred = y, regr.predict(X)
    >>> model_bias(y_true, y_pred, data, 
                   privileged_group, regr_split='mean')
    {'AGE': {'statistical_parity_difference': -0.2041836536594836,
      'disparate_impact': 0.674582301980198,
      'equal_opportunity_difference': 0.018181818181818188,
      'average_odds_difference': -0.0884835589941973,
      'theil_index': 0.06976073748626294}}

    >>> bias_txt = model_bias(y_true, y_pred, data, 
                              privileged_group, regr_split='mean', 
                              returns_text=True)
    >>> print(bias_txt['AGE'])
    The privileged group is predicted with the positive output 20.42% more often than the unprivileged group. This is considered to be not fair.
    The privileged group is predicted with the positive output 1.48 times more often than the unprivileged group. This is considered to be not fair.
    For a person in the unprivileged group, the model predict a correct positive output 1.82% more often than a person in the privileged group. This is considered to be fair.
    For a person in the privileged group, the model predict a correct positive output or a correct negative output 8.85% more often than a person in the unprivileged group. This is considered to be fair.
    The model has 2 fair metrics over 4 (50%).


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
    pos_label: number
        The label of the positive class.
    regr_split: 'mean' or number (default None)
        If it's a regression problem then you can convert result to a
        binary classification using 'mean' or a choosen number.
        both y_true and y_pred become 0 and 1 : 0 if it's equal or less
        than the split value (the average if 'mean') and 1 if more.
        If the favorable label is more than the split value set pos_label=1
        else pos_label=0
    returns_text: bool (default False)
        Whether it return computed metrics score or a text explaination
        for the computed bias.
    Returns
    -------
    dict:
        Dictionnary with metric's name as key and 
        metric function's result as value if returns_text is False
        else it returns a text explaining the model fairness over
        the 4 metrics.
    """
    metrics = ['statistical_parity_difference',
               'disparate_impact',
               'equal_opportunity_difference',
               'average_odds_difference']

    scores = compute_fairness_metrics(y_true, y_pred, df, privileged_group,
                                      metrics, pos_label, regr_split='mean')

    if not returns_text:
        return scores

    res = {}
    for attr, bias_scores in scores.items():
        txt = list()
        n_fair = 0
        for metric, score in bias_scores.items():
            is_fair = is_metric_fair(score, metric)
            fair_text = '' if is_fair else ' not'
            fair_text = ' This is considered to be%s fair.' % (fair_text)

            txt.append(fairness_metrics_text(score, metric) + fair_text)

            if is_fair:
                n_fair += 1

        txt.append('The model has %i fair metrics over 4 (%i%%).' %
                   (n_fair, n_fair*100/4))
        res[attr] = "\n".join(txt)

    return res


def find_correlated_feature(df, privileged_group, corr_threshold=0.4):
    """Finds correlated feature with protected attribute set in the
    privileged_group argument.

    This function is a helper to find out if protected attribute
    can be found in other features.

    Returns a dictionnary with protected attributes name's 
    as key containing a dictionnary with metric's
    name as key and metric function's result as value.

    Example
    =======

    >>> from transparentai.datasets import load_adult
    >>> from transparentai import fairness

    >>> data = load_adult()

    >>> privileged_group = {
            'gender':['Male'],                
            'marital-status': lambda x: 'Married' in x,
            'race':['White']
        }

    >>> fairness.find_correlated_feature(data, privileged_group, 
                                         corr_threshold=0.4)
    {'gender': {'marital-status': 0.4593,
      'occupation': 0.4239,
      'relationship': 0.6465},
     'marital-status': {'relationship': 0.4881,
      'gender': 0.4593,
      'income': 0.4482},
     'race': {'native-country': 0.4006}}


    Parameters
    ----------
    df: pd.DataFrame
        Dataframe to extract privilieged group from.
    privileged_group: dict
        Dictionnary with protected attribute as key (e.g. age or gender)
        and a list of favorable value (like ['Male']) or a function
        returning a boolean corresponding to a privileged group
    corr_threshold: float (default 0.4)
        Threshold for which features are considered to be correlated

    Returns
    -------
    dict:
        Dictionnary with protected attributes name's 
        as key containing a dictionnary with correlated 
        features as key and correlation coeff as value
    """
    privileged_df = create_privilieged_df(df, privileged_group)
    corr_df = variable.compute_correlation(df)

    res = {}
    for attr, values in privileged_df.iteritems():
        cond = corr_df[attr].abs() > corr_threshold
        corr_feats = corr_df[attr][cond].round(4).to_dict()
        del corr_feats[attr]

        res[attr] = corr_feats

    return res
