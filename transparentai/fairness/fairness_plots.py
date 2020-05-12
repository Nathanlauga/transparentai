import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import textwrap

from ..fairness import fairness
from transparentai import plots


def get_protected_attr_values(attr, df, privileged_group, privileged=True):
    """Retrieves all values given the privileged_group argument.

    If privileged is True and privileged_group[attr] is a list then it returns
    the list, if it's a function then values of df[attr] for which the function
    returns True. 

    If privileged is False and privileged_group[attr] is a list then it returns
    values of df[attr] not in the list else if it's a function returns values 
    of df[attr] for which the function returns False. 

    Parameters
    ----------
    attr: str
        Protected attribute which is a key of the privileged_group
        dictionnary
    df: pd.DataFrame
        Dataframe to extract privilieged group from.
    privileged_group: dict
        Dictionnary with protected attribute as key (e.g. age or gender)
        and a list of favorable value (like ['Male']) or a function
        returning a boolean corresponding to a privileged group
    privileged: bool (default True)
        Boolean prescribing whether to
        condition this metric on the `privileged_groups`, if `True`, or
        the `unprivileged_groups`, if `False`.

    Returns
    -------
    list:
        List of privileged values of the protected attribyte attr 
        if privileged is True else unprivileged values

    Raises
    ------
    ValueError:
        attr must be in privileged_group
    """
    if attr not in privileged_group:
        raise ValueError('attr must be in privileged_group')

    val = privileged_group[attr]
    if type(val) == list:
        if privileged:
            return val

        def fn(x): return x in val
    else:
        fn = val

    cond = df[attr].apply(fn) == privileged
    return df[cond][attr].unique().astype(str).tolist()


def format_priv_text(values, max_char):
    """Formats privileged (or unprivileged) values text
    so that it can be shown.

    Parameters
    ----------
    values: list
        List of privileged or unprivileged values
    max_char: int
        Maximum characters allow in the returned string

    Returns
    -------
    str:
        Formated string for given values

    Raises
    ------
    TypeError
        values must be a list
    """
    if type(values) != list:
        raise TypeError('values must be a list')

    priv_text = ''

    for val in values:
        if (len(val) + len(priv_text) > max_char) & (priv_text != ''):
            priv_text = priv_text[:-2] + ' and others  '
            break
        priv_text += val+', '

    return priv_text[:-2]


def plot_attr_title(ax, attr, df, privileged_group):
    """Plots the protected attribute titles with :

    - The attribute name (e.g. Gender)
    - Priviliged and unprivileged values
    - Number of privileged and unprivileged values

    Parameters
    ----------
    ax: plt.axes.Axes
        axe where to add the plot
    attr: str
        Protected attribute which is a key of the privileged_group
        dictionnary
    df: pd.DataFrame
        Dataframe to extract privilieged group from.
    privileged_group: dict
        Dictionnary with protected attribute as key (e.g. age or gender)
        and a list of favorable value (like ['Male']) or a function
        returning a boolean corresponding to a privileged group

    Raises
    ------
    ValueError:
        attr must be in df columns
    ValueError:
        attr must be in privileged_group keys
    """
    if attr not in df.columns:
        raise ValueError('attr must be in df columns')
    if attr not in privileged_group:
        raise ValueError('attr must be in privileged_group keys')

    plt.text(0, 1.4, 'Protected Attribute: %s' %
             attr, fontsize=22, weight="bold")
    priv_df = fairness.create_privilieged_df(df, privileged_group)[attr]

    priv_values = get_protected_attr_values(attr, df, privileged_group)
    unpriv_values = get_protected_attr_values(
        attr, df, privileged_group, privileged=False)

    priv_text = format_priv_text(priv_values, max_char=30)
    unpriv_text = format_priv_text(unpriv_values, max_char=30)

    n_priv = (priv_df == 1).sum()
    n_unpriv = (priv_df == 0).sum()
    n = len(priv_df)

    plt.text(0, 0.8, 'Privileged group values  : %s' %
             (priv_text), fontsize=13)
    plt.text(0, 0.2, 'Unrivileged group values: %s' %
             (unpriv_text), fontsize=13)

    plt.text(1, 0.8, r'# of privileged values : $\bf{%i}$ ($\bf{%.2f}$%%)' % (n_priv, n_priv*100/n),
             fontsize=13, horizontalalignment='right')
    plt.text(1, 0.2, r'# of unprivileged values : $\bf{%i}$ ($\bf{%.2f}$%%)' % (n_unpriv, n_unpriv*100/n),
             fontsize=13, horizontalalignment='right')

    plt.axis('off')
    ax.axhline(0, color='#000', linewidth=5)


def plot_bias_one_attr(ax, metric, score):
    """Plots bias metric score bar with the indication
    if it's considered not fair or fair.

    Parameters
    ----------
    ax: plt.axes.Axes
        axe where to add the plot
    metric: str
        The name of the metric
    score: float:
        Score value of the metric        
    """
    goal, threshold = fairness.fairness_metrics_goal_threshold(metric)
    is_fair = fairness.is_metric_fair(score, metric)

    ax.set_ylim((goal-1, goal+1))
    ax.set_xlim((0, 1))
    ax.axhline(goal, color='#000', linewidth=2)
    ax.axhline(goal-threshold, color='#000', linewidth=1, linestyle='--')
    ax.axhline(goal+threshold, color='#000', linewidth=1, linestyle='--')

    ax.bar(0.5, score, color='#2c3e50', width=0.25, zorder=2)

    y = goal-threshold if is_fair else goal-1 if score < goal else goal+threshold
    h = 2*threshold if is_fair else 1-threshold
    bg_color = '#2ecc71' if is_fair else '#e74c3c'
    text = 'fair' if is_fair else 'not fair'
    color = '#27ae60' if is_fair else '#c0392b'

    ax.text(0.01, goal+threshold+0.04, text, fontsize=13,
            color=color, weight="bold")

    rect = Rectangle((0, y), 1, h, facecolor=bg_color,
                     alpha=0.5, zorder=1)
    ax.add_patch(rect)

    ax.get_xaxis().set_ticks([])
    ax.set_title(metric)


def plot_fairness_text(ax, score, metric):
    """Plots bias metric explanation text.

    The text is retrieved by the fairness_metrics_text()
    function.

    Parameters
    ----------
    ax: plt.axes.Axes
        axe where to add the plot
    metric: str
        The name of the metric
    score: float:
        Score value of the metric  
    """
    text = fairness.fairness_metrics_text(score, metric)

    text = "\n".join(textwrap.wrap(text, width=17))

    ax.text(-0.5, 1, text, ha='left', fontsize=12,
            va='top', wrap=True)

    for sp in ['top', 'right', 'left', 'bottom']:
        ax.spines[sp].set_visible(False)
    ax.set_yticks([])
    ax.set_xticks([])


def plot_bias(y_true, y_pred, df, privileged_group, pos_label=1,
              regr_split=None, with_text=True, **kwargs):
    """Plots the fairness metrics for protected attributes
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

    Example
    =======
    
    Using this function for a binary classifier:

    >>> from transparentai.datasets import load_adult
    >>> from sklearn.ensemble import RandomForestClassifier

    >>> data = load_adult()
    >>> X, Y = data.drop(columns='income'), data['income'].replace({'>50K':1, '<=50K':0})
    >>> X = X.select_dtypes('number')
    >>> clf = RandomForestClassifier().fit(X,Y)
    >>> y_pred = clf.predict(X)

    >>> privileged_group = { 'gender':['Male'] }

    >>> y_pred = clf.predict(X)plot_bias(Y, y_pred, data, privileged_group, with_text=True)


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
    with_text: bool (default True)
        Whether it displays the explanation text for 
        fairness metrics.
    """
    metrics = ['statistical_parity_difference',
               'disparate_impact',
               'equal_opportunity_difference',
               'average_odds_difference']

    scores = fairness.compute_fairness_metrics(y_true, y_pred, df, privileged_group,
                                               metrics, pos_label, regr_split='mean')

    n_attr = len(scores)

    if not with_text:
        widths = [1]*4
        heights = [1, 5, 2]*n_attr
    else:
        widths = [3, 1]*2
        heights = [1, 5, 5, 2]*n_attr

    fig = plt.figure(figsize=(15, 7*n_attr + (int(with_text)*8)))

    gs = fig.add_gridspec(len(heights), 4, wspace=0.3,
                          width_ratios=widths,
                          height_ratios=heights)

    row = 0
    for attr, bias_scores in scores.items():
        ax = fig.add_subplot(gs[row, :])
        plot_attr_title(ax, attr, df, privileged_group)

        axes = [fig.add_subplot(gs[row+1+j, i])
                for j in range(int(with_text)+1)
                for i in range(4)]

        for i, (metric, score) in enumerate(bias_scores.items()):
            ax = axes[i] if not with_text else axes[i*2]
            plot_bias_one_attr(ax, metric, score)

            if not with_text:
                continue

            ax = axes[i*2+1]
            plot_fairness_text(ax, score, metric)

        # Separator line
        ax = fig.add_subplot(gs[row+2, :])
        plt.axis('off')

        row += 3+int(with_text)

    # plt.show()
    return plots.plot_or_figure(fig, **kwargs)
