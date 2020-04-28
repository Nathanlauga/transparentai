import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.image as mpimg
import seaborn as sns
import numpy as np
from IPython.display import display, Markdown

from .protected_attribute import ProtectedAttribute
from .. import utils
from .. import plots

bias_colors = ['#3498db', '#ecf0f1']
predicted_colors = ['#bdc3c7', '#2980b9']


def plot_text_center(ax, text, fontsize=18):
    """
    Display text at the center of an ax from a matplotlib figure.

    Parameters
    ----------
    ax: plt.axes.Axes
        ax where to set centered text
    text: str
        text to display
    fontsize: int (default 18)
        font size of the text
    """
    left, width = .25, .5
    bottom, height = .25, .5
    right = left + width
    top = bottom + height

    ax.text(0.5 * (left + right), 0.5 * (bottom + top), text,
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=fontsize,
            transform=ax.transAxes)

    ax.axis('off')

# TODO : refactor text center & left


def plot_text_left(ax, text, fontsize=18):
    """
    Display text at the left of an ax from a matplotlib figure.

    Parameters
    ----------
    ax: plt.axes.Axes
        ax where to set centered text
    text: str
        text to display
    fontsize: int (default 18)
        font size of the text
    """
    left, width = 0, 0
    bottom, height = .25, .5
    right = left + width
    top = bottom + height

    ax.text(0.5 * (left + right), 0.5 * (bottom + top), text,
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=fontsize, ha='left', multialignment='left',
            transform=ax.transAxes)

    ax.axis('off')


def plot_percentage_bar_man_img(ax, freq, spacing=0):
    """
    Display a percentage bar plot with custom images. only 10 images
    will be plotted, and each image has only one color so if the frequency is
    0.66 then it rounded up (0.7) and print 7 blue men and 3 grey.

    Parameters
    ----------
    ax: plt.axes.Axes
        ax where to plot the image barplot
    freq: str
        frequency to plot with man image
    spacing: int (default 0)
        spacing between images
    """
    heights = [70 for i in range(0, 10)]

    for i, height in enumerate(heights):
        img_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '../src/img/'))
        img_path += '/_man_blue.png' if i < round(
            freq, 1)*10 else '/_man_grey.png'
        img = mpimg.imread(img_path)
        # AR = img.shape[1] / img.shape[0]
        AR = 135 / 256
        width = height * AR
        left = width*i + spacing*i
        right = left + width
        ax.imshow(img, extent=[left, right, 0, height])

    ax.set_xlim(0, right)
    ax.set_ylim(0, max(heights)*1.1)
    ax.axis('off')


def plot_pie(ax, freq, annot=False):
    """
    Display a pie chart from a frequency.

    This function uses bias_colors variable.

    Parameters
    ----------
    ax: plt.axes.Axes
        ax where to plot the pie chart
    freq: str
        frequency for the pie chart
    annot: bool (default False)
        add percentage annotation in the pie chart
    """
    sizes = [freq, 1-freq]
    autopct = '%1.0f%%' if annot else ''

    ax.pie(sizes, labels=['', ''], colors=bias_colors, autopct=autopct,
           shadow=True, startangle=130)
    ax.axis('equal')


def plot_protected_attr_row(axes, protected_attr, target_value, privileged=True):
    """
    Display a row of graphics for a protected attribute.
    privileged has to be True or False and then the function compute
    the frequency for this population.

    `axes` need 4 axes : 

    1. the percentage image plot (`plot_percentage_bar_man_img()`)
    2. an equal symbol
    3. pie chart of frequency
    4. text of the calculus

    Parameters
    ----------
    axes: list
        Axes for this row
    protected_attr:
        Protected attribute to inspect
    target_value: str
        Specific value of the target    
    privileged: bool (default True)
        Boolean prescribing whether to
        condition this metric on the `privileged_groups`, if `True`, or
        the `unprivileged_groups`, if `False`. Defaults to `None`
        meaning this metric is computed over the entire dataset.
    """
    n_total = protected_attr.num_instances(privileged=privileged)
    n = protected_attr.num_spec_value(
        target_value=target_value, privileged=privileged)
    freq = n / n_total
    title = 'Privileged' if privileged else 'Unprivileged'

    plot_percentage_bar_man_img(ax=axes[0], freq=freq)
    axes[0].set_title(title, loc='left', fontsize=18)
    plot_text_center(ax=axes[1], text='=', fontsize=25)
    plot_pie(ax=axes[2], freq=freq)

    text1 = '%.2f%%' % (freq*100)
    text2 = f'\n\n\n\n({n} / {n_total})'
    plot_text_center(ax=axes[3], text=text1, fontsize=32)
    plot_text_center(ax=axes[3], text=text2, fontsize=14)


def setup_bottom_line(ax):
    """
    Removes all axis except the bottom one.

    Code inspired from matplotlib.org_.

    .. _matplotlib.org: https://matplotlib.org/3.1.1/gallery/ticks_and_spines/tick-locators.html#sphx-glr-gallery-ticks-and-spines-tick-locators-py

    Parameters
    ----------
    ax: plt.axes.Axes
        ax to update
    """
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.set_yticks([])
    ax.set_xticks([])


def set_legend_protected_attr(fig, axes, protected_attr, target_value):
    """
    Display title and legend of the bias graphic.

    Parameters
    ----------
    fig:
        Figure object
    axes: list
        Axes for this row
    protected_attr:
        Protected attribute to inspect
    target_value: str
        Specific value of the target  
    """
    target = protected_attr.target
    attr_name = protected_attr.name

    text = f'Focus on {attr_name} for {target} is {target_value}'
    unp_text = r'$\bf{Unprivileged}$' + \
        f' means {protected_attr.get_unprivileged_values()}'
    pri_text = r'$\bf{Privileged}$' + \
        f' means {protected_attr.get_privileged_values()}'

    setup_bottom_line(axes[0])

    axes[0].text(0.0, 0.9, text, fontsize=22,
                 transform=axes[0].transAxes, fontweight='bold')
    axes[0].text(0.0, 0.5, unp_text, fontsize=16,
                 transform=axes[0].transAxes, fontweight='normal')
    axes[0].text(0.0, 0.1, pri_text, fontsize=16,
                 transform=axes[0].transAxes, fontweight='normal')

    text_blue = f'{target} is {target_value}'
    text_grey = f'{target} is not {target_value}'
    fig.legend(labels=[text_blue, text_grey], fontsize=18)


def get_metric_text(protected_attr, target_value, metric_name, bias_type='dataset'):
    """
    Return a text value for a specific metric

    Parameters
    ----------
    protected_attr:
        Protected attribute to inspect
    target_value: str
        Specific value of the target  
    metric_name: str
        Metric's name 
    bias_type: str (default: 'dataset')
        Bias type (can only be 'dataset' or 'model')

    Returns
    -------
    str
        String of the metric calculus

    Raises
    ------
    ValueError:
        `bias_type` parameter has to be 'dataset' or 'model'
    """
    if bias_type not in ['dataset','model']:
        raise ValueError("bias_type has to be 'dataset' or 'model'")

    metrics = protected_attr.metrics
    value = metrics.loc[target_value, metric_name]

    if bias_type == 'dataset':
        freq_unpr = protected_attr.get_freq(target_value, privileged=False)
        freq_priv = protected_attr.get_freq(target_value, privileged=True)

        if metric_name == 'Disparate impact':
            formula = r'$\frac{Pr(Y = %s\ |\ D = unprivileged)}{Pr(Y = %s\ |\ D = privileged)}$' % (
                target_value, target_value)
            return formula + '\n'+r'$ = \frac{%.2f}{%.2f}$ = %.2f' % (freq_unpr, freq_priv, value)

        elif metric_name == 'Statistical parity difference':
            formula = r'$Pr(Y = %s\ |\ _{D = unprivileged}) - Pr(Y = %s\ |\ _{D = privileged})$' % (
                target_value, target_value)

            return formula + '\n' + r'= %.2f - %.2f = %.2f' % (freq_unpr, freq_priv, value)

    elif bias_type == 'model':
        freq_unpr = protected_attr.get_freq(
            target_value, privileged=False, predictions=True)
        freq_priv = protected_attr.get_freq(
            target_value, privileged=True, predictions=True)

        if metric_name == 'Disparate impact':
            formula = r'$\frac{Pr(\hat{Y} = %s\ |\ D = unprivileged)}{Pr(\hat{Y} = %s\ |\ D = privileged)}$' % (
                target_value, target_value)
            return formula + '\n'+r'$ = \frac{%.2f}{%.2f}$ = %.2f' % (freq_unpr, freq_priv, value)

        elif metric_name == 'Statistical parity difference':
            formula = r'$Pr(\hat{Y} = %s\ |\ _{D = unprivileged}) - Pr(\hat{Y} = %s\ |\ _{D = privileged})$' % (
                target_value, target_value)

            return formula + '\n' + r'= %.2f - %.2f = %.2f' % (freq_unpr, freq_priv, value)

        elif metric_name == 'Equal opportunity difference':
            TPR_unpriv = protected_attr.true_positive_rate(
                target_value, privileged=False)
            TPR_priv = protected_attr.true_positive_rate(
                target_value, privileged=True)

            formula = '$TPR_{unprivileged} - TPR_{privileged}$'
            return formula + '\n = %.2f - %.2f = %.2f' % (
                TPR_unpriv, TPR_priv, value)

        elif metric_name == 'Average abs odds difference':
            TPR_unpriv = protected_attr.true_positive_rate(
                target_value, privileged=False)
            TPR_priv = protected_attr.true_positive_rate(
                target_value, privileged=True)
            FPR_unpriv = protected_attr.false_positive_rate(
                target_value, privileged=False)
            FPR_priv = protected_attr.false_positive_rate(
                target_value, privileged=True)

            formula = r'$\frac{1}{2}\left[|FPR_{unprivileged} - FPR_{privileged}| + |TPR_{unprivileged} - TPR_{privileged}|\right]$'
            return formula + '\n' + r'= $\frac{1}{2}\left[|%.2f - %.2f| + |%.2f - %.2f|\right]$ = %.2f' % (
                FPR_unpriv, FPR_priv, TPR_unpriv, TPR_priv, value)

        elif metric_name == 'Theil index':
            formula = r'$\frac{1}{n}\sum_{i=1}^n\frac{b_{i}}{\mu}\ln\frac{b_{i}}{\mu}$'
            return formula + ' = %.2f' % (value)

    return 'no metric selected'


def plot_bias_metric(axes, protected_attr, target_value, metric, bias_type='dataset'):
    """
    Display a row of graphics for a specific bias metric.

    Parameters
    ----------
    axes: list
        Axes for this row
    protected_attr: transparentai.fairness.ProtectedAttribute
        Protected attribute to inspect
    target_value: str
        Specific value of the target    
    metric: str
        Metric's name
    bias_type: str (default: 'dataset')
        Bias type (can only be 'dataset' or 'model')

    Raises
    ------
    ValueError:
        `bias_type` parameter has to be 'dataset' or 'model'
    """
    if bias_type not in ['dataset','model']:
        raise ValueError("bias_type has to be 'dataset' or 'model'")
    value = protected_attr.metrics.loc[target_value, metric]
    goal = utils.get_metric_goal(metric=metric)

    text = get_metric_text(protected_attr, target_value,
                           metric_name=metric, bias_type=bias_type)
    plot_text_left(ax=axes[0], text=text, fontsize=19)
    axes[0].set_title(metric, loc='left', fontsize=22, fontweight='bold')
    axes[0].axis('off')

    color = 'r' if (value < goal-0.2) or (value > goal+0.2) else 'g'
    plots.plot_gauge(ax=axes[1], value=value, goal=goal, bar_color=color)
    axes[1].set_title(
        f'Considered not biased between {goal-0.2} and {goal+0.2}', loc='left', fontsize=17)


def plot_bar_freq_predicted(axes, protected_attr, target_value, privileged=True):
    """
    Display a row with a bar plot for a specific `ProtectedAttribute`.
    The graph represents frequency of predicted 0 versus predicted 1.

    Parameters
    ----------
    axes: list
        Axes for this row
    protected_attr: transparentai.fairness.ProtectedAttribute
        Protected attribute to inspect
    target_value: str
        Specific value of the target    
    privileged: bool (default True)
        Boolean prescribing whether to
        condition this metric on the `privileged_groups`, if `True`, or
        the `unprivileged_groups`, if `False`. Defaults to `None`
        meaning this metric is computed over the entire dataset.
    """
    freq_1 = (protected_attr.get_freq(target_value,
                                      privileged, predictions=True)*100).round(2)
    freq_0 = 100 - freq_1
    perf = protected_attr.performances(target_value, privileged)
    TP, FP, TN, FN = perf['TP'], perf['FP'], perf['TN'], perf['FN']
    TPR = TP / (TP+FN)
    FPR = FP / (TN+FP)
    priv_text = 'Privileged' if privileged else 'Unprivileged'

    rects = axes[0].barh(['', ''], [freq_0, freq_1], color=predicted_colors,
                         left=[0, freq_0], label=['', ''])

    for r in rects:
        h, w, x, y = r.get_height(), r.get_width(), r.get_x(), r.get_y()
        if h == 0:
            continue

        axes[0].annotate(str(w)+'%', xy=(x+w/2, y+h/2), xytext=(0, 0),
                         textcoords="offset points", fontsize=16,
                         ha='center', va='center',
                         color='white', weight='bold', clip_on=True)

    axes[0].text(0, 0.7, r"$FPR_{%s} = \frac{FP}{TN+FP} = \frac{%i}{%i+%i} = %.2f$" % (
        priv_text, FP, TN, FP, FPR), fontsize=20)
    axes[0].text(100, 0.7, r"$TPR_{%s} = \frac{TP}{TP+FN} = \frac{%i}{%i+%i} = %.2f$" % (
        priv_text, TP, TP, FN, TPR), fontsize=20, ha='right')

    axes[0].axis('off')
    axes[0].set_xlim(0, 100)
    axes[0].text(
        0, -0.6, f"{priv_text} predicted not {target_value}", fontsize=16)
    axes[0].text(
        100, -0.6, f"{priv_text} predicted {target_value}", fontsize=16, ha='right')


def plot_protected_attr_row_model(axes, protected_attr, target_value, privileged=True):
    """
    Display a row of graphics for a protected attribute.
    privileged has to be True or False and then the function compute
    the frequency for this population.

    axes need 4 axes : 

    1. the percentage image plot (`plot_percentage_bar_man_img()`) for rows predicted as not target value  (aka negatives)
    2. pie chart of frequency for negatives
    3. pie chart of frequency for positives
    4. the percentage image plot (`plot_percentage_bar_man_img()`) for rows predicted as target value (aka positives)

    Parameters
    ----------
    axes: list
        Axes for this row
    protected_attr:
        Protected attribute to inspect
    target_value: str
        Specific value of the target    
    privileged: bool (default True)
        Boolean prescribing whether to
        condition this metric on the `privileged_groups`, if `True`, or
        the `unprivileged_groups`, if `False`. Defaults to `None`
        meaning this metric is computed over the entire dataset.
    """
    perf = protected_attr.performances(target_value, privileged)
    TP, FP, TN, FN = perf['TP'], perf['FP'], perf['TN'], perf['FN']
    freq_neg = FN / (FN+TN)
    freq_pos = TP / (FP+TP)

    priv_text = 'Privileged' if privileged else 'Unprivileged'

    plot_percentage_bar_man_img(ax=axes[0], freq=freq_neg)
    axes[0].set_title(f'{priv_text} predicted not {target_value}',
                      loc='left', fontsize=18, fontweight='bold')
    axes[0].text(0, -18, f"FN: {FN}", fontsize=16)
    axes[0].text(300, -18, f"TN: {TN}", fontsize=16)

    plot_pie(ax=axes[1], freq=freq_neg, annot=True)
    plot_pie(ax=axes[2], freq=freq_pos, annot=True)
    plot_percentage_bar_man_img(ax=axes[3], freq=freq_pos)
    axes[3].set_title(f'{priv_text} predicted {target_value}',
                      loc='right', fontsize=18, fontweight='bold')
    axes[3].text(0, -18, f"TP: {TP}", fontsize=16)
    axes[3].text(300, -18, f"FP: {FP}", fontsize=16)


def set_metric_title(axes, text, fontsize=22):
    """
    Display bias metrics title.

    Parameters
    ----------
    axes: list
        Axes for this row 
    text: str
        Text for the title
    fontsize: int (default 22)
        font size of the text
    """
    setup_bottom_line(axes[0])
    axes[0].text(0.0, 0.2, text, fontsize=fontsize,
                 transform=axes[0].transAxes, fontweight='bold')


def generate_gridspec_row(fig, gs, rstart, rsize=1, n_blocks=1, b_sizes=[10]):
    """
    Return a list of axes for a gridpsec graphic.
    You can use this function with the following logic :

    if you want a row of 3 blocks of size 2,2,6 at the 3rd row :

    >>> generate_gridspec_row(fig, gs, rstart=3, rsize=1, n_blocks=3, b_sizes=[2,2,6])

    Parameters
    ----------
    fig:
        Figure
    gs:
        returns of fig.add_gridspec() function
    rstart: int
        Row start
    rsize: int (default 1)
        Size of the row (number of gridspec rows)
    n_blocks: int (default 1)
        Number of different axes for this row
    b_sizes: list (default [10])
        Sizes of the blocks

    Returns
    -------
    list:
        list of axes for this row
    """
    rows = list()
    b_end = 0
    for i in range(0, n_blocks):
        b_start = b_end
        b_end += b_sizes[i]
        rows.append(fig.add_subplot(gs[rstart:rstart+rsize, b_start:b_end]))
    return rows


def plot_dataset_bias_metrics(protected_attr, target_value):
    """
    Display matplotlib graphics with different information about
    bias inside a dataset.

    Parameters
    ----------
    protected_attr:
        Protected attribute to inspect
    target_value: str
        Specific value of the target        
    """
    fig = plt.figure(constrained_layout=True, figsize=(16, 14))
    gs = fig.add_gridspec(8, 10)

    # Unpriviliged plot
    axes = generate_gridspec_row(
        fig, gs, rstart=1, rsize=2, n_blocks=4, b_sizes=[5, 1, 2, 2])
    plot_protected_attr_row(axes, protected_attr,
                            target_value, privileged=False)

    # Priviliged plot
    axes = generate_gridspec_row(
        fig, gs, rstart=3, rsize=2, n_blocks=4, b_sizes=[5, 1, 2, 2])
    plot_protected_attr_row(axes, protected_attr,
                            target_value, privileged=True)

    # Plot title and legend
    axes = generate_gridspec_row(
        fig, gs, rstart=0, rsize=1, n_blocks=1, b_sizes=[10])
    set_legend_protected_attr(fig, axes, protected_attr, target_value)

    # Metric title
    axes = generate_gridspec_row(
        fig, gs, rstart=5, rsize=1, n_blocks=1, b_sizes=[10])
    set_metric_title(axes, text='Dataset Bias metrics')

    # Metrics plots
    metrics = ['Disparate impact', 'Statistical parity difference']
    for idx, metric in enumerate(metrics):
        axes = generate_gridspec_row(
            fig, gs, rstart=6+idx, rsize=1, n_blocks=2, b_sizes=[4, 6])
        plot_bias_metric(axes, protected_attr, target_value,
                         metric=metric, bias_type='dataset')

    plots.plot_or_save(fname='dataset_bias_metrics_plot.png')


def plot_model_bias_metrics(protected_attr, target_value):
    """
    Display a matplotlib graphics with differents informations about
    bias inside a dataset.

    Parameters
    ----------
    protected_attr:
        Protected attribute to inspect
    target_value: str
        Specific value of the target        
    """
    fig = plt.figure(constrained_layout=True, figsize=(16, 20))
    gs = fig.add_gridspec(13, 10)

    # Unpriviliged plot
    axes = generate_gridspec_row(
        fig, gs, rstart=1, rsize=2, n_blocks=4, b_sizes=[3, 2, 2, 3])
    plot_protected_attr_row_model(axes, protected_attr,
                                  target_value, privileged=False)
    axes = generate_gridspec_row(
        fig, gs, rstart=3, rsize=1, n_blocks=1, b_sizes=[10])
    plot_bar_freq_predicted(axes, protected_attr,
                            target_value, privileged=False)

    # Priviliged plot
    axes = generate_gridspec_row(
        fig, gs, rstart=4, rsize=2, n_blocks=4, b_sizes=[3, 2, 2, 3])
    plot_protected_attr_row_model(axes, protected_attr,
                                  target_value, privileged=True)
    axes = generate_gridspec_row(
        fig, gs, rstart=6, rsize=1, n_blocks=1, b_sizes=[10])
    plot_bar_freq_predicted(axes, protected_attr,
                            target_value, privileged=True)

    # Plot title and legend
    axes = generate_gridspec_row(
        fig, gs, rstart=0, rsize=1, n_blocks=1, b_sizes=[10])
    set_legend_protected_attr(fig, axes, protected_attr, target_value)

    # Metric title
    axes = generate_gridspec_row(
        fig, gs, rstart=7, rsize=1, n_blocks=1, b_sizes=[10])
    set_metric_title(axes, text='Model Bias metrics')

    # Metrics plots
    metrics = ['Disparate impact', 'Statistical parity difference',
               'Equal opportunity difference', 'Average abs odds difference',
               'Theil index']
    start = 8
    for idx, metric in enumerate(metrics):
        axes = generate_gridspec_row(
            fig, gs, rstart=idx+start, rsize=1, n_blocks=2, b_sizes=[4, 6])
        plot_bias_metric(axes, protected_attr, target_value,
                         metric=metric, bias_type='model')

    plots.plot_or_save(fname='model_bias_metrics_plot.png')


def plot_bias_metrics(protected_attr, bias_type, target_value=None):
    """
    Display a matplotlib graphics with different information about
    bias inside a dataset.

    Parameters
    ----------
    protected_attr:
        Protected attribute to inspect
    bias_type: str
        type of bias to plot ('dataset' or 'model')
    target_value: str (default None)
        Specific value of the target        
    """
    if bias_type not in ['dataset', 'model']:
        raise ValueError("bias_type should be 'dataset' or 'model'")

    func = plot_dataset_bias_metrics if bias_type == 'dataset' else plot_model_bias_metrics
    attr = protected_attr.name
    target = protected_attr.target

    if target_value is not None:
        func(protected_attr=protected_attr, target_value=target_value)
    else:
        for target_value in protected_attr.labels.unique():
            func(protected_attr=protected_attr, target_value=target_value)
