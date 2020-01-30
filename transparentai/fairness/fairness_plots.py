import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.image as mpimg
import seaborn as sns
import numpy as np
from IPython.display import display, Markdown

from transparentai.fairness.protected_attribute import ProtectedAttribute

BIAS_COLORS = ['#3498db', '#ecf0f1']

def plot_text_center(ax, text, fontsize=18):
    """
    Display text at the center of an ax from a matplotlib figure.

    Parameters
    ----------
    ax: plt.axes.Axes
        ax where to set centered text
    text: str
        text to display
    fontsize: int (optionnal)
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


def plot_percentage_bar_man_img(ax, freq, spacing=0):
    """
    Display a percentage bar plot with custom images. only 10 images
    will be ploted, and each images has only one color so if the frequency is
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


def plot_pie(ax, freq):
    """
    Display a pie chart from a frequency.
    
    This function uses BIAS_COLORS variable.
    
    Parameters
    ----------
    ax: plt.axes.Axes
        ax where to plot the pie chart
    freq: str
        frequency for the pie chart
    """
    sizes = [freq, 1-freq]
    ax.pie(sizes, labels=['', ''], colors=BIAS_COLORS,
           shadow=True, startangle=130)
    ax.axis('equal')


def plot_protected_attr_row(axes, protected_attr, target_value, privileged=True):
    """
    Display a row of graphics for a protected attribute.
    privileged has to be True or False and then the function compute
    the frequency for this population.
    
    axes need 4 axes : 
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
    
    .. _matplotlib: https://matplotlib.org/3.1.1/gallery/ticks_and_spines/tick-locators.html#sphx-glr-gallery-ticks-and-spines-tick-locators-py
    
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
        f' : means {protected_attr.get_privileged_values()}'

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


def get_protected_attr_freq(protected_attr, target_value, privileged=None):
    """
    Put into protected attributes class
    
    Parameters
    ----------
    
    Returns
    -------
    float
        Frequency
    """
    n_total = protected_attr.num_instances(privileged=privileged)
    n = protected_attr.num_spec_value(
        target_value=target_value, privileged=privileged)
    return n / n_total

        
def get_metric_text(protected_attr, target_value, metric_name):
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
        
    Returns
    -------
    str
        String of the metric calculus
    """
    metrics = protected_attr.metrics
    freq_unpr = get_protected_attr_freq(protected_attr, target_value, privileged=False)
    freq_priv = get_protected_attr_freq(protected_attr, target_value, privileged=True)
    value = metrics.loc[target_value,metric_name]
    
    if metric_name == 'Disparate impact':
        return r'$\frac{%.2f}{%.2f}$ = %.2f' % (
                freq_unpr, freq_priv, value)
    elif metric_name == 'Statistical parity difference':
        return r'$%.2f - %.2f$ = %.2f' % (freq_unpr*100, freq_priv*100, value*100)
    else:
        return 'no metric selected'
    
    
def get_metric_goal(metric):
    """
    Return bias metric goal given metric name.
    
    Parameters
    ---------- 
    metric: str
        Metric's name 
        
    Returns
    -------
    number
        Bias metric's goal
    """
    if metric == 'Disparate impact':
        return 1
    elif metric == 'Statistical parity difference':
        return 0
    else:
        return 0
    
def plot_gauge(ax, value, goal, bar_color='r', gap=1):
    """
    Display a gauge plot with a blue diverging palette.
    
    Parameters
    ----------
    ax: plt.axes.Axes
        Axe where to put the gauge plot
    value: float
        Current value
    goal: number
        Goal value, the graphic will be centered on this value
    bar_color: str
        Color of the bar where the value is
    gap: int
        Limit gap from the goal value (goal+/-gap)
    """
    cmap = sns.diverging_palette(255, 255, sep=10, n=100, as_cmap=True)
    colors = cmap(np.arange(cmap.N))
    ax.imshow([colors], extent=[goal-gap, goal+gap, 0, 0.25])
    ax.set_yticks([])
    ax.set_xlim(goal-gap, goal+gap)
    ax.axvline(linewidth=4, color=bar_color, x=value)
    
    
def plot_bias_metric(axes, protected_attr, target_value, metric):
    """
    Display a row of graphics for a specific bias metric.
    
    Parameters
    ----------
    axes: list
        Axes for this row
    protected_attr:
        Protected attribute to inspect
    target_value: str
        Specific value of the target    
    metric: str
        Metric's name
    """
    value = protected_attr.metrics.loc[target_value,metric]
    goal = get_metric_goal(metric=metric)
    
    text = get_metric_text(protected_attr, target_value, metric_name=metric)
    plot_text_center(ax=axes[0], text=text, fontsize=26)
    axes[0].set_title(metric, loc='left', fontsize=22)
    axes[0].axis('off')
    
    color = 'r' if (value < goal-0.2) or (value > goal+0.2) else 'g'
    plot_gauge(ax=axes[1], value=value, goal=goal, bar_color=color)
    axes[1].set_title(f'Considered not biased between {goal-0.2} and {goal+0.2}', loc='left', fontsize=17)


def set_metric_title(axes):
    """
    Display bias metrics title.
    
    Parameters
    ----------
    axes: list
        Axes for this row 
    """
    text = 'Dataset Bias metrics'
    setup_bottom_line(axes[0])
    axes[0].text(0.0, 0.2, text, fontsize=22,
                 transform=axes[0].transAxes, fontweight='bold')
    
def generate_gridspec_row(fig, gs, rstart, rsize=1, n_blocks=1, b_sizes=[10]):
    """
    Return a list of axes for a gridpsec graphic.
    You can use this function with the following logic :
    if you want a row of 3 blocks of size 2,2,6 at the 3rd row :
    `generate_gridspec_row(fig, gs, rstart=3, rsize=1, n_blocks=3, b_sizes=[2,2,6])`
    
    Parameters
    ----------
    fig:
        Figure
    gs:
        returns of fig.add_gridspec() function
    rstart: int
        Row start
    rsize: int (default 1)
        Size of the row (number of gridspec rows to occupe)
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
    for i in range(0,n_blocks):
        b_start = b_end
        b_end += b_sizes[i]
        rows.append(fig.add_subplot(gs[rstart:rstart+rsize, b_start:b_end]))
    return rows



def plot_bias_metrics(protected_attr, target_value):
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
    fig = plt.figure(constrained_layout=True, figsize=(16, 14))
    gs = fig.add_gridspec(8, 10)
    
    # Unpriviliged plot
    axes = generate_gridspec_row(fig, gs, rstart=1, rsize=2, n_blocks=4, b_sizes=[5,1,2,2])
    plot_protected_attr_row(axes, protected_attr, target_value, privileged=False)
    
    # Priviliged plot
    axes = generate_gridspec_row(fig, gs, rstart=3, rsize=2, n_blocks=4, b_sizes=[5,1,2,2])
    plot_protected_attr_row(axes, protected_attr, target_value, privileged=True)
    
    # Plot title and legend
    axes = generate_gridspec_row(fig, gs, rstart=0, rsize=1, n_blocks=1, b_sizes=[10])
    set_legend_protected_attr(fig, axes, protected_attr, target_value)
    
    # Metric title
    axes = generate_gridspec_row(fig, gs, rstart=5, rsize=1, n_blocks=1, b_sizes=[10])
    set_metric_title(axes)
    
    # Metrics plots
    metrics = ['Disparate impact', 'Statistical parity difference']
    for idx, metric in enumerate(metrics):
        axes = generate_gridspec_row(fig, gs, rstart=6+idx, rsize=1, n_blocks=2, b_sizes=[4,6])
        plot_bias_metric(axes, protected_attr, target_value, metric=metric)
        
    plt.show()


def plot_dataset_metrics(protected_attr, target_value=None):
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
    if target_value is not None:
        plot_bias_metrics(protected_attr=protected_attr,
                           target_value=target_value)
    else:
        for target_value in protected_attr.labels.unique():
            plot_bias_metrics(protected_attr=protected_attr,
                               target_value=target_value)
