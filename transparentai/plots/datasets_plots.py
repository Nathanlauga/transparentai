import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.image as mpimg
import seaborn as sns
from IPython.display import display, Markdown

from transparentai import utils
from transparentai import plots
from transparentai.datasets.protected_attribute import ProtectedAttribute


__SAVEPLOT__ = False
BIAS_COLORS = ['#3498db', '#ecf0f1']


def plot_or_save(fname=None):
    """

    Parameters
    ----------
    fname: str
        file name where to save the plot
    """
    if not __SAVEPLOT__:
        plt.show()
    else:
        fname = 'plot.png' if fname is None else fname
        plt.savefig(fname)


def plot_missing_values(df):
    """
    Show a bar plot that display percentage of missing values on columns that have some.
    If no missing value then it use `display` & `Markdown` functions to indicate it.

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe to inspect
    """
    df_null = pd.DataFrame(len(df) - df.notnull().sum(), columns=['Count'])
    df_null = df_null[df_null['Count'] > 0].sort_values(
        by='Count', ascending=False)
    df_null = df_null/len(df)*100

    if len(df_null) == 0:
        display(Markdown('No missing value.'))
        return

    x = df_null.index.values
    height = [e[0] for e in df_null.values]

    fig, ax = plt.subplots(figsize=(20, 5))
    ax.bar(x, height, width=0.8)
    plt.xticks(x, x, rotation=30)
    plt.xlabel('Columns')
    plt.ylabel('Percentage')
    plt.title('Percentage of missing values in columns')

    plot_or_save(fname='missing_values.png')


def plot_numerical_var(df, var, target=None):
    """
    Show variable information in graphics for numerical variables.
    At least the displot & boxplot.
    If target is set 2 more plots : stack plot and stack plot with percentage

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe to inspect
    var: str
        Column name that contains numerical values
    target: str (optional)
        Target column for classifier
    """
    rows = 1 if target is None else 2
    fig, ax = plt.subplots(figsize=(16, rows*5))

    ax = plt.subplot(int(f'{rows}21'))
    if target == None:
        sns.distplot(df[var])
    else:
        labels = sorted(df[target].unique())
        for l in labels:
            df_target = df[df[target] == l]
            if df_target[var].nunique() <= 1:
                sns.distplot(df_target[var], kde=False)
            else:
                sns.distplot(df_target[var])
            del df_target

    ax = plt.subplot(int(f'{rows}22'))
    x = df[target] if target != None else None

    sns.boxplot(x=x, y=df[var])

    if target != None:
        tab = pd.crosstab(df[var], df[target])

        ax = plt.subplot(223)
        plots.plot_stack(ax=ax, tab=tab, labels=labels)

        tab = tab.div(tab.sum(axis=1), axis=0)

        ax = plt.subplot(224)
        plots.plot_stack(ax=ax, tab=tab, labels=labels)

    plot_or_save(fname=f'{var}_variable_plot.png')


def plot_categorical_var(df, var, target=None):
    """
    Show variable information in graphics for categorical variables.
    For 10 most frequents values : bar plot and a pie chart

    If target is set : plot stack bar for 10 most frequents values

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe to inspect
    var: str
        Column name that contains categorical values
    target: str (optional)
        Target column for classifier
    """
    val_cnt = df[var].value_counts()
    if len(val_cnt) > 10:
        val_cnt = val_cnt.head(10)

    labels = val_cnt.index
    sizes = val_cnt.values
    colors = sns.color_palette("Blues", len(labels))

    rows = 1 if target is None else 2
    fig, ax = plt.subplots(figsize=(16, rows*5))

    ax = plt.subplot(int(f'{rows}21'))
    ax.bar(labels, sizes, width=0.8)
    plt.xticks(labels, labels, rotation=60)

    ax = plt.subplot(int(f'{rows}22'))
    ax.pie(sizes, labels=labels, colors=colors,
           autopct='%1.0f%%', shadow=True, startangle=130)
    ax.axis('equal')
    ax.legend(loc=0, frameon=True)

    if target != None:
        ax = plt.subplot(212)

        legend_labels = sorted(df[target].unique())
        tab = pd.crosstab(df[var], df[target])
        tab = tab.loc[labels]
        plots.plot_stack_bar(ax=ax, tab=tab, labels=labels,
                             legend_labels=legend_labels)

    plot_or_save(fname=f'{var}_variable_plot.png')


def plot_datetime_var(df, var, target=None):
    """
    Show variable information in graphics for datetime variables.
    Display only the time series line if no target is set else, it shows
    2 graphics one with differents lines by value of target and one stack line plot

    If difference between maximum date and minimum date is above 1000 then plot by year.

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe to inspect
    var: str
        Column name that contains datetime values
    target: str (optional)
        Target column for classifier
    """
    df = df.copy()
    rows = 1 if target is None else 2
    fig, ax = plt.subplots(figsize=(16, rows*5))

    date_min = df[var].min()
    date_max = df[var].max()
    if (date_max - date_min).days > 1000:
        df[var] = df[var].dt.year

    if target == None:
        val_cnt = df[var].value_counts()
        sns.lineplot(data=val_cnt)
    else:
        ax = plt.subplot(121)

        legend_labels = sorted(df[target].unique())
        tab = pd.crosstab(df[var], df[target])
        sns.lineplot(data=tab)

        tab.div(tab.sum(axis=1), axis=0)

        ax = plt.subplot(122)
        plots.plot_stack(ax=ax, tab=tab, labels=legend_labels)

    plot_or_save(fname=f'{var}_variable_plot.png')


def display_meta_var(df, var):
    """
    Display some meta informations about a specific variable of a given dataframe
    Meta informations : # of null values, # of uniques values and 2 most frequent values

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe to inspect
    var: str
        Column name that is in df        
    """
    nb_null = df[var].isnull().sum()
    nb_uniq = df[var].nunique()
    val_cnt = df[var].value_counts()
    n = 2 if len(val_cnt) > 1 else len(val_cnt)
    most_freq = df[var].value_counts().head(n).to_dict()
    display(Markdown(
        f'**{var} :** {nb_null} nulls, {nb_uniq} unique vals, most common: {most_freq}'))


def plot_numerical_jointplot(df, var1, var2, target=None):
    """
    Show two numerical variables relations with jointplot.

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe to inspect
    var1: str
        Column name that contains first numerical values
    var2: str
        Column name that contains second numerical values
    target: str (optional)
        Target column for classifier
    """
    if target == None:
        g = sns.jointplot(var1, var2, data=df, kind="hex", space=0, height=8)
    else:
        legend_labels = sorted(df[target].unique())
        cols = [var1, var2, target] if target is not None else [var1, var2]
        df = df[cols].dropna()
        grid = sns.JointGrid(x=var1, y=var2, data=df, height=7)
        
        g = grid.plot_joint(sns.scatterplot, hue=target, data=df, alpha=0.3)
        for l in legend_labels:
            sns.distplot(df.loc[df[target] == l, var1], ax=g.ax_marg_x)
            sns.distplot(df.loc[df[target] == l, var2],
                         ax=g.ax_marg_y, vertical=True)

    plot_or_save(fname=f'{var1}_{var2}_variable_jointplot.png')


def plot_boxplot_two_variables(df, var1, var2, target=None):
    """
    Show boxplots for a specific pair of categorical and numerical variables
    If target is set, separate dataset for each target value.

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe to inspect
    var1: str
        Column name that contains categorical values
    var2: str
        Column name that contains numerical values
    target: str (optional)
        Target column for classifier
    """
    val_cnt = df[var1].value_counts()
    if len(val_cnt) > 10:
        val_cnt = val_cnt.head(10)

    df_plot = df[df[var1].apply(lambda x: x in val_cnt.index.values)]

    fig, ax = plt.subplots(figsize=(16, 5))
    palette = "Blues" if target == None else "colorblind"
    sns.boxplot(x=var1, y=var2, hue=target,
                data=df_plot, palette=palette)
    plt.xticks(rotation=40)

    plot_or_save(fname=f'{var1}_{var2}_variable_boxplot.png')


def plot_correlation_matrix(corr_df):
    """
    Plot a seaborn heatmap based on a correlation dataframe.

    Parameters
    ----------
    corr_df: pd.DataFrame
        Correlation dataframe
    """
    annot = max(corr_df.shape) <= 20

    fig, ax = plt.subplots(figsize=(11, 12))
    ax = sns.heatmap(
        corr_df,
        vmin=-1, vmax=1, center=0,
        cmap=sns.color_palette("RdBu_r", 100),
        square=(corr_df.shape[0] == corr_df.shape[1]),
        annot=annot,
        fmt='.2f'
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    )
    plt.show()


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
    """
    sizes = [freq, 1-freq]
    ax.pie(sizes, labels=['', ''], colors=BIAS_COLORS,
           shadow=True, startangle=130)
    ax.axis('equal')


def plot_protected_attr_row(axes, protected_attr, target_value=None, privileged=True):
    """
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
    code from : https://matplotlib.org/3.1.1/gallery/ticks_and_spines/tick-locators.html#sphx-glr-gallery-ticks-and-spines-tick-locators-py
    """
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.yaxis.set_major_locator(ticker.NullLocator())
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(which='major', width=1.00, length=5)
    ax.tick_params(which='minor', width=0.75, length=2.5, labelsize=10)
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 1)
    ax.patch.set_alpha(0.0)
    ax.xaxis.set_major_locator(ticker.NullLocator())
    ax.xaxis.set_minor_locator(ticker.NullLocator())


def set_legend_protected_attr(fig, axes, protected_attr, target_value):
    """
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
    """
    n_total = protected_attr.num_instances(privileged=privileged)
    n = protected_attr.num_spec_value(
        target_value=target_value, privileged=privileged)
    return n / n_total


def plot_metrics_text(axes, protected_attr, target_value):
    """
    """
    metrics = protected_attr.metrics
    freq_unpr = get_protected_attr_freq(
        protected_attr=protected_attr, target_value=target_value, privileged=False)
    freq_priv = get_protected_attr_freq(
        protected_attr=protected_attr, target_value=target_value, privileged=True)

    for i, metric in enumerate(metrics.columns):
        text = ''
        if metric == 'Disparate impact':
            text = r'$\frac{%.2f}{%.2f}$ = %.2f' % (
                freq_unpr, freq_priv, metrics.loc[target_value, metric])
        elif metric == 'Statistical parity difference':
            text = r'$%.2f - %.2f$ = %.2f' % (freq_unpr*100,
                                              freq_priv*100, metrics.loc[target_value, metric]*100)

        plot_text_center(ax=axes[i], text=text, fontsize=32)

        axes[i].set_title(metric, fontsize=24)
        axes[i].axis('off')


def plot_metrics_label(protected_attr, target_value):
    """
    """
    fig = plt.figure(constrained_layout=True, figsize=(16, 10))
    gs = fig.add_gridspec(6, 10)

    leg_ax1 = fig.add_subplot(gs[0, :])
    unp_ax1 = fig.add_subplot(gs[1:3, :-5])
    unp_ax2 = fig.add_subplot(gs[1:3, -5])
    unp_ax3 = fig.add_subplot(gs[1:3, 6:8])
    unp_ax4 = fig.add_subplot(gs[1:3, 8:10])
    pri_ax1 = fig.add_subplot(gs[3:5:, :-5])
    pri_ax2 = fig.add_subplot(gs[3:5, -5])
    pri_ax3 = fig.add_subplot(gs[3:5, 6:8])
    pri_ax4 = fig.add_subplot(gs[3:5, 8:10])
    met_ax1 = fig.add_subplot(gs[5, 0:5])
    met_ax2 = fig.add_subplot(gs[5, 5:])

    axes_legend = [leg_ax1]
    axes_unpriv = [unp_ax1, unp_ax2, unp_ax3, unp_ax4]
    axes_priv = [pri_ax1, pri_ax2, pri_ax3, pri_ax4]
    axes_metrics = [met_ax1, met_ax2]

    plot_protected_attr_row(axes_unpriv, protected_attr,
                            target_value, privileged=False)
    plot_protected_attr_row(axes_priv, protected_attr,
                            target_value, privileged=True)

    set_legend_protected_attr(fig, axes_legend, protected_attr, target_value)

    plot_metrics_text(axes=axes_metrics,
                      protected_attr=protected_attr, target_value=target_value)

    plt.show()


def plot_dataset_metrics(protected_attr, target_value=None):
    """
    """
    if not isinstance(protected_attr, ProtectedAttribute):
        raise TypeError("Must be a ProtectedAttribute object!")

    if target_value != None:
        plot_metrics_label(protected_attr=protected_attr,
                           target_value=target_value)
    else:
        for target_value in protected_attr.labels.unique():
            plot_metrics_label(protected_attr=protected_attr,
                               target_value=target_value)
