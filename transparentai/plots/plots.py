"""
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.image as mpimg
import os

import matplotlib.ticker as ticker

from transparentai.datasets.protected_attribute import ProtectedAttribute


def plot_text_center(ax, text, fontsize=18):
    """
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


def plot_pie_freq(ax, freq):
    """
    """
    sizes = [freq, 1-freq]
    colors = ['#3498db', '#ecf0f1']
    ax.pie(sizes, labels=['', ''], colors=colors, shadow=True, startangle=130)
    ax.axis('equal')


def plot_protected_attr_row(axes, protected_attr, label_value=None, privileged=True):
    """
    """
    n_total = protected_attr.num_instances(privileged=privileged)
    n = protected_attr.num_spec_value(
        label_value=label_value, privileged=privileged)
    freq = n / n_total
    title = 'Privileged' if privileged else 'Unprivileged'

    plot_percentage_bar_man_img(ax=axes[0], freq=freq)
    axes[0].set_title(title, loc='left', fontsize=18)
    plot_text_center(ax=axes[1], text='=', fontsize=25)
    plot_pie_freq(ax=axes[2], freq=freq)

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


def set_legend_protected_attr(fig, axes, protected_attr, label_value):
    """
    """
    target = protected_attr.target
    attr_name = protected_attr.name

    text = f'Focus on {attr_name} for {target} is {label_value}'
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

    text_blue = f'{target} is {label_value}'
    text_grey = f'{target} is not {label_value}'
    fig.legend(labels=[text_blue, text_grey], fontsize=18)


def get_protected_attr_freq(protected_attr, label_value, privileged=None):
    """
    """
    n_total = protected_attr.num_instances(privileged=privileged)
    n = protected_attr.num_spec_value(
        label_value=label_value, privileged=privileged)
    return n / n_total


def plot_metrics_text(axes, protected_attr, label_value):
    """
    """
    metrics = protected_attr.metrics
    freq_unpr = get_protected_attr_freq(
        protected_attr=protected_attr, label_value=label_value, privileged=False)
    freq_priv = get_protected_attr_freq(
        protected_attr=protected_attr, label_value=label_value, privileged=True)

    for i, metric in enumerate(metrics.columns):
        text = ''
        if metric == 'Disparate impact':
            text = r'$\frac{%.2f}{%.2f}$ = %.2f' % (
                freq_unpr, freq_priv, metrics.loc[label_value, metric])
        elif metric == 'Statistical parity difference':
            text = r'$%.2f - %.2f$ = %.2f' % (freq_unpr*100,
                                              freq_priv*100, metrics.loc[label_value, metric]*100)

        plot_text_center(ax=axes[i], text=text, fontsize=32)

        axes[i].set_title(metric, fontsize=24)
        axes[i].axis('off')


def plot_metrics_label(protected_attr, label_value):
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
                            label_value, privileged=False)
    plot_protected_attr_row(axes_priv, protected_attr,
                            label_value, privileged=True)

    set_legend_protected_attr(fig, axes_legend, protected_attr, label_value)

    plot_metrics_text(axes=axes_metrics,
                      protected_attr=protected_attr, label_value=label_value)

    plt.show()


def plot_dataset_metrics(protected_attr, label_value=None):
    """
    """
    if not isinstance(protected_attr, ProtectedAttribute):
        raise TypeError("Must be a ProtectedAttribute object!")

    if label_value != None:
        plot_metrics_label(protected_attr=protected_attr,
                           label_value=label_value)
    else:
        for label_value in protected_attr.labels.unique():
            plot_metrics_label(protected_attr=protected_attr,
                               label_value=label_value)


def plot_stack(ax, tab, labels):
    """
    Add a stack plot into the current ax plot
    """
    colors = sns.color_palette("colorblind", len(labels))
    x = tab.index.values
    y = [list(tab[str(l)].values) for l in labels]

    ax.stackplot(x, y, labels=labels, colors=colors)
    ax.legend(loc=0, frameon=True)


def plot_stack_bar(ax, tab, labels, legend_labels):
    """
    Add a stack bar plot into the current ax plot
    """
    colors = sns.color_palette("colorblind", len(legend_labels))

    # tab.div(tab.sum(axis=1), axis=0).round(2).astype(int)
    for i, row in tab.iterrows():
        tab.loc[i] = ((row / row.sum()).round(2)*100).astype(int)

    for i, l in enumerate(legend_labels):
        bottom = tab[legend_labels[i-1]] if i > 0 else None

        rects = ax.bar(labels, tab[l], label=l, bottom=bottom,
                       align='center', color=colors[i])

        for r in rects:
            h, w, x, y = r.get_height(), r.get_width(), r.get_x(), r.get_y()
            if h == 0:
                continue

            ax.annotate(str(h)+'%', xy=(x+w/2, y+h/2), xytext=(0, 0),
                        textcoords="offset points",
                        ha='center', va='center',
                        color='white', weight='bold', clip_on=True)

    ax.legend(loc=0, frameon=True)


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
        grid = sns.JointGrid(x=var1, y=var2, data=df, height=7)

        g = grid.plot_joint(sns.scatterplot, hue=target, data=df, alpha=0.3)
        for l in legend_labels:
            sns.distplot(df.loc[df[target] == l, var1], ax=g.ax_marg_x)
            sns.distplot(df.loc[df[target] == l, var2],
                         ax=g.ax_marg_y, vertical=True)
    plt.show()


def plot_barplot_cat_num_var(df, cat_var, num_var, target=None):
    """
    Show boxplots for a specific pair of categorical and numerical variables
    If target is set, separate dataset for each target value.

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe to inspect
    cat_var: str
        Column name that contains categorical values
    num_var: str
        Column name that contains numerical values
    target: str (optional)
        Target column for classifier
    """
    val_cnt = df[cat_var].value_counts()
    if len(val_cnt) > 10:
        val_cnt = val_cnt.head(10)

    df_plot = df[df[cat_var].apply(lambda x: x in val_cnt.index.values)]

    fig, ax = plt.subplots(figsize=(16, 5))
    palette = "Blues" if target == None else "colorblind"
    sns.boxplot(x=cat_var, y=num_var, hue=target,
                data=df_plot, palette=palette)
    plt.xticks(rotation=60)
    plt.show()


def plot_correlation_matrix(corr_df):
    """
    Plot a seaborn heatmap based on a correlation dataframe.

    Parameters
    ----------
    corr_df: pd.DataFrame
        Correlation dataframe
    """
    annot = False if max(corr_df.shape) > 20 else True

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
