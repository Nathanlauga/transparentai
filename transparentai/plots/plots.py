import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


save_plot = False
save_dir = './'
verbose = 0


def plot_or_save(fname=None):
    """

    Parameters
    ----------
    fname: str
        file name where to save the plot
    """
    if not save_plot:
        plt.show()
    else:
        fname = 'plot.png' if fname is None else fname
        plt.savefig(save_dir+fname)
        if verbose > 0:
            print(f'Plot saved at : {save_dir}{fname}')


def plot_stack(ax, tab, labels):
    """
    Add a stack plot into the current ax plot
    """
    colors = sns.color_palette("colorblind", len(labels))
    x = tab.index.values
    y = [list(tab[l].values) for l in labels]

    ax.stackplot(x, y, labels=labels, colors=colors)
    ax.legend(loc=0, frameon=True)


def plot_stack_bar(ax, tab, labels, legend_labels):
    """
    Add a stack bar plot into the current ax plot
    """
    colors = sns.color_palette("colorblind", len(legend_labels))

    # tab.div(tab.sum(axis=1), axis=0).round(2).astype(int)
    for i, row in tab.iterrows():
        if row.sum() > 0:
            tab.loc[i] = ((row / row.sum()).round(2)*100).astype(int)
        else:
            tab.loc[i] = 0

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
