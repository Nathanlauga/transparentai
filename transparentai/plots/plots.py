import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import numpy as np

import transparentai

save_plot = False
save_dir = './'
verbose = 0

def plot_or_save(fname=None):
    """
    Using `save_plot` global attribute (in this file) it uses
    `plt.show()` or it saves the figure to a file

    To update the `save_plot` to `True` you can use the following code : 


    The following directive changes the hightlight language to SQL.

    .. code-block:: python
    
        import transparentai.plots as plots
        # save_plot allows to save
        plots.save_plot = True

        # For a different directory than the current
        plots.save_dir = 'plots/'
        # If you want to see verbose detail
        plots.verbose = 1

    Parameters
    ----------
    fname: str
        file name where to save the plot
    """
    if transparentai.plots.save_plot:
        save_dir = transparentai.plots.save_dir
        
        fname = 'plot.png' if fname is None else fname
        plt.savefig(save_dir+fname)
        if transparentai.plots.verbose > 0:
            print(f'Plot saved at : {save_dir}{fname}')
    else:
        plt.show()


def plot_stack(ax, df, labels):
    """
    Add a stack plot into the current ax plot.

    Parameters
    ----------
    ax: plt.axes.Axes
        axe where to add the plot 
    df: pd.DataFrame
        Dataframe object to analyse
    labels: list
        list of labels to display in the plot

    Raises
    ------
    TypeError:
        `df` parameter has to be a `pd.DataFrame`
    """
    if type(df) != pd.DataFrame:
        raise TypeError('df has to be a DataFrame')

    colors = sns.color_palette("colorblind", len(labels))
    x = df.index.values
    y = [list(df[l].values) for l in labels if l in df.columns]

    ax.stackplot(x, y, labels=labels, colors=colors)
    ax.legend(loc=0, frameon=True)


def plot_stack_bar(ax, df, labels, legend_labels):
    """
    Add a stack bar plot into the current ax plot.

    Parameters
    ----------
    ax: plt.axes.Axes
        axe where to add the plot 
    df: pd.DataFrame
        Dataframe object to analyse
    labels: list
        list of labels to display in the plot
    legend_labels: list
        list of labels where to split the data in
        stacked area

    Raises
    ------
    TypeError:
        `df` parameter has to be a `pd.DataFrame`
    """
    if type(df) != pd.DataFrame:
        raise TypeError('df has to be a DataFrame')

    colors = sns.color_palette("colorblind", len(legend_labels))

    # df.div(df.sum(axis=1), axis=0).round(2).astype(int)
    for i, row in df.iterrows():
        if row.sum() > 0:
            df.loc[i] = ((row / row.sum()).round(2)*100).astype(int)
        else:
            df.loc[i] = 0

    for i, l in enumerate(legend_labels):
        bottom = df[legend_labels[i-1]] if i > 0 else None

        rects = ax.bar(labels, df[l], label=l, bottom=bottom,
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


def plot_boxplot_cat_num_var(df, cat_var, num_var, target=None):
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
        Color of the vertical line
    gap: int
        Limit gap from the goal value (goal+/-gap)
    """
    # https://github.com/mwaskom/seaborn/issues/1907
    # cmap = sns.diverging_palette(255, 255, sep=10, n=100, as_cmap=True)

    cmap = ListedColormap(sns.color_palette("coolwarm", 100).as_hex())
    colors = cmap(np.arange(cmap.N))
    ax.imshow([colors], extent=[goal-gap, goal+gap, 0, 0.25])
    ax.set_yticks([])
    ax.set_xlim(goal-gap, goal+gap)
    ax.axvline(linewidth=4, color=bar_color, x=value)


def plot_bar(ax, value, label):
    """
    Display a bar chart into the current ax plot.

    Parameters
    ----------
    ax: plt.axes.Axes
        Axe where to put the plot
    value: list or number
        Values list (same size than `label`)
    label: list or str
        Labels list (same size than `value`)
    """
    rects = ax.bar(label, value)

    for r in rects:
        h, w, x, y = r.get_height(), r.get_width(), r.get_x(), r.get_y()

        ax.annotate(str(round(h, 3)), xy=(x+w/2, y+h/2), xytext=(0, 0),
                    textcoords="offset points", fontsize=16,
                    ha='center', va='center',
                    color='white', weight='bold', clip_on=True)
