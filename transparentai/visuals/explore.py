"""
File with all visuals functions for explore submodule.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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
    fig, ax = plt.subplots(figsize=(10, 9))
    ax = sns.heatmap(
        corr_df,
        vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(220, 20, n=200),
        square=True,
        annot=True
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    )
    plt.show()
