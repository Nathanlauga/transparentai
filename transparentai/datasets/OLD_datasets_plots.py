import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown

from transparentai import plots


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

    plots.plot_or_save(fname='missing_values.png')


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
    target: str (default None)
        Target column for classifier
    """
    rows = 1 if target is None else 2
    fig, ax = plt.subplots(figsize=(16, rows*5))

    ax = plt.subplot(int(f'{rows}21'))
    if target == None:
        sns.distplot(df[var].dropna())
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

    sns.boxplot(x=x, y=df[var].dropna())

    if target != None:
        tab = pd.crosstab(df[var], df[target])

        ax = plt.subplot(223)
        plots.plot_stack(ax=ax, df=tab, labels=labels)

        tab = tab.div(tab.sum(axis=1), axis=0)

        ax = plt.subplot(224)
        plots.plot_stack(ax=ax, df=tab, labels=labels)

    plots.plot_or_save(fname=f'{var}_variable_plot.png')


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
    target: str (default None)
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
        plots.plot_stack_bar(ax=ax, df=tab, labels=labels,
                             legend_labels=legend_labels)

    plots.plot_or_save(fname=f'{var}_variable_plot.png')


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
    target: str (default None)
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
        plots.plot_stack(ax=ax, df=tab, labels=legend_labels)

    plots.plot_or_save(fname=f'{var}_variable_plot.png')


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
    target: str (default None)
        Target column for classifier
    """
    if df[var1].nunique() <= 1:
        display(Markdown(f'{var1} has only one value'))
        return
    if df[var2].nunique() <= 1:
        display(Markdown(f'{var2} has only one value'))
        return

    if target is not None:
        if (min(df[target].value_counts()) <= 1) | (df[target].nunique() == 1):
            display(
                Markdown(f'{target} has not enough data to be represented'))
            target = None

    if target == None:
        g = sns.jointplot(var1, var2, data=df, space=0, height=8)
    else:
        legend_labels = sorted(df[target].unique())
        cols = [var1, var2, target] if target is not None else [var1, var2]
        df = df[cols].dropna()
        grid = sns.JointGrid(x=var1, y=var2, data=df, height=7)

        g = grid.plot_joint(sns.scatterplot, hue=target, data=df, alpha=0.3)
        for l in legend_labels:
            df_v1 = df.loc[df[target] == l, var1]
            df_v2 = df.loc[df[target] == l, var2]
            sns.distplot(df_v1, ax=g.ax_marg_x, kde=(df_v1.nunique() > 1))
            sns.distplot(df_v2, ax=g.ax_marg_y, kde=(
                df_v2.nunique() > 1), vertical=True)

    plots.plot_or_save(fname=f'{var1}_{var2}_variable_jointplot.png')


def plot_one_cat_and_num_variables(df, var1, var2, target=None):
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
    target: str (default None)
        Target column for classifier
    """
    if df[var1].nunique() <= 1:
        display(Markdown(f'{var1} has only one value'))
        return
    if df[var2].nunique() <= 1:
        display(Markdown(f'{var2} has only one value'))
        return

    val_cnt = df[var1].value_counts()
    if len(val_cnt) > 10:
        val_cnt = val_cnt.head(10)

    df_plot = df[df[var1].apply(lambda x: x in val_cnt.index.values)]

    fig, ax = plt.subplots(figsize=(16, 5))
    palette = "Blues" if target == None else "colorblind"
    sns.boxplot(x=var1, y=var2, hue=target,
                data=df_plot, palette=palette)
    plt.xticks(rotation=40)

    plots.plot_or_save(fname=f'{var1}_{var2}_variable_boxplot.png')


def plot_correlation_matrix(corr_df, fname='correlation_matrix_plot.png'):
    """
    Plot a seaborn heatmap based on a correlation dataframe.

    Parameters
    ----------
    corr_df: pd.DataFrame
        Correlation dataframe
    fname: str
        File path where to save the plot if plots.save_plot if True
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
    plots.plot_or_save(fname=fname)
