import pandas as pd
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown

from transparentai import utils
from transparentai import plots


def show_missing_values(df):
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
    plt.xticks(x, x, rotation=60)
    plt.xlabel('Columns')
    plt.ylabel('Percentage')
    plt.title('Percentage of missing values in columns')
    plt.show()


def show_numerical_var(df, var, target=None):
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
    fig, ax = plt.subplots(figsize=(16, 5))

    ax = plt.subplot(121)
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

    ax = plt.subplot(122)
    x = df[target] if target != None else None

    sns.boxplot(x=x, y=df[var])

    if target != None:
        fig, ax = plt.subplots(figsize=(16, 5))
        tab = pd.crosstab(df[var], df[target])

        ax = plt.subplot(121)
        plots.plot_stack(ax=ax, tab=tab, labels=labels)

        tab.div(tab.sum(axis=1), axis=0)

        ax = plt.subplot(122)
        plots.plot_stack(ax=ax, tab=tab, labels=labels)

    plt.show()


def show_categorical_var(df, var, target=None):
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

    fig, ax = plt.subplots(figsize=(16, 5))

    ax = plt.subplot(121)
    ax.bar(labels, sizes, width=0.8)
    plt.xticks(labels, labels, rotation=60)

    ax = plt.subplot(122)
    ax.pie(sizes, labels=labels, colors=colors,
           autopct='%1.0f%%', shadow=True, startangle=130)
    ax.axis('equal')
    ax.legend(loc=0, frameon=True)

    if target != None:
        fig, ax = plt.subplots(figsize=(16, 5))

        legend_labels = sorted(df[target].unique())
        tab = pd.crosstab(df[var], df[target])
        tab = tab.loc[labels]
        plots.plot_stack_bar(ax=ax, tab=tab, labels=labels,
                               legend_labels=legend_labels)

    plt.show()


def show_datetime_var(df, var, target=None):
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
    fig, ax = plt.subplots(figsize=(16, 5))

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

    plt.show()


def show_meta_var(df, var):
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
    most_freq = df[var].value_counts().head(2).to_dict()
    display(Markdown(
        f'**{var} :** {nb_null} nulls, {nb_uniq} unique vals, most common: {most_freq}'))


def show_df_vars(df, target=None):
    """
    Show all variables with graphics to understand each variable.
    If target is set, complement visuals will be added to take a look on the
    influence that a variable can have on target

    Data type handle : categorical, numerical, datetime

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe to inspect
    target: str (optional)
        Target column for classifier 
    """
    cat_vars = df.select_dtypes(['object','category'])
    num_vars = df.select_dtypes('number')
    dat_vars = df.select_dtypes('datetime')

    display(Markdown('### Numerical variables'))
    for var in num_vars:
        display(Markdown(''))
        show_meta_var(df, var)
        if len(df[var].unique()) <= 1:
            display('Only one value.')
            continue
        show_numerical_var(df, var, target)

    display(Markdown('### Categorical variables'))
    for var in cat_vars:
        display(Markdown(''))
        show_meta_var(df, var)
        if len(df[var].unique()) <= 1:
            display('Only one value.')
            continue
        show_categorical_var(df, var, target)

    display(Markdown('### Datetime variables'))
    for var in dat_vars:
        display(Markdown(''))
        show_meta_var(df, var)
        if len(df[var].unique()) <= 1:
            display('Only one value.')
            continue
        show_datetime_var(df, var, target)


def show_df_numerical_relations(df, target=None):
    """
    Show all numerical variables 2 by 2 with graphics understand their relation.
    If target is set, separate dataset for each target value.

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe to inspect
    target: str (optional)
        Target column for classifier 
    """
    num_vars = df.select_dtypes('number')
    num_vars = utils.remove_var_with_one_value(num_vars)

    cols = num_vars.columns.values
    var_combi = [tuple(sorted([v1, v2]))
                 for v1 in cols for v2 in cols if v1 != v2]
    var_combi = list(set(var_combi))

    for var1, var2 in var_combi:
        display(Markdown(''))
        display(Markdown(f'Joint plot for **{var1}** & **{var2}**'))
        plots.plot_numerical_jointplot(
            df=df, var1=var1, var2=var2, target=target)


def show_df_num_cat_relations(df, target=None):
    """
    Show boxplots for each pair of categorical and numerical variables
    If target is set, separate dataset for each target value.

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe to inspect
    target: str (optional)
        Target column for classifier 
    """
    df = df.copy()
    df = utils.remove_var_with_one_value(df)

    num_vars = df.select_dtypes('number')
    cat_vars = df.select_dtypes(['object','category'])

    var_combi = [(v1, v2) for v1 in num_vars.columns for v2 in cat_vars.columns if (
        v1 != v2) & (v2 != target)]

    for num_var in num_vars.columns:
        display(Markdown(''))
        display(Markdown(f'Box plot for **{target}** & **{num_var}**'))
        plots.plot_barplot_cat_num_var(
            df=df, cat_var=target, num_var=num_var, target=None)


    for num_var, cat_var in var_combi:
        display(Markdown(''))
        display(Markdown(f'Box plot for **{cat_var}** & **{num_var}**'))
        plots.plot_barplot_cat_num_var(
            df=df, cat_var=cat_var, num_var=num_var, target=target)


def show_df_correlations(df):
    """
    Show differents correlations matrix for 3 cases :
    - numerical to numerical (using Pearson coeff)
    - categorical to categorical (using Cramers V & Chi square)
    - numerical to categorical (discrete) (using Point Biserial)

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe to inspect
    """
    df = df.copy()
    df = utils.remove_var_with_one_value(df)

    num_df = df.select_dtypes('number')
    cat_df = df.select_dtypes(['object','category'])
    num_vars = num_df.columns
    cat_vars = cat_df.columns

    ignore_cat_vars = list()
    for var in cat_vars:
        if cat_df[var].nunique() > 100:
            ignore_cat_vars.append(var)
    cat_vars = [v for v in cat_vars if v not in ignore_cat_vars]

    print('Ignored categorical variables because there are more than 100 values :', ', '.join(ignore_cat_vars))

    pearson_corr = num_df.corr()
    display(Markdown('#### Pearson correlation matrix for numerical variables'))
    plots.plot_correlation_matrix(pearson_corr)

    var_combi = [tuple(sorted([v1, v2]))
                 for v1 in cat_vars for v2 in cat_vars if v1 != v2]
    var_combi = list(set(var_combi))

    cramers_v_corr = utils.init_corr_matrix(columns=cat_vars, index=cat_vars)

    for var1, var2 in var_combi:
        corr = utils.cramers_v(cat_df[var1], cat_df[var2])
        cramers_v_corr.loc[var1, var2] = corr
        cramers_v_corr.loc[var2, var1] = corr

    display(Markdown('#### Cramers V correlation matrix for categorical variables'))
    plots.plot_correlation_matrix(cramers_v_corr)

    data_encoded = utils.encode_categorical_vars(df)
    # pearson_corr = data_encoded.corr()
    # display(Markdown('#### Pearson correlation matrix for categorical variables'))
    # plots.plot_correlation_matrix(pearson_corr)

    var_combi = [(v1, v2) for v1 in cat_vars for v2 in num_vars if v1 != v2]

    pbs_corr = utils.init_corr_matrix(
        columns=num_vars, index=cat_vars, fill_diag=0.)
    
    for cat_var, num_var in var_combi:
        corr, p_value = ss.pointbiserialr(
            data_encoded[cat_var], data_encoded[num_var])
        pbs_corr.loc[cat_var, num_var] = corr

    display(Markdown(
        '#### Point Biserial correlation matrix for numerical & categorical variables'))
    plots.plot_correlation_matrix(pbs_corr)
