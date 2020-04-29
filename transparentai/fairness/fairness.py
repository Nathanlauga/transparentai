import warnings

import pandas as pd


def create_privilieged_df(df, privileged_group):
    """Returns a formated dataframe with protected attribute columns
    and whether the row is privileged (1) or not (0).

    example of a privileged_group dictionnary :
    >>> privileged_group = {
    >>>     'gender':['Male'],                # privileged group is man for gender attribute
    >>>     'age': lambda x: x > 30 & x < 55  # privileged group aged between 30 and 55 years old
    >>> }

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe to extract privilieged group from.
    privileged_group: dict
        Dictionnary with protected attribute as key (e.g. age or gender)
        and a list of favorable value (like ['Male']) or a function
        returning a boolean corresponding to a privileged group

    Returns
    -------
    pd.DataFrame
        DataFrame with protected attribute columns
        and whether the row is privileged (1) or not (0)

    Raises
    ------
    TypeError:
        df is not a pandas.DataFrame
    TypeError:
        privileged_group is not a dictionnary
    ValueError:
        privileged_group has not valid keys (in df columns)
    """

    if type(df) != pd.DataFrame:
        raise TypeError('df is not a pandas.DataFrame')
    if type(privileged_group) != dict:
        raise TypeError('privileged_group is not a dictionnary')
    for k in privileged_group:
        if k not in df.columns:
            warnings.warn('%s variable is not in df columns' % k, Warning)
    if all([k not in df.columns for k in privileged_group]):
        raise ValueError('privileged_group has not valid keys (in df columns)')

    res = list()
    for k, e in privileged_group.items():
        if k not in df.columns:
            continue
        if type(e) == list:
            tmp = df[k].isin(e)
        else:
            tmp = df[k].apply(e)
        tmp = tmp.astype(int).values
        res.append(pd.Series(tmp, name=k))

        del tmp

    return pd.concat(res, axis=1)
