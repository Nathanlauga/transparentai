import os
import pandas as pd
import numpy as np
import scipy.stats as ss
from sklearn.preprocessing import LabelEncoder


class OpenFile:
    """
    Class that open a file and close it at the end
    Attributes
    ----------
    fname : str
        file name
    mode : str
        mode for open() method
    """

    def __init__(self, fname: str, mode='r'):
        self.fname = fname
        self.mode = mode

    def __enter__(self):
        self.file = open(self.fname, self.mode)
        return self.file

    def __exit__(self, type, value, traceback):
        self.file.close()


def str_to_file(string: str, fpath: str):
    """
    Create a file based on a string.
    Parameters
    ----------
    string: str
        string to write into the file
    fname: str
        file name
    """
    with OpenFile(fpath, 'w') as file:
        file.write(string)
    file.close()


def remove_var_with_one_value(df):
    """
    Remove dataset's columns that only contains one unique value.

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe to inspect

    Returns
    -------
    pd.DataFrame:
        Dataframe without columns with only one unique value
    """
    if len(df) <= 1:
        return df

    drop_cols = list()
    for var in df:
        if df[var].nunique() <= 1:
            drop_cols.append(var)
    return df.drop(columns=drop_cols)


def encode_categorical_vars(df):
    """
    Encode categorical variables from a dataframe to be numerical (discrete)
    It uses LabelEncoder class from scikit-learn

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe to update

    Returns
    -------
    pd.DataFrame:
        Encoded dataframe
    dict:
        Encoders with feature name on keys and
        encoder as value
    """
    cat_vars = df.select_dtypes(['object','category']).columns
    data_encoded = df.copy()
    for var in df.select_dtypes('category').columns:
        data_encoded[var] = data_encoded[var].cat.add_categories('Unknown')

    data_encoded[cat_vars] = data_encoded[cat_vars].fillna('Unknown')

    # Use Label Encoder for categorical columns (including target column)
    encoders = {}
    for feature in cat_vars:
        le = LabelEncoder()
        le.fit(data_encoded[feature].dropna())

        data_encoded[feature] = le.transform(data_encoded[feature])
        encoders[feature] = le

    return data_encoded, encoders


def cramers_v(x, y):
    """
    Function that return the Cramer V value for two categorical variables using
    chi square. This correlation metric is between 0 and 1.

    Code source found on this article : 
    https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9

    Parameters
    ----------
    x:
        first categorical variable
    y:
        second categorical variable

    Returns
    -------
    float:
        Cramer V value
    """
    confusion_matrix = pd.crosstab(x, y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    if min((kcorr-1), (rcorr-1)) == 0:
        return 0
    return np.sqrt(phi2corr/min((kcorr-1), (rcorr-1)))


def init_corr_matrix(columns, index, fill_diag=1.):
    """
    Return a matrix n by m fill of 0 (except on the diagonal if squared matrix)
    Recommended for correlation matrix

    Parameters
    ----------
    columns: 
        list of columns names
    index:
        list of index names
    fill_diag: float
        if squared matrix then set diagonal with this value

    Returns
    -------
    pd.DataFrame
        Initialized matrix
    """
    zeros = np.zeros((len(index), len(columns)), float)
    if len(columns) == len(index):
        rng = np.arange(len(zeros))
        zeros[rng, rng] = fill_diag
    return pd.DataFrame(zeros, columns=columns, index=index)
