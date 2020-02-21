import os
import pandas as pd
import numpy as np
import scipy.stats as ss
from sklearn.preprocessing import LabelEncoder
import json


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
    Removes dataset's columns that only contain one unique value.

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
    It uses LabelEncoder classes from scikit-learn

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
    cat_vars = df.select_dtypes(['object', 'category']).columns
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
    Returns the Cramer V value of two categorical variables using
    chi square. This correlation metric is between 0 and 1.

    Code source found in this article : 
    https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9

    Parameters
    ----------
    x: array like
        first categorical variable
    y: array like
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
        list of column names
    index:
        list of index names
    fill_diag: float
        if squared matrix, then set diagonal with this value

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


def regression_to_classification(df, target, mean):
    """
    Convert a dataframe for regression to classification by 
    computing if the value is above the average.

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe to update
    target: str
        Target column that has to be a numerical column

    Returns
    -------
    pd.DataFrame
        Dataframe updated with target column as a category variable
    np.array
        Original values of the target column

    Raises
    ------
    ValueError:
        `target` parameter has to be a numeric variable
    """
    if target not in df.select_dtypes('number').columns:
        raise ValueError('target column is not a number')

    orig_target_val = df[target].values
    df[target] = np.where(orig_target_val > mean, f'>{mean}', f'<={mean}')
    df[target] = df[target].astype('category')

    return df, orig_target_val


def labelencoder_to_dict(encoder):
    """
    Convert a LabelEncoder class from scikit-learn 
    to a dictionary with index as key and original value as value

    Example:
    encoder.classes\_ is ['Male', 'Female']
    returns {0:'Male', 1:'Female'}

    Parameters
    ----------
    encoder: LabelEncoder
        encoder fitted

    Returns
    -------
    dict:
        encoder transformed

    Raises
    ------
    TypeError:
        `encoder` has to be a `LabelEncoder` 
        from `sklearn.preprocessing` module
    """
    if type(encoder) != LabelEncoder:
        raise TypeError('encoder has to be a LabelEncoder.')

    feat_classes = encoder.classes_
    classes_dict = {}
    for i, val in enumerate(feat_classes):
        classes_dict[i] = val

    return classes_dict


def get_metric_goal(metric):
    """
    Returns bias metric goal given metric name.

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
    else:
        return 0

def save_dict_to_json(obj, fname):
    """
    Save a dictionary object to json file.

    Parameters
    ----------
    obj: dict
        dictionary to save
    fname: str
        string of the file path (including filename)
    """
    with OpenFile(fname, mode='w') as file:
        json.dump(obj, file)
    file.close()

def reduce_df_nrows(df, nrows):
    """
    Returns a dataframe reduced of n rows using `sample()` method.

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe to reduce
    nrows: int
        Number of rows to keep
    """
    if type(df) != pd.DataFrame:
        raise TypeError('df has to be a pandas DataFrame object')

    if len(df) <= nrows:
        return df

    df_filter = df.sample(nrows)
    return df_filter