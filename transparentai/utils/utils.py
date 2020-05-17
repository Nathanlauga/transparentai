import numpy as np
import pandas as pd
import warnings

from sklearn.preprocessing import LabelEncoder


def find_dtype(arr, len_sample=1000):
    """Find the general dtype of an array.
    Three possible dtypes :

    - Number
    - Datetime
    - Object

    Parameters
    ----------
    arr: array-like
        Array to inspect
    len_sample: int (default, 1000)
        Number max of items to analyse
        if len_sample > len(arr) then use len(arr)

    Returns
    -------
    str:
        dtype string ('number', 'datetime' or 'object')

    Raises
    ------
    TypeError:
        arr is not an array like
    """
    if not is_array_like(arr):
        raise TypeError('arr is not an array like')

    if type(arr) in [list, np.ndarray]:
        arr = pd.DataFrame(arr)
    elif type(arr) == pd.Series:
        arr = arr.to_frame()
        
    n = len_sample if len(arr) > len_sample else len(arr)
    arr = arr.iloc[:n]

    is_number = arr.select_dtypes('number').shape[1] > 0
    is_datetime = arr.select_dtypes(['datetime', 'datetimetz']).shape[1] > 0

    if is_number:
        return 'number'

    elif is_datetime:
        return 'datetime'

    try:
        pd.to_datetime(arr)
        return 'datetime'
    except:
        pass

    return 'object'


def is_array_like(obj, n_dims=1):
    """Returns whether an object is an array like.
    Valid dtypes are list, np.ndarray, pd.Series, pd.DataFrame.

    Parameters
    ----------
    obj:
        Object to inspect
    n_dims: int (default 1)
        number of dimension accepted

    Returns
    -------
    bool:
        Whether the object is an array like or not
    """

    dtype = type(obj)

    valid_types = [list, np.ndarray, pd.Series, pd.DataFrame]
    if dtype not in valid_types:
        return False

    if dtype == list:
        obj = np.array(obj)
    elif dtype != np.ndarray:
        obj = obj.to_numpy()

    if len(obj.shape) <= n_dims:
        return type(obj[0]) != list

    return obj.shape[n_dims] == 1


def format_describe_str(desc, max_len=20):
    """Returns a formated list for the matplotlib table
    cellText argument.

    Each element of the list is like this : ['key    ','value    ']

    Number of space at the end of the value depends on 
    len_max argument.

    Parameters
    ----------
    desc: dict
        Dictionnary returned by the variable.describe
        function
    len_max: int (default 20)
        Maximum length for the values

    Returns
    -------
    list(list):
        Formated list for the matplotlib table
        cellText argument
    """
    res = {}
    _max = max([len(str(e)) for k, e in desc.items()])
    max_len = _max if _max < max_len else max_len

    n_valid = desc['valid values']
    n_missing = desc['missing values']
    n = n_valid + n_missing

    for k, e in desc.items():
        if k == 'valid values':
            e = str(e) + ' (' + str(int(n_valid*100/n)) + '%)'
        elif k == 'missing values':
            e = str(e) + ' (' + str(int(n_missing*100/n)) + '%)'
        else:
            e = str(e)
        e = e.ljust(max_len) if len(e) <= 15 else e[:max_len]
        res[k.ljust(15).title()] = e

    return [[k, e] for k, e in res.items()]


def preprocess_metrics(input_metrics, metrics_dict):
    """Preprocess the inputed metrics so that it maps
    with the appropriate function in metrics_dict global variable.

    input_metrics can have str or function. If it's a string
    then it has to be a key from metrics_dict global variable dict

    Returns a dictionnary with metric's name as key and 
    metric function as value

    Parameters
    ----------
    input_metrics: list
        List of metrics to compute
    metrics_dict: dict
        Dictionnary to compare input_metrics with

    Returns
    -------
    dict:
        Dictionnary with metric's name as key and 
        metric function as value

    Raises
    ------
    TypeError:
        input_metrics must be a list
    """
    if type(input_metrics) != list:
        raise TypeError('input_metrics must be a list')

    fn_dict = {}
    cnt_custom = 1

    for fn in input_metrics:
        if type(fn) == str:
            if fn in metrics_dict:
                fn_dict[fn] = metrics_dict[fn]
            else:
                warnings.warn('%s function not found' % fn)
        else:
            fn_dict['custom_'+str(cnt_custom)] = fn
            cnt_custom += 1

    if len(fn_dict.keys()) == 0:
        raise ValueError('No valid metrics found')

    return fn_dict


def init_corr_matrix(columns, index, fill_diag=1.):
    """Returns a matrix n by m fill of 0 (except on the diagonal if squared matrix)
    Recommended for correlation matrix

    Parameters
    ----------
    columns: 
        list of column names
    index:
        list of index names
    fill_diag: float (default 1.)
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


def encode_categorical_vars(df):
    """Encodes categorical variables from a dataframe to be numerical (discrete)
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


def object_has_function(obj, fn):
    return bool(getattr(obj, fn, None))
