import numpy as np
import pandas as pd


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

    if type(arr) == list:
        arr = np.array(arr)
    if type(arr) in [pd.Series, pd.DataFrame]:
        arr = arr.to_numpy()

    n = len_sample if len(arr) > len_sample else len(arr)
    arr = arr[:n]

    try:
        arr.astype(int)
        return 'number'
    except:
        pass

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

    return [[k,e] for k,e in res.items()]