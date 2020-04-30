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
