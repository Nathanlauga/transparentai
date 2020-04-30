__all__ = [
    'describe_number',
    'describe_datetime',
    'describe_object',
    'describe'
]

import pandas as pd
import numpy as np
from scipy import stats

from transparentai import utils


def describe_common(arr):
    """Common descriptive statistics about an array.

    Returned statistics:

    - Count of valid values
    - Count of missing values

    Parameters
    ----------
    arr: array like
        Array of value to get desriptive statistics from

    Raises
    ------
    TypeError:
        arr is not an array like
    """
    if not utils.is_array_like(arr):
        raise TypeError('arr is not an array like')
    if type(arr) in [list, np.ndarray]:
        arr = pd.Series(arr)

    n = len(arr)
    missing_val = arr.isna().sum()

    desc = {}
    desc['valid values'] = n - missing_val
    desc['missing values'] = missing_val
    return desc


def describe_number(arr):
    """Descriptive statistics about a number array.

    Returned statistics:

    - Count of valid values
    - Count of missing values
    - Mean
    - Mode
    - Min
    - Quantitle 25%
    - Median
    - Quantile 75%
    - Max

    Parameters
    ----------
    arr: array like
        Array of value to get desriptive statistics from

    Raises
    ------
    TypeError:
        arr is not an array like
    TypeError:
        arr is not a number array
    """
    if not utils.is_array_like(arr):
        raise TypeError('arr is not an array like')
    if utils.find_dtype(arr) != 'number':
        raise TypeError('arr is not a number array')

    desc = describe_common(arr)

    desc['mean'] = np.round(np.mean(arr), 4)
    desc['mode'] = stats.mode(arr)[0][0]
    desc['min'] = np.min(arr)
    desc['quantile 25%'] = np.quantile(arr, 0.25)
    desc['quantile 50%'] = np.median(arr)
    desc['quantile 75%'] = np.quantile(arr, 0.75)
    desc['max'] = np.max(arr)

    return desc


def describe_datetime(arr, format='%Y-%m-%d'):
    """Descriptive statistics about a datetime array.

    Returned statistics:

    - Count of valid values
    - Count of missing values
    - Count of unique values
    - Most common value
    - Min
    - Mean
    - Max

    Parameters
    ----------
    arr: array like
        Array of value to get desriptive statistics from
    format: str
        String format for datetime value

    Raises
    ------
    TypeError:
        arr is not an array like
    TypeError:
        arr is not a datetime array
    """
    if not utils.is_array_like(arr):
        raise TypeError('arr is not an array like')
    if utils.find_dtype(arr) != 'datetime':
        raise TypeError('arr is not a datetime array')

    if type(arr) in [list, np.ndarray]:
        arr = pd.Series(arr)

    arr = pd.to_datetime(arr, errors='coerce')

    desc = describe_common(arr)

    desc['unique values'] = arr.nunique()
    desc['most common'] = arr.mode()[0].strftime(format)
    desc['min'] = arr.min().strftime(format)
    desc['mean'] = arr.mean().strftime(format)
    desc['max'] = arr.max().strftime(format)

    return desc


def describe_object(arr):
    """Descriptive statistics about an object array.

    Returned statistics:

    - Count of valid values
    - Count of missing values
    - Count of unique values
    - Most common value

    Parameters
    ----------
    arr: array like
        Array of value to get desriptive statistics from

    Raises
    ------
    TypeError:
        arr is not an array like
    TypeError:
        arr is not an object array
    """
    if not utils.is_array_like(arr):
        raise TypeError('arr is not an array like')
    if utils.find_dtype(arr) != 'object':
        raise TypeError('arr is not an object array')

    if type(arr) in [list, type(np.array([]))]:
        arr = pd.Series(arr)

    desc = describe_common(arr)

    desc['unique values'] = arr.nunique()
    desc['most common'] = arr.mode()[0]

    return desc


def describe(arr):
    """Descriptive statistics about an array.
    Depending on the detected dtype (number, date, object)
    it returns specific stats.

    Common statistics for all dtype (using describe_common):

    - Count of valid values
    - Count of missing values

    Number statistics (using describe_number):

    - Mean
    - Mode
    - Min
    - Quantitle 25%
    - Median
    - Quantile 75%
    - Max

    Datetime statistics (using describe_datetime):

    - Count of unique values
    - Most common value
    - Min
    - Mean
    - Max

    Object statistics (using describe_datetime):

    - Count of unique values
    - Most common value

    Parameters
    ----------
    arr: array like
        Array of value to get desriptive statistics from

    Returns
    -------
    dict
        Dictionnary with descriptive statistics

    Raises
    ------
    TypeError:
        arr is not an array like
    """
    if not utils.is_array_like(arr):
        raise TypeError('arr is not an array like')

    if type(arr) == list:
        arr = np.array(arr)
    if type(arr) in [pd.Series, pd.DataFrame]:
        arr = arr.to_numpy()

    dtype = utils.find_dtype(arr)

    if dtype == 'number':
        return describe_number(arr)
    elif dtype == 'object':
        return describe_object(arr)
    elif dtype == 'datetime':
        return describe_datetime(arr)

    return None
