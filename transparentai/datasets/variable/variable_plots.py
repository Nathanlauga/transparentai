import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from transparentai import utils
from transparentai import plots
from .variable import describe


DEFAULT_COLOR = '#3498db'


def plot_table_describe(ax, cell_text):
    """Insert a table in a matplotlib graphic
    using an axis.

    Parameters
    ----------
    ax: plt.axes.Axes
        axe where to add the plot 
    cell_text: list(list)
        The texts to place into the table cells.
    """

    table = ax.table(cellText=cell_text, cellLoc='left', loc='center')

    table[(0, 0)].get_text().set_color('green')
    table[(1, 0)].get_text().set_color('red')

    for i in range(len(cell_text)):
        table[(i, 0)].get_text().set_weight('bold')

    table.set_fontsize(24)
    table.scale(1, 3)

    ax.axis('off')


def plot_number_var(ax, arr, color=DEFAULT_COLOR, label=None, alpha=1.):
    """Plots an histogram into an matplotlib axe.

    Parameters
    ----------
    ax: plt.axes.Axes
        axe where to add the plot 
    arr: array like
        Array of number values
    color: str (default DEFAULT_COLOR)
        color of the plot
    label: str (default None)
        label of the plot
    alpha: float (default 1.)
        opacity

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

    ax.hist(arr, bins=50, color=color, label=label, alpha=alpha)


def plot_datetime_var(ax, arr, color=DEFAULT_COLOR, label=None, alpha=1.):
    """Plots a line plot into an matplotlib axe.

    Parameters
    ----------
    ax: plt.axes.Axes
        axe where to add the plot 
    arr: array like
        Array of datetime values
    color: str (default DEFAULT_COLOR)
        color of the plot
    label: str (default None)
        label of the plot
    alpha: float (default 1.)
        opacity

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

    arr = pd.to_datetime(arr, errors='coerce')

    date_min = arr.min()
    date_max = arr.max()
    gap = (date_max - date_min).days

    if gap > 1500:
        arr = arr.dt.year.astype(str)
    elif gap > 100:
        arr = arr.dt.strftime('%Y-%m')
    elif gap > 5:
        arr = arr.dt.strftime('%Y-%m-%d')
    else:
        arr = arr.dt.strftime('%Y-%m-%d-%r')

    v_c = arr.value_counts().sort_index()
    dates = mdates.num2date(mdates.datestr2num(v_c.index))
    y = v_c.values

    ax.plot(dates, y, color=color, label=label)
    ax.fill_between(dates, 0, y, color=color, alpha=alpha)


def plot_object_var(ax, arr, top=10, color=DEFAULT_COLOR, label=None, alpha=1.):
    """Plots a bar plot into an matplotlib axe.

    Parameters
    ----------
    ax: plt.axes.Axes
        axe where to add the plot 
    arr: array like
        Array of object values
    color: str (default DEFAULT_COLOR)
        color of the plot
    label: str (default None)
        label of the plot
    alpha: float (default 1.)
        opacity

    Raises
    ------
    TypeError:
        arr is not an array like
    TypeError:
        arr is not a object array
    """
    if not utils.is_array_like(arr):
        raise TypeError('arr is not an array like')
    if utils.find_dtype(arr) != 'object':
        raise TypeError('arr is not an object array')

    if type(arr) in [list, np.ndarray]:
        arr = pd.Series(arr)

    v_c = arr.value_counts().sort_values(ascending=False)

    v_c = v_c if len(v_c) <= top else v_c[:top]
    x, y = v_c.index, v_c.values

    bar = ax.bar(x, y, color=color, label=label, alpha=alpha)


def plot_variable(arr, legend=None, colors=None, xlog=False, ylog=False, **kwargs):
    """Plots a graph with two parts given an array.
    First part is the plot custom plot depending on the array dtype.
    Second part is the describe statistics table.

    First plot is:

    - Histogram if dtype is number (using plot_number_var)
    - Line plot if dtype is datetime (using plot_datetime_var)
    - Bar plot  if dtype is object (using plot_object_var)

    If legend array is set then automaticly plots differents values.

    Parameters
    ----------
    arr: array like
        Array of values to plots
    legend: array like (default None)
        Array of values of legend (same length than arr)
    colors: list (default None)
        Array of colors, used if legend is set
    xlog: bool (default False)
        Scale xaxis in log scale
    ylog: bool (default False)
        Scale yaxis in log scale

    Raises
    ------
    TypeError:
        arr is not an array like
    TypeError:
        legend is not an array like
    ValueError:
         arr and legend have not the same length
    """
    if not utils.is_array_like(arr):
        raise TypeError('arr is not an array like')
    if (legend is not None) & (not utils.is_array_like(legend)):
        raise TypeError('legend is not an array like')
    if legend is not None:
        if len(arr) != len(legend):
            raise ValueError('arr and legend have not the same length')

    name = ''
    if type(arr) == pd.Series:
        name = arr.name
    elif type(arr) == pd.DataFrame:
        name = arr.columns[0]
    elif type(arr) in [list, np.ndarray]:
        arr = pd.Series(arr)

    if (legend is not None) & (colors is None):
        colors = ['#3498db', '#e67e22', '#2ecc71',
                  '#f1c40f', '#9b59b6', '#e74c3c']

    if legend is not None:
        legend_name = ''
        if type(legend) == pd.Series:
            legend_name = legend.name
        elif type(legend) == pd.DataFrame:
            legend_name = legend.columns[0]

    dtype = utils.find_dtype(arr)
    desc = describe(arr)

    # Init figure
    fig = plt.figure(figsize=(15, 5), constrained_layout=False)
    gs = fig.add_gridspec(1, 12)

    # 2 axes : one for the plot, one for the stats
    ax1 = fig.add_subplot(gs[0, :8])
    ax2 = fig.add_subplot(gs[0, 8:])

    # format title
    title = 'Histogram' if dtype == 'number' else 'Plot'
    title = title if name is None else title + ' of ' + name
    title = title if legend is None else title + ' by ' + legend_name

    ax1.set_title(title, loc='center', fontsize=22)

    # Use plot depending on the dtype
    # Number : histogram
    if dtype == 'number':
        plot_fun = plot_number_var

    # Datetime : line plot
    elif dtype == 'datetime':
        plot_fun = plot_datetime_var
        fig.autofmt_xdate()

    # Object : bar plot
    else:
        plot_fun = plot_object_var
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=18)

    if legend is None:
        plot_fun(ax1, arr)
    else:
        for i, label in enumerate(set(list(legend))):
            arr_val = arr[legend == label]

            if len(arr_val) == 0:
                continue

            plot_fun(ax1, arr_val,
                     color=colors[i % len(colors)],
                     label=label,
                     alpha=0.5)

    # If log is needed
    if xlog:
        ax1.set_xscale('log')
    if ylog:
        ax1.set_yscale('log')

    # put legend if it's necessary
    if legend is not None:
        ax1.legend(loc=0, frameon=True)

    # Add describe stats table
    desc_formated = utils.format_describe_str(desc)
    plot_table_describe(ax2, desc_formated)

    # plt.show()
    return plots.plot_or_figure(fig, **kwargs)
