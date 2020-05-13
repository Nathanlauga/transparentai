import matplotlib.pyplot as plt

def plot_or_figure(fig, plot=True):
    """
    
    Parameters
    ----------
    fig: matplotlib.figure.Figure
        figure to plot or to returns
    plot: bool (default True)
        Whether you want to plot a figure or 
        return it

    Returns
    -------
    matplotlib.figure.Figure
        Figure
    """
    if plot:
        fig.show()
    else:
        return fig

def plot_table_score(perf):
    """Insert a table of scores on a
    matplotlib graphic

    Parameters
    ----------
    perf: dict
        Dictionnary with computed score  
    """
    cell_text = [(k,round(v,4)) for (k,v) in perf.items()]
    table = plt.table(cell_text, cellLoc='left', loc='center')

    for i in range(len(cell_text)):
        table[(i, 0)].get_text().set_weight('bold')

    table.set_fontsize(12)
    table.scale(1, 2)
    plt.title('Table of scores', loc='left')

    plt.axis('off')