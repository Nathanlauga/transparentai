import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.image as mpimg
from IPython.display import display, Markdown

from transparentai.fairness.protected_attribute import ProtectedAttribute

BIAS_COLORS = ['#3498db', '#ecf0f1']


def plot_text_center(ax, text, fontsize=18):
    """
    Display text at the center of an ax from a matplotlib figure.

    Parameters
    ----------
    ax: plt.axes.Axes
        ax where to set centered text
    text: str
        text to display
    fontsize: int (optionnal)
        font size of the text
    """
    left, width = .25, .5
    bottom, height = .25, .5
    right = left + width
    top = bottom + height

    ax.text(0.5 * (left + right), 0.5 * (bottom + top), text,
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=fontsize,
            transform=ax.transAxes)

    ax.axis('off')


def plot_percentage_bar_man_img(ax, freq, spacing=0):
    """
    """
    heights = [70 for i in range(0, 10)]

    for i, height in enumerate(heights):
        img_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '../src/img/'))
        img_path += '/_man_blue.png' if i < round(
            freq, 1)*10 else '/_man_grey.png'
        img = mpimg.imread(img_path)
        # AR = img.shape[1] / img.shape[0]
        AR = 135 / 256
        width = height * AR
        left = width*i + spacing*i
        right = left + width
        ax.imshow(img, extent=[left, right, 0, height])

    ax.set_xlim(0, right)
    ax.set_ylim(0, max(heights)*1.1)
    ax.axis('off')


def plot_pie(ax, freq):
    """
    """
    sizes = [freq, 1-freq]
    ax.pie(sizes, labels=['', ''], colors=BIAS_COLORS,
           shadow=True, startangle=130)
    ax.axis('equal')


def plot_protected_attr_row(axes, protected_attr, target_value=None, privileged=True):
    """
    """
    n_total = protected_attr.num_instances(privileged=privileged)
    n = protected_attr.num_spec_value(
        target_value=target_value, privileged=privileged)
    freq = n / n_total
    title = 'Privileged' if privileged else 'Unprivileged'

    plot_percentage_bar_man_img(ax=axes[0], freq=freq)
    axes[0].set_title(title, loc='left', fontsize=18)
    plot_text_center(ax=axes[1], text='=', fontsize=25)
    plot_pie(ax=axes[2], freq=freq)

    text1 = '%.2f%%' % (freq*100)
    text2 = f'\n\n\n\n({n} / {n_total})'
    plot_text_center(ax=axes[3], text=text1, fontsize=32)
    plot_text_center(ax=axes[3], text=text2, fontsize=14)


def setup_bottom_line(ax):
    """
    code from : https://matplotlib.org/3.1.1/gallery/ticks_and_spines/tick-locators.html#sphx-glr-gallery-ticks-and-spines-tick-locators-py
    """
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.yaxis.set_major_locator(ticker.NullLocator())
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(which='major', width=1.00, length=5)
    ax.tick_params(which='minor', width=0.75, length=2.5, labelsize=10)
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 1)
    ax.patch.set_alpha(0.0)
    ax.xaxis.set_major_locator(ticker.NullLocator())
    ax.xaxis.set_minor_locator(ticker.NullLocator())


def set_legend_protected_attr(fig, axes, protected_attr, target_value):
    """
    """
    target = protected_attr.target
    attr_name = protected_attr.name

    text = f'Focus on {attr_name} for {target} is {target_value}'
    unp_text = r'$\bf{Unprivileged}$' + \
        f' means {protected_attr.get_unprivileged_values()}'
    pri_text = r'$\bf{Privileged}$' + \
        f' : means {protected_attr.get_privileged_values()}'

    setup_bottom_line(axes[0])
    axes[0].text(0.0, 0.9, text, fontsize=22,
                 transform=axes[0].transAxes, fontweight='bold')
    axes[0].text(0.0, 0.5, unp_text, fontsize=16,
                 transform=axes[0].transAxes, fontweight='normal')
    axes[0].text(0.0, 0.1, pri_text, fontsize=16,
                 transform=axes[0].transAxes, fontweight='normal')

    text_blue = f'{target} is {target_value}'
    text_grey = f'{target} is not {target_value}'
    fig.legend(labels=[text_blue, text_grey], fontsize=18)


def get_protected_attr_freq(protected_attr, target_value, privileged=None):
    """
    """
    n_total = protected_attr.num_instances(privileged=privileged)
    n = protected_attr.num_spec_value(
        target_value=target_value, privileged=privileged)
    return n / n_total


def plot_metrics_text(axes, protected_attr, target_value):
    """
    """
    metrics = protected_attr.metrics
    freq_unpr = get_protected_attr_freq(
        protected_attr=protected_attr, target_value=target_value, privileged=False)
    freq_priv = get_protected_attr_freq(
        protected_attr=protected_attr, target_value=target_value, privileged=True)

    for i, metric in enumerate(metrics.columns):
        text = ''
        if metric == 'Disparate impact':
            text = r'$\frac{%.2f}{%.2f}$ = %.2f' % (
                freq_unpr, freq_priv, metrics.loc[target_value, metric])
        elif metric == 'Statistical parity difference':
            text = r'$%.2f - %.2f$ = %.2f' % (freq_unpr*100,
                                              freq_priv*100, metrics.loc[target_value, metric]*100)

        plot_text_center(ax=axes[i], text=text, fontsize=32)

        axes[i].set_title(metric, fontsize=24)
        axes[i].axis('off')


def plot_metrics_label(protected_attr, target_value):
    """
    """
    fig = plt.figure(constrained_layout=True, figsize=(16, 10))
    gs = fig.add_gridspec(6, 10)

    leg_ax1 = fig.add_subplot(gs[0, :])
    unp_ax1 = fig.add_subplot(gs[1:3, :-5])
    unp_ax2 = fig.add_subplot(gs[1:3, -5])
    unp_ax3 = fig.add_subplot(gs[1:3, 6:8])
    unp_ax4 = fig.add_subplot(gs[1:3, 8:10])
    pri_ax1 = fig.add_subplot(gs[3:5:, :-5])
    pri_ax2 = fig.add_subplot(gs[3:5, -5])
    pri_ax3 = fig.add_subplot(gs[3:5, 6:8])
    pri_ax4 = fig.add_subplot(gs[3:5, 8:10])
    met_ax1 = fig.add_subplot(gs[5, 0:5])
    met_ax2 = fig.add_subplot(gs[5, 5:])

    axes_legend = [leg_ax1]
    axes_unpriv = [unp_ax1, unp_ax2, unp_ax3, unp_ax4]
    axes_priv = [pri_ax1, pri_ax2, pri_ax3, pri_ax4]
    axes_metrics = [met_ax1, met_ax2]

    plot_protected_attr_row(axes_unpriv, protected_attr,
                            target_value, privileged=False)
    plot_protected_attr_row(axes_priv, protected_attr,
                            target_value, privileged=True)

    set_legend_protected_attr(fig, axes_legend, protected_attr, target_value)

    plot_metrics_text(axes=axes_metrics,
                      protected_attr=protected_attr, target_value=target_value)

    plt.show()


def plot_dataset_metrics(protected_attr, target_value=None):
    """
    """
    if not isinstance(protected_attr, ProtectedAttribute):
        raise TypeError("Must be a ProtectedAttribute object!")

    if target_value != None:
        plot_metrics_label(protected_attr=protected_attr,
                           target_value=target_value)
    else:
        for target_value in protected_attr.labels.unique():
            plot_metrics_label(protected_attr=protected_attr,
                               target_value=target_value)
