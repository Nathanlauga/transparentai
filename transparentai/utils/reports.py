from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

from transparentai.datasets import variable
from transparentai.models import classification, explainers, regression
from transparentai import utils
from datetime import datetime


def generate_head_page(document_title):
    """Generate a figure with a given title.

    Parameters
    ----------
    document_title: str
        Name of the document

    Returns
    -------
    matplotlib.figure.Figure:
        Document head figure
    """
    fig = plt.figure(figsize=(8.27, 11.69))

    plt.text(0.5, 0.75, document_title, fontsize=23,
             ha='center',
             va='center',
             bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=1'))

    date = datetime.today().strftime('%Y-%m-%d')
    plt.text(0.5, 0.65, date, fontsize=15, ha='center', va='center')

    plt.axis('off')

    return fig


def generate_validation_report(model, X, y_true, X_valid=None, y_true_valid=None,
                               metrics=None, model_type='classification', out='validation_report.pdf'):
    """Generate a pdf report on the model performance 
    with the following graphics: 

    - First page with the report title
    - An histogram of the y_true distribution
    - Model performance plot
    - Model feature importance plot

    This function is usefull to keep a proof of the validation.

    Parameters
    ----------
    model:
        Model to analyse
    X: array like
        Features 
    y_true: array like
        True labels
    X_valid: array like
        Features for validation set
    y_true_valid: array like (default None)
        True labels for validation set
    metrics: list (default None)
        List of metrics to plots
    model_type: str (default 'classification')
        'classification' or 'regression'
    out: str (default 'validation_report.pdf')
        path where to save the report

    Raises
    ------
    ValueError:
        'model_type must be 'classification' or 'regression'
    """
    if model_type not in ['classification', 'regression']:
        raise ValueError(
            'model_type must be \'classification\' or \'regression\'')

    if utils.object_has_function(model, 'predict_proba'):
        fn = model.predict_proba
    else:
        fn = model.predict

    y_pred = fn(X)
    if X_valid is not None:
        y_pred_valid = fn(X_valid)
    else:
        y_pred_valid = None

    figs = list()
    document_title = 'Validation report'
    figs.append(generate_head_page(document_title))

    # Plot y_true variable
    print('Generating y_true distribution')
    f = variable.plot_variable(y_true, plot=False)
    figs.append(f)

    if model_type == 'classification':
        module = classification
    else:
        module = regression

    print('Generating model performance')
    f = module.plot_performance(
        y_true, y_pred, y_true_valid, y_pred_valid, plot=False)
    figs.append(f)

    nsamples = 1000
    print('Generating model feature influence (over %i samples)' % nsamples)
    explainer = explainers.ModelExplainer(model, X)

    f = explainer.plot_global_explain(X, nsamples=nsamples, plot=False)
    figs.append(f)

    pp = PdfPages(out)
    for f in figs:
        pp.savefig(f)
    pp.close()
    print('report generated at %s' % out)
