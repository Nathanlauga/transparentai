import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown

from .. import plots

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


def plot_classification_scores(scores, y_true, y_proba):
    """
    Display different charts for all the metrics : 
    - a dataframe for accuracy, f1, precision, recall & roc_auc
    - confusion matrix
    - ROC curve 
    - Probality distribution

    Parameters
    ----------
    scores: pd.DataFrame
        scores dataframe attribute of a `ClassificationModel` object
    y_true: array-like
        true labels
    y_proba: array
        Target scores, can either be probability estimates of the positive class, 
        confidence values, or binary decisions.
    """
    n_classes = len(y_proba[0])
    fig = plt.figure(figsize=(15, 10))

    n_rows = 1 + n_classes
    ax = plt.subplot(int(f'221'))
    plot_confusion_matrix(matrix=scores['confusion_matrix'])

    ax = plt.subplot(int(f'222'))
    plot_roc_curve(roc_curve=scores['roc_curve'],
                   roc_auc=scores['roc_auc'], n_classes=n_classes)

    ax = plt.subplot(int(f'212'))
    plot_class_distribution(
        y_true=y_true, y_proba=y_proba, n_classes=n_classes)

    plots.plot_or_save(fname='classification_scores_plot.png')


def plot_confusion_matrix(matrix):
    """
    Show confusion matrix.

    Parameters
    ----------
    matrix: array
        confusion_matrix metrics result
    """
    sns.heatmap(matrix,
                cmap='Blues',
                square=True,
                fmt='d',
                annot=True)
    plt.xlabel('Predicted')
    plt.ylabel('Real')
    plt.title('Confusion Matrix')


def plot_roc_curve(roc_curve, roc_auc, n_classes):
    """
    Show a roc curve plot with roc_auc score on the legend.

    Parameters
    ----------
    roc_curve: array
        roc_curve metrics result for each class
    roc_auc: array
        roc_auc metrics result for each class       
    n_classes: int
        Number of classes
    """
    fpr, tpr = dict(), dict()

    for v in list(roc_curve.keys()):
        fpr[v] = roc_curve[v][0]
        tpr[v] = roc_curve[v][1]
    lw = 2

    colors = sns.color_palette("colorblind", n_classes)
    for v, color in zip(list(roc_auc.keys()), colors):
        n = 1 if n_classes == 1 else v
        plt.plot(fpr[v], tpr[v], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(n, roc_auc[v]))
        if n_classes <= 2:
            break

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic curve')
    plt.legend(loc="lower right")


def plot_class_distribution(y_true, y_proba, n_classes):
    """
    Show class distribution using seaborn `distplot`_ function.
    if n_classes is 2 then display probability for class 1 only.
    So class 0 probabilities should tend towards 0 and class 1 towards 1.

    else it displays all probabilities for current class so all 
    probabilities should tend towards 1.

    .. _distplot: https://seaborn.pydata.org/generated/seaborn.distplot.html

    Parameters
    ----------
    y_true: array-like
        true labels
    y_proba: array
        Target scores, can either be probability estimates of the positive class, 
        confidence values, or binary decisions.
    n_classes: int
        Number of classes
    """
    df = pd.DataFrame()
    df['y_true'] = y_true
    for idx, v in enumerate(list(df['y_true'].unique())):
        df[v] = [p[idx] for p in y_proba]

    if n_classes == 2:
        df[0] = df[1]

    colors = sns.color_palette("colorblind", n_classes)
    for v in df['y_true'].unique():
        kde = df[df['y_true'] == v][v].nunique() > 1
        sns.distplot(df[df['y_true'] == v][v], label=v, kde=kde)
        plt.legend(loc=0)

    plt.xlabel('Predicted probabilities')


def plot_curve(df, var):
    """
    Plot one curve based on a dataframe with a specific variable

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe to inspect
    var: str
        Variable name inside dataframe
    """
    lw = 2
    plt.plot(df.index, df[var], color='navy', lw=lw, label=var)
    plt.xlabel('threshold')
    plt.ylabel(var)
    plt.legend(loc="lower right")


def compare_threshold_predictions(threshold_df, y_true):
    """
    Display curves for four metrics by threshold on the x axis :
    - accuracy
    - f1 score
    - precision
    - recall

    Parameters
    ----------
    threshold_df: pd.DataFrame
        dataframe with different threshold on columns
    y_true: array like
        true labels        
    """
    df = list()
    for threshold in threshold_df:
        accuracy = accuracy_score(y_true, threshold_df[threshold])
        f1 = f1_score(y_true, threshold_df[threshold])
        precision = precision_score(y_true, threshold_df[threshold])
        recall = recall_score(y_true, threshold_df[threshold])
        df.append([accuracy, f1, precision, recall])

    df = pd.DataFrame(df,
                      columns=['accuracy', 'f1_score',
                               'precision_score', 'recall_score'],
                      index=threshold_df.columns
                      )

    scores = df.columns
    fig = plt.figure(figsize=(15, 10))

    for i in range(0, 4):
        ax = fig.add_subplot(2, 2, i+1)
        plot_curve(df=df, var=scores[i])

    plots.plot_or_save(fname='compare_threshold_predictions_plot.png')
