import sklearn.metrics
import numpy as np

# Evaluation function for regressors

def explained_variance(y_true, y_pred, **args):
    """Explained variance score based on the `sklearn.metrics.explained_variance_score`_ function.

    More details here : `Explained variance score`_

    .. _sklearn.metrics.explained_variance_score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.explained_variance_score.html#sklearn.metrics.explained_variance_score
    .. _Explained variance score: https://scikit-learn.org/stable/modules/model_evaluation.html#explained-variance-score
    """
    return sklearn.metrics.explained_variance_score(y_true, y_pred, **args)


def max_error(y_true, y_pred, **args):
    """Max error based on the `sklearn.metrics.max_error`_ function.

    More details here : `Max error`_

    .. _sklearn.metrics.max_error: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.max_error.html#sklearn.metrics.max_error
    .. _Max error: https://scikit-learn.org/stable/modules/model_evaluation.html#max-error
    """
    return sklearn.metrics.max_error(y_true, y_pred, **args)


def mean_absolute_error(y_true, y_pred, **args):
    """Mean absolute error based on the `sklearn.metrics.mean_absolute_error`_ function.

    More details here : `Mean absolute error`_

    .. _sklearn.metrics.mean_absolute_error: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html#sklearn.metrics.mean_absolute_error
    .. _Mean absolute error: https://scikit-learn.org/stable/modules/model_evaluation.html#mean-absolute-error
    """
    return sklearn.metrics.mean_absolute_error(y_true, y_pred, **args)


def mean_squared_error(y_true, y_pred, **args):
    """Mean squared error based on the `sklearn.metrics.mean_squared_error`_ function.

    More details here : `Mean squared error`_

    .. _sklearn.metrics.mean_squared_error: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html#sklearn.metrics.mean_squared_error
    .. _Mean squared error: https://scikit-learn.org/stable/modules/model_evaluation.html#mean-squared-error
    """
    return sklearn.metrics.mean_squared_error(y_true, y_pred, **args)


def root_mean_squared_error(y_true, y_pred, **args):
    """Root mean squared error based on the `sklearn.metrics.mean_squared_error`_ function.

    squared argument is set to False.

    More details here : `Mean squared error`_

    .. _sklearn.metrics.mean_squared_error: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html#sklearn.metrics.mean_squared_error
    .. _Mean squared error: https://scikit-learn.org/stable/modules/model_evaluation.html#mean-squared-error
    """
    return sklearn.metrics.mean_squared_error(y_true, y_pred, squared=False, **args)


def mean_squared_log_error(y_true, y_pred, **args):
    """Mean squared logarithmic error based on the `sklearn.metrics.mean_squared_log_error`_ function.

    More details here : `Mean squared logarithmic error`_

    .. _sklearn.metrics.mean_squared_log_error: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_log_error.html#sklearn.metrics.mean_squared_log_error
    .. _Mean squared logarithmic error: https://scikit-learn.org/stable/modules/model_evaluation.html#mean-squared-log-error
    """
    return sklearn.metrics.mean_squared_log_error(y_true, y_pred, **args)


def median_absolute_error(y_true, y_pred, **args):
    """Median absolute error based on the `sklearn.metrics.median_absolute_error`_ function.

    More details here : `Median absolute error`_

    .. _sklearn.metrics.median_absolute_error: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.median_absolute_error.html#sklearn.metrics.median_absolute_error
    .. _Median absolute error: https://scikit-learn.org/stable/modules/model_evaluation.html#median-absolute-error
    """
    return sklearn.metrics.median_absolute_error(y_true, y_pred, **args)


def r2(y_true, y_pred, **args):
    """R² score, the coefficient of determination 
    based on the `sklearn.metrics.r2_score`_ function.

    More details here : `R² score, the coefficient of determination`_

    .. _sklearn.metrics.r2_score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html#sklearn.metrics.r2_score
    .. _R² score, the coefficient of determination: https://scikit-learn.org/stable/modules/model_evaluation.html#r2-score-the-coefficient-of-determination
    """
    return sklearn.metrics.r2_score(y_true, y_pred, **args)


def mean_poisson_deviance(y_true, y_pred, **args):
    """Mean Poisson deviances based on the `sklearn.metrics.mean_poisson_deviance`_ function.

    More details here : `Mean Poisson, Gamma, and Tweedie deviances`_

    .. _sklearn.metrics.mean_poisson_deviance: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_poisson_deviance.html#sklearn.metrics.mean_poisson_deviance
    .. _Mean Poisson, Gamma, and Tweedie deviances: https://scikit-learn.org/stable/modules/model_evaluation.html#mean-tweedie-deviance
    """
    return sklearn.metrics.mean_poisson_deviance(y_true, y_pred, **args)


def mean_gamma_deviance(y_true, y_pred, **args):
    """Mean Gamma deviance based on the `sklearn.metrics.mean_gamma_deviance`_ function.

    More details here : `Mean Poisson, Gamma, and Tweedie deviances`_

    .. _sklearn.metrics.mean_gamma_deviance: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_gamma_deviance.html#sklearn.metrics.mean_gamma_deviance
    .. _Mean Poisson, Gamma, and Tweedie deviances: https://scikit-learn.org/stable/modules/model_evaluation.html#mean-tweedie-deviance
    """
    return sklearn.metrics.mean_gamma_deviance(y_true, y_pred, **args)
