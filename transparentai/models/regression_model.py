import pandas as pd
from IPython.display import display, Markdown

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import transparentai.models.models_plots as plots
from transparentai.models.base_model import BaseModel


class RegressionModel(BaseModel):
    """
    Class to inspect a regression model based on a model which has a `predict`  function.
    It could help you to explore your model performance and validate or not your model.

    Example
    -------

    >>> from transparentai.models import ClassificationModel
    >>> model = RegressionModel(model=clf)
    >>> model.compute_scores(X=X_test, y=y_test)

    For more details please see the `RegressionModel with LinearRegression`_ notebook.

    .. _RegressionModel with LinearRegression : https://github.com/Nathanlauga/transparentai/notebooks/example_RegressionModel_regression.ipynb
    
    Attributes
    ----------
    X: pd.DataFrame
        X data that were used to get the predictions
    y_preds: np.array or pd.Series
        Predictions using X parameters
    y_true: np.array or pd.Series (optional)
        Real output
    model: 
        object with at least a `predict` function that returns
        a series or np.array
    scores_dict: dict
        Dictionary with metric name as keys and score value
        as values
    model_type: str
        String that indicates what model type it is : 'regression'
    """

    def __init__(self, model=None, X=None, y=None, y_preds=None):
        """
        Parameters
        ----------
        model:
            a classifier model that have a `predict` and 
            `predict_proba` (if classification) functions
        X: pd.DataFrame (optional)
            X data that will be used to get the predictions
        y: np.array or pd.Series (optional)
            Real output
        y_preds: np.array or pd.Series (optional)
            Predictions using X parameters if model is not set
        """
        super().__init__(model=model, X=X, y=y, y_preds=y_preds)
        self.model_type = 'regression'

        if (X is not None) & (y is not None) & (y_preds is not None):
            self.scores()

    def compute_scores(self, X=None, y=None):
        """
        Compute all predictions and scores

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            feature samples
        y: array-like of shape (n_samples,) or (n_samples, n_outputs)
            true labels for X.

        Raises
        ------
        AttributeError:
            X has to be set in init part or in the parameters
        AttributeError:
            y_true has to be set in init part or in the parameters as y
        AttributeError:
            model attribute was not set at the init step
        """
        if (X is None) & (self.X is None):
            raise AttributeError('X is mandatory to compute scores')
        if (y is None) & (self.y_true is None):
            raise AttributeError('y is mandatory to compute scores')
        if self.model is None:
            raise AttributeError('model attribute was not set at the init step')

        self.X = X if X is not None else self.X
        self.y_true = y if y is not None else self.y_true
        self.y_preds = self.model.predict(X)

        self.scores()

    def scores(self):
        """
        Compute classification metrics scores based on `skearn Model evaluation`_
        Current metrics : mean_absolute_error, mean_squared_error, root_mean_squared_error and r2_score

        .. _skearn Model evaluation: https://scikit-learn.org/0.15/modules/model_evaluation.html
        """
        scores = {}
        scores['MAE'] = mean_absolute_error(self.y_true, self.y_preds)
        scores['MSE'] = mean_squared_error(self.y_true, self.y_preds)
        scores['RMSE'] = mean_squared_error(self.y_true, self.y_preds, squared=True)
        scores['R2'] = r2_score(self.y_true, self.y_preds)

        self.scores_dict = scores

    
    def scores_to_json(self):
        """
        Formats scores_dict attributes to be read as a json object.

        If 'roc_auc' is in the keys then it computes the score mean
        for the different target values 

        Returns
        -------
        dict:
            Scores dict formated for json
        """
        scores_names = ['MAE', 'MSE', 'RMSE', 'R2']
        return self._scores_to_json(scores_names)

    def _plot_scores(self):
        """
        Display different charts for all the metrics.

        """
        # fun = plots.plot_classification_scores
        # self.plot_overall_scores(fun=fun, preds=self.y_proba)
