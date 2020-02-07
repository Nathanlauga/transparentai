import pandas as pd
from IPython.display import display, Markdown

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import transparentai.models.models_plots as plots
from transparentai.models.base_model import BaseModel


class RegressionModel(BaseModel):
    """
    """

    def __init__(self, model):
        """
        Parameters
        ----------
        model:
            a classifier model that have a `predict` and `predict_proba` functions
        """
        super().__init__(model=model)

    def compute_scores(self, X, y):
        """
        Compute all predictions and scores

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            feature samples
        y: array-like of shape (n_samples,) or (n_samples, n_outputs)
            true labels for X.
        """
        self.X = X
        self.y_true = y
        self.y_preds = self.model.predict(X)

        self.scores()

    def scores(self):
        """
        Compute classification metrics scores based on `skearn Model evaluation`_
        Current metrics : mean_absolute_error, mean_squared_error, root_mean_squared_error & r2_score

        .. _skearn Model evaluation: https://scikit-learn.org/0.15/modules/model_evaluation.html
        """
        scores = {}
        scores['MAE'] = mean_absolute_error(self.y_true, self.y_preds)
        scores['MSE'] = mean_squared_error(self.y_true, self.y_preds)
        scores['RMSE'] = mean_squared_error(self.y_true, self.y_preds, squared=True)
        scores['R2'] = r2_score(self.y_true, self.y_preds)

        self.scores_dict = scores

    def plot_scores(self):
        """
        Display different charts for all the metrics.

        Raises
        ------
        """
        # fun = plots.plot_classification_scores
        # self.plot_overall_scores(fun=fun, preds=self.y_proba)
