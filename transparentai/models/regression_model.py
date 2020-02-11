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

    def __init__(self, model=None, X=None, y=None, y_preds=None):
        """
        Parameters
        ----------
        model:
            a classifier model that have a `predict` and `predict_proba` functions
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
        """
        if (X is None) & (self.X is None):
            raise ValueError('X is mandatory to compute scores')
        if (y is None) & (self.y_true is None):
            raise ValueError('y is mandatory to compute scores')
        if self.model is None:
            raise ValueError('model attribute was not set at the init step')

        self.X = X if X is not None else self.X
        self.y_true = y if y is not None else self.y_true
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

    
    def scores_to_json(self):
        """
        """
        scores_names = ['MAE', 'MSE', 'RMSE', 'R2']
        return self._scores_to_json(scores_names)

    def plot_scores(self):
        """
        Display different charts for all the metrics.

        Raises
        ------
        """
        # fun = plots.plot_classification_scores
        # self.plot_overall_scores(fun=fun, preds=self.y_proba)
