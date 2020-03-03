import pandas as pd
import numpy as np
from IPython.display import display, Markdown
from abc import abstractmethod

from ..models import models_plots as plots
from .. import utils


class BaseModel():
    """    
    Base class for different model types. 

    It allows to inspect a model based on a model which has a `predict` and 
    `predict_proba` (if classification) functions.

    It could help you to explore your model performance and validate or not your model.

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
        String that indicates what model type it is.
        Can be 'classification' or 'regression'
    """

    scores_dict = None
    model_type = None

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
        self.model = model
        self.X = X
        self.y_true = y
        self.y_preds = y_preds

    def _scores_to_json(self, scores_names):
        """
        Formats scores_dict attributes to be read as a json object.

        If 'roc_auc' is in the keys then it computes the score mean
        for the different target values 

        Returns
        -------
        dict:
            Scores dict formated for json
        """
        scores_json = {}
        for k, v in self.scores_dict.items():
            if k in scores_names:
                if k == 'roc_auc':
                    scores_json[k] = np.mean([val for label, val in v.items()])
                else:
                    scores_json[k] = v
        return scores_json

    def save_scores(self, fname):
        """
        Saves metric scores dictionary to a json file.

        Parameters
        ----------
        fname: str 
            string of the file path (including filename)
        """
        scores_json = self.scores_to_json()
        utils.save_dict_to_json(obj=scores_json, fname=fname)

    def display_scores(self):
        """
        Display current scores computed by `compute_scores()` function.

        Raises
        ------
        ValueError:
            compute_scores() function has to be compute first.
        """
        if self.scores_dict is None:
            raise ValueError('Use compute_scores() function first.')

        scores_to_display = None
        if self.model_type == 'classification':
            scores_to_display = ['accuracy', 'f1',
                                 'precision', 'recall', 'roc_auc']
        elif self.model_type == 'regression':
            scores_to_display = ['MAE', 'MSE', 'RMSE', 'R2']

        scores = {k: v for k, v in self.scores_dict.items()
                  if k in scores_to_display}
        scores = pd.Series(scores).to_frame().T
        scores.index = ['score']

        display(scores)

    def plot_overall_scores(self, fun, preds):
        """
        Display different charts for all the metrics.

        Raises
        ------
        ValueError:
            compute_scores() function has to be compute first.
        """
        if self.scores_dict is None:
            raise ValueError('Use compute_scores() function first.')

        display(Markdown('### Overall model performance'))
        self.display_scores()

        fun(self.scores_dict, self.y_true, preds)

    # Asbtract methods

    @abstractmethod
    def compute_scores(self):
        return None

    @abstractmethod
    def scores(self):
        return None

    @abstractmethod
    def scores_to_json(self):
        return None

    @abstractmethod
    def plot_scores(self):
        return None
