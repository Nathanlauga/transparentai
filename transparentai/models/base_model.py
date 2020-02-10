import pandas as pd
from IPython.display import display, Markdown
from abc import abstractmethod

import transparentai.models.models_plots as plots
import transparentai.utils as utils

class BaseModel():
    """    
    """
    
    scores = None
    scores_dict = None
    model_type = None

    def __init__(self, model):
        """
        Parameters
        ----------
        model:
            a classifier model that have a `predict` and `predict_proba` functions
        """
        self.model = model
        
    @abstractmethod
    def compute_scores(self):
        return None
    
    @abstractmethod
    def scores(self):
        return None 

    @abstractmethod
    def scores_to_json(self):
        return None 

    def _scores_to_json(self, scores_names):
        """
        """
        scores_json = {}
        for k, v in self.scores_dict.items():
            if k in scores_names:
                scores_json[k] = v
        return scores_json

    def save_scores(self, fname):
        """
        """
        scores_json = self.scores_to_json()
        utils.save_dict_to_json(obj=scores_json, fname=fname)
        
    def display_scores(self):
        """
        Display current scores computed by `compute_scores` function.
        """
        if self.scores is None:
            raise ValueError('Use compute_scores() function first.')
        
        scores_to_display = None
        if self.model_type == 'classification':
            scores_to_display = ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']
        elif self.model_type == 'regression':
            scores_to_display = ['MAE','MSE','RMSE','R2']
        
        scores = {k: v for k, v in self.scores_dict.items() if k in scores_to_display}
        scores = pd.Series(scores).to_frame().T
        scores.index = ['score']
        
        display(scores)
        
    @abstractmethod
    def plot_scores(self):
        return None

    def plot_overall_scores(self, fun, preds):
        """
        Display different charts for all the metrics.
        
        Raises
        ------
        """
        if self.scores is None:
            raise ValueError('Use compute_scores() function first.')
            
        display(Markdown('### Overall model performance'))
        self.display_scores()
        
        fun(self.scores_dict, self.y_true, preds)