import pandas as pd
from IPython.display import display, Markdown

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.preprocessing import label_binarize

import transparentai.models.models_plots as plots
from transparentai.models.base_model import BaseModel


class ClassificationModel(BaseModel):
    """    
    Class to inspect a classification model based on a model which has a `predict` and 
    `predict_proba` functions.
    It could help you to explore your model performance and validate or not your model.

    Example
    -------
    
    >>> from transparentai.models import ClassificationModel
    >>> model = ClassificationModel(model=clf)
    >>> model.compute_scores(X=X_test, y=y_test, threshold=0.5)

    If you want to analyse false positives or false negatives for example
    you can use the following code (also available for true positives and negatives)
    
    >>> FP_df = model.get_false_positives()
    >>> FN_df = model.get_false_negatives()

    For more details please see the `ClassificationModel for binary classification`_ or
    `ClassificationModel for multi labels classification`_ notebooks.

    .. _ClassificationModel for binary classification : https://github.com/Nathanlauga/transparentai/notebooks/example_ClassificationModel_binary_classification.ipynb
    .. _ClassificationModel for multi labels classification : https://github.com/Nathanlauga/transparentai/notebooks/example_ClassificationModel_multi_label_classification.ipynb

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
        String that indicates what model type it is : 'classification'
    n_classes: int
        Number of classes inside y
    threshold_df: pd.DataFrame
        Dataframe with different probability thresholds as columns
        and their prediction (0 or 1), it can be use only for
        binary classification
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
        self.threshold_df = None
        self.y_proba = None
        self.model_type = 'classification'

        if (X is not None) & (y is not None) & (y_preds is not None):
            self.n_classes = len(list(set(y)))
            self.scores()

    def compute_scores(self, X=None, y=None, threshold=0.5):
        """
        Compute all predictions, probalities and scores
        if it's binary classifier you can custom the probability threshold

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            feature samples
        y: array-like of shape (n_samples,) or (n_samples, n_outputs)
            true labels for X.
        threshold: float (default 0.5)
            only for binary classifier, custome threshold probability 
            for the prediction

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
        self.n_classes = len(list(set(y)))

        self.y_proba = self.model.predict_proba(X)

        if self.n_classes == 2:
            self.y_preds = (self.y_proba[:, 1] >= threshold).astype(int)
            self.compute_threshold(X=X, y=y)
        else:
            self.y_preds = self.model.predict(X)

        self.scores()

    def scores(self):
        """
        Compute classification metrics scores based on `skearn Model evaluation`_
        Current metrics : accuracy, confusion_matrix, f1_score, precision, recall and roc_auc

        .. _skearn Model evaluation: https://scikit-learn.org/0.15/modules/model_evaluation.html
        """
        scores = {}
        scores['accuracy'] = accuracy_score(self.y_true, self.y_preds)
        scores['confusion_matrix'] = confusion_matrix(
            self.y_true, self.y_preds)

        average = 'micro' if self.n_classes != 2 else 'weighted'
        scores['f1'] = f1_score(self.y_true, self.y_preds, average=average)
        scores['precision'] = precision_score(
            self.y_true, self.y_preds, average=average)
        scores['recall'] = recall_score(
            self.y_true, self.y_preds, average=average)

        if self.y_proba is not None:
            roc_auc, roc_curves = dict(), dict()
            if self.n_classes > 2:
                y = label_binarize(self.y_true, classes=self.y_true.unique())
                for i, v in enumerate(list(self.y_true.unique())):
                    try:
                        roc_auc[int(v)] = roc_auc_score(y[:, i], self.y_proba[:, i])
                        roc_curves[int(v)] = roc_curve(y[:, i], self.y_proba[:, i])
                    except ValueError:
                        roc_auc[int(v)] = 1
                        roc_curves[int(v)] = [[1,1], [0,1]]
                        pass
            else:
                roc_auc[0] = roc_auc_score(self.y_true, self.y_proba[:, 1])
                roc_curves[0] = roc_curve(self.y_true, self.y_proba[:, 1])

            scores['roc_auc'] = roc_auc # [v for k, v in roc_auc.items()]
            scores['roc_curve'] = roc_curves # [v for k, v in roc_curves.items()]

        self.scores_dict = scores

    def get_true_positives(self):
        """
        Return true positives rows in the feature sample X

        Returns
        -------
        pd.DataFrame
            True positives rows 
        """
        return self.X[(self.y_preds == 1) & (self.y_true == 1)]

    def get_true_negatives(self):
        """
        Return true negatives rows in the feature sample X

        Returns
        -------
        pd.DataFrame
            True negatives rows 
        """
        return self.X[(self.y_preds == 0) & (self.y_true == 0)]

    def get_false_positives(self):
        """
        Return false positives rows in the feature sample X

        Returns
        -------
        pd.DataFrame
            False positives rows 
        """
        return self.X[(self.y_preds == 1) & (self.y_true == 0)]

    def get_false_negatives(self):
        """
        Return false negatives rows in the feature sample X

        Returns
        -------
        pd.DataFrame
            False negatives rows 
        """
        return self.X[(self.y_preds == 0) & (self.y_true == 1)]

    def plot_scores(self):
        """
        Display different charts for all the metrics : 

        - a dataframe for accuracy, f1, precision, recall & roc_auc
        - confusion matrix
        - ROC curve 
        - Probalities distribution
        """
        fun = plots.plot_classification_scores
        self.plot_overall_scores(fun=fun, preds=self.y_proba)

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
        scores_names = ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']
        return self._scores_to_json(scores_names)

    def compute_threshold(self, X, y, start=0, end=1, step=0.05):
        """
        Compute all predictions, probalities and scores
        if it's binary classifier you can custom the probability threshold

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            feature samples
        y: array-like of shape (n_samples,) or (n_samples, n_outputs)
            true labels for X.
        start: int, (default 0)
            minimum probability to compare 
            (should be between 0 and 1 and less than end value)
        end: int, (default 1)
            minimum probability to compare 
            (should be between 0 and 1 and greater than start value)
        step: float, (default 0.05)
            value between each threshold steps

        Raises
        ------
        ValueError:
            This function can be only called for binary classification.
        ValueError:
            start has to be smaller than end
        ValueError:
            start has to be between 0 and 1
        ValueError:
            end has to be between 0 and 1
        """
        if self.n_classes > 2:
            raise ValueError(
                'This function can be only called for binary classification.')
        if start > end:
            raise ValueError('start has to be smaller than end')
        if (start < 0) or (start > 1):
            raise ValueError('start has to be between 0 and 1')
        if (end < 0) or (end > 1):
            raise ValueError('end has to be between 0 and 1')

        threshold = start
        threshold_df = pd.DataFrame()

        proba = self.model.predict_proba(X)

        while threshold <= end:
            threshold = round(threshold, 2)
            preds = (proba[:, 1] >= threshold).astype(int)

            threshold_df[threshold] = preds
            threshold += step

        self.threshold_df = threshold_df

    def plot_threshold(self):
        """
        Display curves for four metrics by threshold on x axis :

        - accuracy
        - f1 score
        - precision
        - recall

        Raises
        ------
        ValueError:
            This function can be only called for binary classification.
        ValueError:
            compute_threshold() function has to be compute first.
        """
        if self.n_classes > 2:
            raise ValueError(
                'This function can be only called for binary classification.')
        if self.threshold_df is None:
            raise ValueError('Use compute_threshold() function first.')

        display(Markdown(
            '## Proba threshold comparison for accuracy, f1 score, precision & recall'))

        plots.compare_threshold_predictions(self.threshold_df, self.y_true)
