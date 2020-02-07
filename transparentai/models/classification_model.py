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
    And you can also find out model bias with `plot_bias` function.
    """

    def __init__(self, model):
        """
        Parameters
        ----------
        model:
            a classifier model that have a `predict` and `predict_proba` functions
        """
        super().__init__(model=model)
        self.threshold_df = None

    def compute_scores(self, X, y, threshold=0.5):
        """
        Compute all predictions, probalities and scores
        if it's binary classifier you can custom the probability threshold

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            feature samples
        y: array-like of shape (n_samples,) or (n_samples, n_outputs)
            true labels for X.
        threshold: float (optional) default=0.5
            only for binary classifier, custome threshold probability 
            for the prediction
        """
        self.X = X
        self.y_true = y
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
        Current metrics : accuracy, confusion_matrix, f1_score, precision, recall & roc_auc

        .. _skearn Model evaluation: https://scikit-learn.org/0.15/modules/model_evaluation.html
        """
        scores = {}
        scores['accuracy'] = accuracy_score(self.y_true, self.y_preds)
        scores['confusion_matrix'] = confusion_matrix(
            self.y_true, self.y_preds)

        average = 'micro'
        scores['f1'] = f1_score(self.y_true, self.y_preds, average=average)
        scores['precision'] = precision_score(
            self.y_true, self.y_preds, average=average)
        scores['recall'] = recall_score(
            self.y_true, self.y_preds, average=average)

        roc_auc, roc_curves = dict(), dict()
        if self.n_classes > 2:
            y = label_binarize(self.y_true, classes=list(
                range(0, self.n_classes)))
            for i in range(0, self.n_classes):
                roc_auc[i] = roc_auc_score(y[:, i], self.y_proba[:, i])
                roc_curves[i] = roc_curve(y[:, i], self.y_proba[:, i])
        else:
            roc_auc[0] = roc_auc_score(self.y_true, self.y_proba[:, 1])
            roc_curves[0] = roc_curve(self.y_true, self.y_proba[:, 1])

        scores['roc_auc'] = [v for k, v in roc_auc.items()]
        scores['roc_curve'] = [v for k, v in roc_curves.items()]

        self.scores_dict = scores

    def plot_scores(self):
        """
        Display different charts for all the metrics : 
        - a dataframe for accuracy, f1, precision, recall & roc_auc
        - confusion matrix
        - ROC curve 
        - Probalities distribution

        Raises
        ------
        """
        fun = plots.plot_classification_scores
        self.plot_overall_scores(fun=fun, preds=self.y_proba)

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
        start: int, default=0
            minimum probability to compare 
            (should be between 0 and 1 and less than end value)
        end: int, default=1
            minimum probability to compare 
            (should be between 0 and 1 and greater than start value)
        step: float, default=0.05
            value between each threshold steps

        Raises
        ------
        """
        if self.n_classes > 2:
            raise Exception(
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
        """
        if self.n_classes > 2:
            raise Exception(
                'This function can be only called for binary classification.')
        if self.threshold_df is None:
            raise ValueError('Use compute_threshold() function first.')

        display(Markdown(
            '## Proba threshold comparison for accuracy, f1 score, precision & recall'))

        plots.compare_threshold_predictions(self.threshold_df, self.y_true)
