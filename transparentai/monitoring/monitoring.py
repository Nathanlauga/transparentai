import numpy as np
import pandas as pd
import warnings

from ..models import ClassificationModel, RegressionModel
from ..datasets import StructuredDataset
from ..fairness import DatasetBiasMetric, ModelBiasMetric
from ..monitoring import monitoring_plots as plots


class Monitoring():
    """
    This class allows to see if your model is still with the same performance than
    just after training and gives you some insight with graphics about the current performance 
    and dataset / model bias.  

    Example
    -------
    >>> from transparentai.monitoring import Monitoring
    >>> monitoring = Monitoring(X=new_X, y_preds=new_y_preds, 
                                y_real=new_y_real,
                                privileged_groups=privileged_groups,
                                alert_threshold=alert_threshold,
                                model_type='classification')
 
    For more details please see the `Monitoring for binary classification`_ or
    `Montioring for regression`_ notebooks.

    .. _Monitoring for binary classification : https://github.com/Nathanlauga/transparentai/notebooks/example_monitoring_binary_classification.ipynb
    .. _Montioring for regression : https://github.com/Nathanlauga/transparentai/notebooks/example_monitoring_regression.ipynb

    Attributes
    ----------
    X: pd.DataFrame
        New X data that were used to get the predictions
    y_preds: np.array or pd.Series
        Predictions using X parameters
    y_real: np.array or pd.Series (optional)
        Real output
    model_type: str
        'classification' or 'regression'
    orig_metrics: dict(dict)
        Dictionary with metrics group in keys ('performance', 'bias_dataset', 'bias_model') 
        and original metric values in values
    new_metrics: dict(dict)
        Dictionary with metrics group in keys ('performance', 'bias_dataset', 'bias_model') 
        and new metric values in values
    privileged_groups: dict(list())
        Dictionary with protected attributes as keys (e.g. gender) and
        list of value(s) that are considered privileged (e.g. ['Male'])
    alert_threshold: dict (optional)
        Dictionary with metrics in keys and
        threshold values in values
    dateset: transparentai.datasets.StructuredDataset
        Dataset with X and either y_preds if y_real is None or 
        y_real if not.
    """

    def __init__(self, X, y_preds, y_real=None, model_type='classification',
                 orig_metrics=None, privileged_groups=None, alert_threshold=None, mean=None):
        """
        Parameters
        ----------
        X: pd.DataFrame
            New X data that were used to get the predictions
        y_preds: np.array or pd.Series
            Predictions using X parameters
        y_real: np.array or pd.Series (optional)
            Real output
        model_type: str
            'classification' or 'regression'
        orig_metrics: dict(dict)
            Dictionary with metrics group in keys ('performance', 'bias_dataset', 'bias_model') 
            and original metric values in values
        privileged_groups: dict(list())
            Dictionary with protected attributes as keys (e.g. gender) and
            list of value(s) that are considered privileged (e.g. ['Male'])
        alert_threshold: dict (optional)
            Dictionary with metrics in keys and
            threshold values in values
        mean: number
            Mean of target (for regression) to get the same label than
            original metrics for bias.

        Raises
        ------
        ValueError
            Only regression and classification are handled  for model_type.
        TypeError
            X has to be a pandas dataframe
        ValueError
            y_preds and X must have the same length
        ValueError
            y_real and X must have the same length
        ValueError
            Valid keys for orig_metrics are 'performance', 'bias_dataset' or 'bias_model'.
        """
        if model_type not in ['classification', 'regression']:
            raise ValueError(
                'Only regression and classification are handled  for model_type.')
        if type(X) not in [pd.DataFrame]:
            raise TypeError('X has to be a pandas dataframe')
        if len(X) != len(y_preds):
            raise ValueError('y_preds and X must have the same length')
        if y_real is not None:
            if len(X) != len(y_real):
                raise ValueError('y_real and X must have the same length')
        if orig_metrics is not None:
            metrics_keys = ['performance', 'bias_dataset', 'bias_model']
            if not any([k in metrics_keys for k in orig_metrics]):
                raise ValueError(
                    "Valid keys for orig_metrics are 'performance', 'bias_dataset' or 'bias_model'.")

        self.X = X
        self.y_preds = y_preds
        self.y_real = y_real
        self.model_type = model_type
        self.orig_metrics = orig_metrics
        self.privileged_groups = privileged_groups
        self.alert_threshold = alert_threshold

        df = X.copy().reset_index(drop=True)
        if type(y_preds) in [pd.Series, pd.DataFrame]:
            y_preds = y_preds.values
        if y_real is not None:
            if type(y_real) in [pd.Series, pd.DataFrame]:
                y_real = y_real.values

        df['target'] = y_preds if y_real is None else y_real

        target_regr = model_type == 'regression'
        self.dataset = StructuredDataset(df=df, target='target', mean=mean, target_regr=target_regr)

        self._compute_new_metrics()

        if orig_metrics is not None:
            self._check_orig_and_new_metrics()

    def _compute_orig_metrics(self, X_orig, y_orig):
        """
        Only if you don't have original metrics already stored
        """
        # Todo
        orig_metrics = {}

        self.orig_metrics = orig_metrics

    def _compute_new_metrics(self):
        """
        Computes new metrics based on given attributes in init.

        If y_real is not None then computes new performance scores.

        If privileged_groups then it also computes dataset bias and if y_real
        is define the model bias
        """
        new_metrics = {}
        # handle only 2 first ?
        # model_bias_metrics = ['Disparate impact', 'Statistical parity difference']
        # model_bias_metrics += ['Equal opportunity difference', 'Average abs odds difference', 'Theil index']

        # I have y_real ==> compute model perf & define model bias to compute
        if (self.y_real is not None):
            if self.model_type == 'classification':
                model_obj = ClassificationModel
            elif self.model_type == 'regression':
                model_obj = RegressionModel

            model = model_obj(X=self.X, y=self.y_real, y_preds=self.y_preds)
            new_metrics['performance'] = model.scores_to_json()

        # I have protected attr ==> compute dataset bias & model bias
        if (self.privileged_groups is not None):

            bias = DatasetBiasMetric(self.dataset, self.privileged_groups)
            new_metrics['bias_dataset'] = bias.metrics_to_json()

            if self.y_real is not None:
                bias = ModelBiasMetric(
                    self.dataset, self.y_preds, self.privileged_groups)
                new_metrics['bias_model'] = bias.metrics_to_json()

        self.new_metrics = new_metrics

    def _check_one_metric(self, metric):
        """
        Compares differents keys of a specified metric.

        Show a warning if a key is in a dictionary but not in the other one.

        Parameters
        ----------
        metric: str
            Metric key to compare in both orig_metrics and
            new_metrics
        """
        if (metric in self.orig_metrics) & (metric in self.new_metrics):
            not_in_new = [k for k in self.orig_metrics[metric]
                          if k not in self.new_metrics[metric]]
            not_in_orig = [k for k in self.new_metrics[metric]
                           if k not in self.orig_metrics[metric]]
            for k in not_in_new:
                warnings.warn(
                    f"In original {metric} dict '{k}' key is in but not in the new one.")
            for k in not_in_orig:
                warnings.warn(f"In new {metric} dict '{k}' key is in but not in the orignal one.")

    def _check_orig_and_new_metrics(self):
        """
        Checks that performance metrics in both new and original dictionaries
        are the same.
        """
        if (self.orig_metrics is None) | (self.new_metrics is None):
            return
        # Check performance
        self._check_one_metric(metric='performance')

    def plot_perfomance(self):
        """
        Plot performance with one bar chart for each metric.

        If `orig_metrics` is not None but `new_metrics` is, then plots only original 
        performance metrics.
        If `orig_metrics` is None but `new_metrics` is not, then plots only new 
        performance metrics.
        If neither `orig_metrics` and `new_metrics` are not None, then plots
        comparison between original metrics and new ones.
        """
        if (self.orig_metrics is None) & (self.y_real is None):
            raise ValueError(
                "Either original metrics or y_real has to bas set in init to show performance")
        if self.orig_metrics is not None:
            if ('performance' not in self.orig_metrics) & (self.y_real is None):
                raise ValueError(
                    "'performance' has to be in orig_metrics dict")

        if 'performance' not in self.new_metrics:
            orig_perf = self.orig_metrics['performance']
            plots.plot_bar_performance(orig_perf=orig_perf,
                                       alert_threshold=self.alert_threshold)

        elif self.orig_metrics is None:
            new_perf = self.new_metrics['performance']
            plots.plot_bar_performance(
                new_perf=new_perf, alert_threshold=self.alert_threshold)

        else:
            orig_perf = self.orig_metrics['performance']
            new_perf = self.new_metrics['performance']
            plots.plot_bar_performance(
                orig_perf, new_perf, self.alert_threshold)

    def _plot_bias(self, bias_key='bias_dataset', attr=None, target_value=None):
        """
        Plot bias with one gauge plot for each metric.
        
        If `attr` and `target_value` it reduces the number of plots 
        Example : if you have 2 target values (e.g. 0 and 1) and different protected
        attributes (e.g. gender and marital status) then you can say that it will
        only plots for gender and target value = 1 

        If `orig_metrics` is not None but `new_metrics` is, then plots only original 
        performance metrics.
        If `orig_metrics` is None but `new_metrics` is not, then plots only new 
        performance metrics.
        If neither `orig_metrics` and `new_metrics` are not None, then plots
        comparison between original metrics and new ones.

        Parameters
        ----------
        bias_key: str (default 'bias_dataset')
            Key for bias metrics
            Can be 'bias_dataset' or 'bias_model'
        attr: str (optional)
            Name of the attribute to analyse
        target_value: (optional)
            Specific label value 
        """
        if (self.orig_metrics is None) & (self.y_real is None):
            raise ValueError(
                "Either original metrics or y_real has to bas set in init to show bias")
        if self.orig_metrics is not None:
            if self.privileged_groups is None:
                raise ValueError(
                    f"privileged_groups object attribute was not set at the init")

            if (bias_key not in self.orig_metrics) & (self.y_real is None):
                raise ValueError(
                    f"'{bias_key}' has to be in orig_metrics dict when y_real is not set")

        if bias_key not in self.new_metrics:
            orig_bias = self.orig_metrics[bias_key]
            plots.plot_gauge_bias(attr, target_value, orig_bias=orig_bias)

        elif self.orig_metrics is None:
            new_bias = self.new_metrics[bias_key]
            plots.plot_gauge_bias(attr, target_value, new_bias=new_bias)

        else:
            orig_bias = self.orig_metrics[bias_key]
            new_bias = self.new_metrics[bias_key]
            plots.plot_gauge_bias(attr, target_value, orig_bias, new_bias)

    def plot_dataset_bias(self, attr=None, target_value=None):
        """
        Plot bias with one gauge plot for each dataset bias metric.

        Parameters
        ----------
        attr: str (optional)
            Name of the attribute to analyse
        target_value: (optional)
            Specific label value 
        """
        self._plot_bias(bias_key='bias_dataset', attr=attr,
                        target_value=target_value)

    def plot_model_bias(self, attr=None, target_value=None):
        """
        Plot bias with one gauge plot for each model bias metric.

        Parameters
        ----------
        attr: str (optional)
            Name of the attribute to analyse
        target_value: (optional)
            Specific label value 
        """
        self._plot_bias(bias_key='bias_model', attr=attr,
                        target_value=target_value)
