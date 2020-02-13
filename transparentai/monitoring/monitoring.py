import numpy as np
import pandas as pd
import warnings

from transparentai.models import ClassificationModel, RegressionModel
from transparentai.datasets import StructuredDataset
from transparentai.fairness import DatasetBiasMetric, ModelBiasMetric
import transparentai.monitoring.monitoring_plots as plots


class Monitoring():
    """
    """

    def __init__(self, X, y_preds, y_real=None, model_type='classification',
                 orig_metrics=None, privileged_groups=None, alert_threshold=None, mean=None):
        """
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
                    "Valid keys are 'performance', 'bias_dataset' or 'bias_model'.")

        self.X = X
        self.y_preds = y_preds
        self.y_real = y_real
        self.model_type = model_type
        self.orig_metrics = orig_metrics
        self.privileged_groups = privileged_groups
        self.alert_threshold = alert_threshold

        df = X.copy()
        df['target'] = y_preds if y_real is None else y_real
        self.dataset = StructuredDataset(df=df, target='target', mean=mean)

        self._compute_new_metrics()

        if orig_metrics is not None:
            self._check_orig_and_new_metrics()

    def compute_orig_metrics(self, X_orig, y_orig):
        """
        Only if you don't have original metrics already stored
        """
        # Todo
        orig_metrics = {}

        self.orig_metrics = orig_metrics

    def _compute_new_metrics(self):
        """
        """
        new_metrics = {}
        # handle only 2 first ?
#         model_bias_metrics = ['Disparate impact', 'Statistical parity difference']
#         model_bias_metrics += ['Equal opportunity difference', 'Average abs odds difference', 'Theil index']

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
        """
        if (metric in self.orig_metrics) & (metric in self.new_metrics):
            not_in_new = [k for k in self.orig_metrics[metric]
                          if k not in self.new_metrics[metric]]
            not_in_orig = [k for k in self.new_metrics[metric]
                           if k not in self.orig_metrics[metric]]
            for k in not_in_new:
                warnings.warn(
                    f"In original {metric} dict '{k}' key is in but not in the new one.", Warning)
            for k in not_in_orig:
                warnings.warn(
                    f"In new {metric} dict '{k}' key is in but not in the orignal one.", Warning)

    def _check_orig_and_new_metrics(self):
        """
        """
        if (self.orig_metrics is None) | (self.new_metrics is None):
            return
        # Check performance
        self._check_one_metric(metric='performance')

    def plot_perfomance(self):
        """
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
        """
        if (self.orig_metrics is None) & (self.y_real is None):
            raise ValueError(
                "Either original metrics or y_real has to bas set in init to show bias")
        if self.orig_metrics is not None:
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
        """
        self._plot_bias(bias_key='bias_dataset', attr=attr,
                        target_value=target_value)

    def plot_model_bias(self, attr=None, target_value=None):
        """
        """
        self._plot_bias(bias_key='bias_model', attr=attr,
                        target_value=target_value)
