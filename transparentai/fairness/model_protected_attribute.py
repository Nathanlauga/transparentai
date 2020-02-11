import pandas as pd
import numpy as np

from transparentai.fairness.protected_attribute import ProtectedAttribute


class ModelProtectedAttribute(ProtectedAttribute):
    """
    """

    def __init__(self, dataset, attr, privileged_values, preds, favorable_label):
        """
        """
        super().__init__(dataset, attr, privileged_values, favorable_label)
        self.preds = preds
        self.preds_crosstab = pd.crosstab(
            self.values, self.preds, margins=True)
        self.compute_model_bias_metrics()

    def confusion_matrix(self, privileged=None):
        """
        """
        perf_df = self.to_frame()
        perf_df['preds'] = self.preds
        if privileged is True:
            perf_df = perf_df[perf_df[self.name] == 1]
        elif privileged is False:
            perf_df = perf_df[perf_df[self.name] == 0]

        matrix = pd.crosstab(perf_df[self.target], perf_df['preds'])

        true = perf_df[self.target].values
        pred = perf_df['preds'].values

        for val in matrix.columns.values:
            if val not in matrix.index.values:
                matrix.loc[val, :] = 0
                
        for val in matrix.index.values:
            if val not in matrix.columns.values:
                matrix.loc[:, val] = 0

        return matrix

    def performances(self, target_value, privileged=None):
        """
        """
        other_values = [v for v in list(set(self.preds)) if v != target_value]
        matrix = self.confusion_matrix(privileged=privileged)

        TP = matrix.loc[target_value, target_value]
        FP = sum(matrix.loc[other_values, target_value].values)
        TN = sum(sum(matrix.loc[other_values, other_values].values))
        if type(target_value) == type(int()):
            FN = sum(matrix.iloc[target_value, other_values].values)
        else:
            FN = sum(matrix.loc[target_value, other_values].values)
        return dict(
            TP=TP, FP=FP, TN=TN, FN=FN
        )

    def num_spec_value(self, target_value, privileged=None, predictions=False):
        r"""
        Compute the number of a particular value,
        :math:`P = \sum_{i=1}^n \mathbb{1}[y_i = v]`,
        optionally conditioned on protected attributes.

        Parameters
        ----------
        target_value:
            Specific label value for which it will compute the metric
        privileged (bool, optional):
            Boolean prescribing whether to
            condition this metric on the `privileged_groups`, if `True`, or
            the `unprivileged_groups`, if `False`. Defaults to `None`
            meaning this metric is computed over the entire dataset.
        """
        crosstab = self.crosstab if not predictions else self.preds_crosstab
        if privileged == None:
            return crosstab.loc['All', target_value]
        elif privileged:
            return crosstab.loc[1, target_value]
        else:
            return crosstab.loc[0, target_value]

    def get_freq(self, target_value, privileged=None, predictions=False):
        """
        

        Parameters
        ----------

        Returns
        -------
        float
            Frequency
        """
        n_total = self.num_instances(privileged=privileged)
        
        if not predictions:
            n = self.num_spec_value(
                target_value=target_value, privileged=privileged)
        else:
            n = self.num_spec_value(
                target_value=target_value, privileged=privileged, predictions=True)

        return n / n_total

    def base_rate(self, target_value, privileged=None):
        """
        Compute the base rate, :math:`Pr(Y = 1) = P/(P+N)`, optionally
        conditioned on protected attributes.

        Parameters
        ----------
        target_value:
            Specific label value for which it will compute the metric
        privileged (bool, optional):
                Boolean prescribing whether to
                condition this metric on the `privileged_groups`, if `True`, or
                the `unprivileged_groups`, if `False`. Defaults to `None`
                meaning this metric is computed over the entire dataset.
        Returns
        -------
        float:
            Base rate (optionally conditioned).
        """
        return (self.num_spec_value(target_value=target_value, privileged=privileged, predictions=True)
                / self.num_instances(privileged=privileged))

    def false_positive_rate(self, target_value, privileged=None):
        """
        Return the ratio of true positives to positive examples in the
        dataset, :math:`TPR = TP/P`, optionally conditioned on protected
        attributes.
        Here negatives values correspond to :math:`y != v`
        So false positives values correspond to :math:`y != v & \hat{y} = v`

        target_value:
            Specific label value for which it will compute the metric
        privileged (bool, optional):
                Boolean prescribing whether to
                condition this metric on the `privileged_groups`, if `True`, or
                the `unprivileged_groups`, if `False`. Defaults to `None`
                meaning this metric is computed over the entire dataset.
        """
        other_values = [v for v in list(set(self.preds)) if v != target_value]

        FP = self.performances(target_value=target_value,
                               privileged=privileged)['FP']
        N = 0
        for v in other_values:
            N += self.num_spec_value(target_value=v, privileged=privileged)
        return FP / N

    def true_positive_rate(self, target_value, privileged=None):
        """
        Return the ratio of true positives to positive examples in the
        dataset, :math:`TPR = TP/P`, optionally conditioned on protected
        attributes.
        Here positives values correspond to :math:`y = v`
        So true positives values correspond to :math:`y = v & \hat{y} = v`

        target_value:
            Specific label value for which it will compute the metric
        privileged (bool, optional):
                Boolean prescribing whether to
                condition this metric on the `privileged_groups`, if `True`, or
                the `unprivileged_groups`, if `False`. Defaults to `None`
                meaning this metric is computed over the entire dataset.
        """
        TP = self.performances(target_value=target_value,
                               privileged=privileged)['TP']
        P = self.num_spec_value(
            target_value=target_value, privileged=privileged)
        return TP / P

    def disparate_impact(self, target_value):
        r"""
        Compute the disparate impact for a specific label value

        .. math::
           \frac{Pr(\hat{Y} = v | D = \text{unprivileged})}
           {Pr(\hat{Y} = v | D = \text{privileged})}

        code source inspired from aif360_

        # ClassificationMetric.disparate_impact
        .. _aif360: https://aif360.readthedocs.io/en/latest/_modules/aif360/metrics/classification_metric.html

        Parameters
        ----------
        target_value:
            Specific label value for which it will compute the metric
        """
        return self.ratio(self.base_rate, target_value=target_value)

    def statistical_parity_difference(self, target_value):
        r"""
        Compute the statistical parity difference for a specific label value

        .. math::
           Pr(\hat{Y} = v | D = \text{unprivileged})
           - Pr(\hat{Y} = v | D = \text{privileged})

        code source inspired from aif360_

        # ClassificationMetric.statistical_parity_difference
        .. _aif360: https://aif360.readthedocs.io/en/latest/_modules/aif360/metrics/classification_metric.html

        Parameters
        ----------
        target_value:
            Specific label value for which it will compute the metric
        """
        return self.difference(self.base_rate, target_value=target_value)

    def equal_opportunity_difference(self, target_value):
        r"""
        :math:`TPR_{D = \text{unprivileged}} - TPR_{D = \text{privileged}}`

        code source from aif360_

        # ClassificationMetric.true_positive_rate_difference
        .. _aif360: https://aif360.readthedocs.io/en/latest/_modules/aif360/metrics/classification_metric.html

        Parameters
        ----------
        target_value:
            Specific label value for which it will compute the metric
        """
        return self.difference(self.true_positive_rate, target_value=target_value)

    def average_abs_odds_difference(self, target_value):
        """
        Average of absolute difference in FPR and TPR for unprivileged and
        privileged groups:

        .. math::

           \frac{1}{2}\left[|FPR_{D = \text{unprivileged}} - FPR_{D = \text{privileged}}|
           + |TPR_{D = \text{unprivileged}} - TPR_{D = \text{privileged}}|\right]

        A value of 0 indicates equality of odds.

        code source from aif360_

        # ClassificationMetric.average_abs_odds_difference
        .. _aif360: https://aif360.readthedocs.io/en/latest/_modules/aif360/metrics/classification_metric.html

        Parameters
        ----------
        target_value:
            Specific label value for which it will compute the metric
        """
        return 0.5 * (np.abs(self.difference(self.false_positive_rate, target_value=target_value))
                      + np.abs(self.difference(self.true_positive_rate, target_value=target_value)))

    def theil_index(self, target_value):
        """
        Theil index or Generalized entropy index (with $\alpha=1$) is proposed as a unified individual and
        group fairness measure in [3]_.  With :math:`b_i = \hat{y}_i - y_i + 1`:

        .. math:: \frac{1}{n}\sum_{i=1}^n\frac{b_{i}}{\mu}\ln\frac{b_{i}}{\mu}

        References:
            .. [3] T. Speicher, H. Heidari, N. Grgic-Hlaca, K. P. Gummadi, A. Singla, A. Weller, and M. B. Zafar,
               "A Unified Approach to Quantifying Algorithmic Unfairness: Measuring Individual and Group Unfairness via Inequality Indices,"
               ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2018.


        code source from aif360_

        # ClassificationMetric.theil_index
        .. _aif360: https://aif360.readthedocs.io/en/latest/_modules/aif360/metrics/classification_metric.html

        Parameters
        ----------
        target_value:
            Specific label value for which it will compute the metric
        """
        y_pred = self.preds
        y_true = self.labels
        y_pred = (y_pred == target_value).astype(np.float64)
        y_true = (y_true == target_value).astype(np.float64)
        b = 1 + y_pred - y_true

        return np.mean(np.log((b / np.mean(b))**b) / np.mean(b))

    def compute_model_bias_metrics(self):
        """
        Compute automaticaly all dataset bias metrics for this
        protected attribute.
        """
        metrics_dict = {
            'Disparate impact': self.disparate_impact,
            'Statistical parity difference': self.statistical_parity_difference,
            'Equal opportunity difference': self.equal_opportunity_difference,
            'Average abs odds difference': self.average_abs_odds_difference,
            'Theil index': self.theil_index,
        }

        self.compute_bias_metrics(metrics_dict=metrics_dict)
