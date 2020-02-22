import pandas as pd
import numpy as np

from transparentai.fairness.protected_attribute import ProtectedAttribute


class ModelProtectedAttribute(ProtectedAttribute):
    """
    This class retrieves all informations on a protected attribute on a specific model.
    It computes automatically the `Disparate impact`, `Statistical parity difference`, 
    `Equal opportunity difference`, `Average abs odds difference` and `Theil index` to
    get some insight about bias in the model.

    This class is inspired by the ClassificationMetric_ class from aif360 module but it 
    depends on a unique attribute.

    .. _ClassificationMetric: https://aif360.readthedocs.io/en/latest/modules/metrics.html#classification-metric

    Attributes
    ----------
    name: str
        Attribute name (the column's name in original dataframe)
    target:
        Name of the label (or target) column
    favorable_label:
        A specific value that is has an advantage over the other(s) values
    preds:
        Array with predictions on the dataset  
    privileged_values: list
        List with privileged values inside the column (e.g. ['Male'] for 'gender' attribute)  
    unprivileged_values: list
        List with unprivileged values inside the column (e.g. ['Female'] for 'gender' attribute)  
    values: pd.Series
        Serie with the same length than labels attribute with 0 or 1 values 
        1 if it correspond to a privileged value, 0 if not.
    labels: np.array 
        Array with original label values        
    crosstab: pd.DataFrame
        Crosstab of values and labels
    preds_crosstab:
        Crosstab of values and predicted labels
    """

    def __init__(self, dataset, attr, privileged_values, preds, favorable_label):
        """
        Parameters
        ----------
        dataset: pd.DataFrame
            Dataframe to inspect
        attr: str
            Name of the attribute to analyse (it has to be into dataset columns)
        privileged_values: list
            List with privileged values inside the column (e.g. ['Male'] for 'gender' attribute)  
        preds:
            Array with predictions on the dataset  
        favorable_label:
            A specific value that is has an advantage over the other(s) values
        """
        super().__init__(dataset, attr, privileged_values, favorable_label)
        self.preds = preds
        self.preds_crosstab = pd.crosstab(
            self.values, self.preds, margins=True)
        self.compute_model_bias_metrics()

    def confusion_matrix(self, privileged=None):
        """
        Computes the confusion matrix on all data if privileged is None else
        on whether privileged sample or unprivileged one.

        Parameters
        ----------
        privileged (bool, optional):
            Boolean prescribing whether to
            condition this metric on the `privileged_groups`, if `True`, or
            the `unprivileged_groups`, if `False`. Defaults to `None`
            meaning this metric is computed over the entire dataset.

        Returns
        -------
        pd.DataFrame:
            Confusion matrix
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

        for val in list(set(self.preds)):
            if val not in matrix.columns:
                matrix[val] = 0

        for val in matrix.columns.values:
            if val not in matrix.index.values:
                matrix.loc[val, :] = 0
                
        for val in matrix.index.values:
            if val not in matrix.columns.values:
                matrix.loc[:, val] = 0

        return matrix

    def performances(self, target_value, privileged=None):
        """
        Computes the performances on all data if privileged is None else
        on whether privileged sample or unprivileged one.

        Performances corresponds to TP, FP, TN and FN.

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
        dict:
            dictionary with True Positives, False Positives,
            True Negatives and False Negatives
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
            Specific value of the target
        privileged (bool, optional): 
            Boolean prescribing whether to
            condition this metric on the `privileged_groups`, if `True`, or
            the `unprivileged_groups`, if `False`. Defaults to `None`
            meaning this metric is computed over the entire dataset.

        Returns
        -------
        int:
            Number of a specified target value of all data if privileged is None, privileged
            values if privileged is True and unprivileged values if False.
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
        Returns the frequency of a specified target value. 

        Parameters
        ----------
        target_value:
            Specific value of the target
        privileged (bool, default None): 
            Boolean prescribing whether to
            condition this metric on the `privileged_groups`, if `True`, or
            the `unprivileged_groups`, if `False`. Defaults to `None`
            meaning this metric is computed over the entire dataset.
        predictions: bool
            Whether the frequency is based on predictions or real labels
            
        Returns
        -------
        float
            Frequency of a specified target value
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
            Base rate
        """
        return (self.num_spec_value(target_value=target_value, privileged=privileged, predictions=True)
                / self.num_instances(privileged=privileged))

    def false_positive_rate(self, target_value, privileged=None):
        """
        Returns the ratio of true positives to positive examples in the
        dataset, :math:`TPR = TP/P`, optionally conditioned on protected
        attributes.
        Here negatives values correspond to :math:`y <> v`
        So false positives values correspond to :math:`y <> v\ \&\ \hat{y} = v`

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
            False positive rate
        """
        other_values = [v for v in list(set(self.preds)) if v != target_value]

        FP = self.performances(target_value=target_value,
                               privileged=privileged)['FP']
        N = 0
        for v in other_values:
            N += self.num_spec_value(target_value=v, privileged=privileged)
        if N == 0:
            return 0
        return FP / N

    def true_positive_rate(self, target_value, privileged=None):
        """
        Returns the ratio of true positives to positive examples in the
        dataset, :math:`TPR = TP/P`, optionally conditioned on protected
        attributes.
        Here positives values correspond to :math:`y = v`
        So true positives values correspond to :math:`y = v\ \&\ \hat{y} = v`

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
            True positive rate
        """
        TP = self.performances(target_value=target_value,
                               privileged=privileged)['TP']
        P = self.num_spec_value(
            target_value=target_value, privileged=privileged)
        if P == 0:
            return 0
        return TP / P

    def disparate_impact(self, target_value):
        r"""
        Computes the disparate impact for a specific label value

        .. math::
           \frac{Pr(\hat{Y} = v | D = \text{unprivileged})}
           {Pr(\hat{Y} = v | D = \text{privileged})}

        code source inspired from aif360_

        .. _aif360: https://aif360.readthedocs.io/en/latest/_modules/aif360/metrics/classification_metric.html

        Parameters
        ----------
        target_value:
            Specific label value for which it will compute the metric

        Returns
        -------
        float:
            Disparate impact bias metric
        """
        return self.ratio(self.base_rate, target_value=target_value)

    def statistical_parity_difference(self, target_value):
        r"""
        Computes the statistical parity difference for a specific label value

        .. math::
           Pr(\hat{Y} = v | D = \text{unprivileged})
           - Pr(\hat{Y} = v | D = \text{privileged})

        code source inspired from aif360_

        .. _aif360: https://aif360.readthedocs.io/en/latest/_modules/aif360/metrics/classification_metric.html

        Parameters
        ----------
        target_value:
            Specific label value for which it will compute the metric

        Returns
        -------
        float:
            Statistical parity difference bias metric
        """
        return self.difference(self.base_rate, target_value=target_value)

    def equal_opportunity_difference(self, target_value):
        r"""
        Computes the statistical parity difference for a specific label value
        
        :math:`TPR_{D = \text{unprivileged}} - TPR_{D = \text{privileged}}`

        code source from aif360_

        .. _aif360: https://aif360.readthedocs.io/en/latest/_modules/aif360/metrics/classification_metric.html

        Parameters
        ----------
        target_value:
            Specific label value for which it will compute the metric

        Returns
        -------
        float:
            Equal opportunity difference bias metric
        """
        return self.difference(self.true_positive_rate, target_value=target_value)

    def average_abs_odds_difference(self, target_value):
        r"""
        Average of absolute difference in FPR and TPR for unprivileged and
        privileged groups:

        .. math::

           \frac{1}{2}\left[|FPR_{D = \text{unprivileged}} - FPR_{D = \text{privileged}}|
           + |TPR_{D = \text{unprivileged}} - TPR_{D = \text{privileged}}|\right]

        A value of 0 indicates equality of odds.

        code source from aif360_

        .. _aif360: https://aif360.readthedocs.io/en/latest/_modules/aif360/metrics/classification_metric.html

        Parameters
        ----------
        target_value:
            Specific label value for which it will compute the metric

        Returns
        -------
        float:
            Average of absolute difference bias metric
        """
        return 0.5 * (np.abs(self.difference(self.false_positive_rate, target_value=target_value))
                      + np.abs(self.difference(self.true_positive_rate, target_value=target_value)))

    def theil_index(self, target_value):
        r"""
        Theil index or Generalized entropy index (with $\alpha=1$) is proposed as a unified individual and
        group fairness measure.
        
        With :math:`b_i = \hat{y}_i - y_i + 1`:

        .. math:: 

            \frac{1}{n}\sum_{i=1}^n\frac{b_{i}}{\mu}\ln\frac{b_{i}}{\mu}
            
        code source from aif360_

        .. _aif360: https://aif360.readthedocs.io/en/latest/_modules/aif360/metrics/classification_metric.html

        Parameters
        ----------
        target_value:
            Specific label value for which it will compute the metric

        Returns
        -------
        float:
            Theil index bias metric
        """
        y_pred = self.preds
        y_true = self.labels
        y_pred = (y_pred == target_value).astype(np.float64)
        y_true = (y_true == target_value).astype(np.float64)
        b = 1 + y_pred - y_true

        return np.mean(np.log((b / np.mean(b))**b) / np.mean(b))

    def compute_model_bias_metrics(self):
        """
        Computes automaticaly all dataset bias metrics for this
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
