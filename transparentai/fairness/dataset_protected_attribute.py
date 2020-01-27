import pandas as pd
from transparentai.fairness.protected_attribute import ProtectedAttribute


class DatasetProtectedAttribute(ProtectedAttribute):
    """
    This class retrieves all informations on a protected attribute on a specific dataset.
    It computes automatically the `Disparate impact` and the `Statistical parity difference` to
    get some insight about bias in the dataset.

    This class is inspired by the BinaryLabelDatasetMetric_ class from aif360 module but it 
    depends on a unique attribute.

    .. _BinaryLabelDatasetMetric: https://aif360.readthedocs.io/en/latest/modules/metrics.html#binary-label-dataset-metric
    """

    def __init__(self, dataset, attr, privileged_values, favorable_label=None):
        """
        """
        super().__init__(dataset, attr, privileged_values, favorable_label)
        self.compute_dataset_bias_metrics()

    def num_spec_value(self, target_value, privileged=None):
        r"""
        Compute the number of a particular value,
        :math:`P = \sum_{i=1}^n \mathbb{1}[y_i = v]`,
        optionally conditioned on protected attributes.

        Parameters
        ----------
        privileged (bool, optional): 
            Boolean prescribing whether to
            condition this metric on the `privileged_groups`, if `True`, or
            the `unprivileged_groups`, if `False`. Defaults to `None`
            meaning this metric is computed over the entire dataset.
        """
        if privileged == None:
            return self.crosstab.loc['All', target_value]
        elif privileged:
            return self.crosstab.loc[1, target_value]
        else:
            return self.crosstab.loc[0, target_value]

    def base_rate(self, target_value, privileged=None):
        """
        Compute the base rate, :math:`Pr(Y = 1) = P/(P+N)`, optionally
        conditioned on protected attributes.

        Parameters
        ----------
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
        return (self.num_spec_value(target_value=target_value, privileged=privileged)
                / self.num_instances(privileged=privileged))

    def disparate_impact(self, target_value):
        r"""
        Compute the disparate impact for a specific label value

        .. math::
           \frac{Pr(Y = v | D = \text{unprivileged})}
           {Pr(Y = v | D = \text{privileged})}

        Parameters
        ----------
        target_value:
            Specific label value for which it will compute
            the metric
        """
        return self.ratio(self.base_rate, target_value=target_value)

    def statistical_parity_difference(self, target_value):
        r"""
        Compute the statistical parity difference for a specific label value

        .. math::
           Pr(Y = v | D = \text{unprivileged})
           - Pr(Y = v | D = \text{privileged})

        Parameters
        ----------
        target_value:
            Specific label value for which it will compute
            the metric
        """
        return self.difference(self.base_rate, target_value=target_value)

    def compute_dataset_bias_metrics(self):
        """
        Compute automaticaly all dataset bias metrics for this
        protected attribute.
        """
        metrics_dict = {
            'Disparate impact': self.disparate_impact,
            'Statistical parity difference': self.statistical_parity_difference
        }

        self.compute_bias_metrics(metrics_dict=metrics_dict)
