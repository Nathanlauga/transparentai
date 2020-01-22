import pandas as pd
import numpy as np


class ProtectedAttribute():
    """
    This class retrieves all informations on a protected attribute on a specific dataset.
    It computes automatically the `Disparate impact` and the `Statistical parity difference` to
    get some insight about bias in the dataset.

    This class is inspired by the BinaryLabelDatasetMetric_ class from aif360 module.

    .. _BinaryLabelDatasetMetric: https://aif360.readthedocs.io/en/latest/modules/metrics.html#binary-label-dataset-metric
    """

    def __init__(self, df, attr, target, privileged_values):
        """
        Parameters
        ----------
        df: pd.DataFrame
            Dataframe to inspect
        attr: str
            Name of the attribute to analyse (it has to be into df columns)
        target: str
            Name of the label (or target) column
        privileged_values: list
            List with privileged values inside the column (e.g. ['Male'] for 'gender' attribute)  
        """
        self.name = attr
        self.target = target
        self.privileged_values = privileged_values
        self.unprivileged_values = [
            v for v in df[attr].unique() if v not in privileged_values]

        self.values = pd.Series(
            np.where(df[attr].isin(privileged_values), 1, 0), name=attr)
        self.labels = df[target]
        self.crosstab = pd.crosstab(self.values, self.labels, margins=True)
        self.compute_dataset_bias_metrics()

    def get_privileged_values(self):
        return f'{self.name} = '+' or '.join([str(v) for v in self.privileged_values])

    def get_unprivileged_values(self):
        return f'{self.name} = '+' or '.join([str(v) for v in self.unprivileged_values])

    def num_instances(self, privileged=None):
        """
        Compute the number of instances, :math:`n`, in the dataset conditioned
        on protected attributes if necessary.

        Parameters
        ----------
        privileged (bool, optional): 
                Boolean prescribing whether to
                condition this metric on the `privileged_groups`, if `True`, or
                the `unprivileged_groups`, if `False`. Defaults to `None`
                meaning this metric is computed over the entire dataset.
        """
        if privileged == None:
            return self.crosstab.loc['All', 'All']
        elif privileged:
            return self.crosstab.loc[1, 'All']
        else:
            return self.crosstab.loc[0, 'All']

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

    def difference(self, metric_fun, target_value):
        """
        Compute difference of the metric for unprivileged and privileged groups.
        """
        return (metric_fun(target_value=target_value, privileged=False)
                - metric_fun(target_value=target_value, privileged=True))

    def ratio(self, metric_fun, target_value):
        """
        Compute ratio of the metric for unprivileged and privileged groups.
        """
        return (metric_fun(target_value=target_value, privileged=False)
                / metric_fun(target_value=target_value, privileged=True))

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
        metrics = pd.DataFrame()
        metrics.name = self.name

        for target_value in self.labels.unique():
            metrics.loc[target_value, 'Disparate impact'] = self.disparate_impact(
                target_value=target_value)
            metrics.loc[target_value, 'Statistical parity difference'] = \
                self.statistical_parity_difference(target_value=target_value)

        self.metrics = metrics
