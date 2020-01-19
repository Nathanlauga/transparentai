import pandas as pd
import numpy as np


class ProtectedAttribute():
    """
    """

    def __init__(self, df, attr, label_name, privileged_values):
        """
        """
        self.name = attr
        self.label_name = label_name
        self.privileged_values = privileged_values
        self.unprivileged_values = [
            v for v in df[attr].unique() if v not in privileged_values]

        self.values = pd.Series(
            np.where(df[attr].isin(privileged_values), 1, 0), name=attr)
        self.labels = df[label_name]
        self.crosstab = pd.crosstab(self.values, self.labels, margins=True)
        self._compute_metrics()

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

    def num_spec_value(self, label_value, privileged=None):
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
            return self.crosstab.loc['All', label_value]
        elif privileged:
            return self.crosstab.loc[1, label_value]
        else:
            return self.crosstab.loc[0, label_value]

    def base_rate(self, label_value, privileged=None):
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
        return (self.num_spec_value(label_value=label_value, privileged=privileged)
                / self.num_instances(privileged=privileged))

    def difference(self, metric_fun, label_value):
        """
        Compute difference of the metric for unprivileged and privileged groups.
        """
        return (metric_fun(label_value=label_value, privileged=False)
                - metric_fun(label_value=label_value, privileged=True))

    def ratio(self, metric_fun, label_value):
        """
        Compute ratio of the metric for unprivileged and privileged groups.
        """
        return (metric_fun(label_value=label_value, privileged=False)
                / metric_fun(label_value=label_value, privileged=True))

    def disparate_impact(self, label_value):
        r"""
        Compute the disparate impact for a specific label value

        .. math::
           \frac{Pr(Y = v | D = \text{unprivileged})}
           {Pr(Y = v | D = \text{privileged})}

        Parameters
        ----------
        label_value:
            Specific label value for which it will compute
            the metric
        """
        return self.ratio(self.base_rate, label_value=label_value)

    def statistical_parity_difference(self, label_value):
        r"""
        Compute the disparate impact for a specific label value

        .. math::
           Pr(Y = v | D = \text{unprivileged})
           - Pr(Y = v | D = \text{privileged})

        Parameters
        ----------
        label_value:
            Specific label value for which it will compute
            the metric
        """
        return self.difference(self.base_rate, label_value=label_value)

    def _compute_metrics(self):
        """
        """
        metrics = pd.DataFrame()
        metrics.name = self.name

        for label_value in self.labels.unique():
            metrics.loc[label_value, 'Disparate impact'] = self.disparate_impact(
                label_value=label_value)
            metrics.loc[label_value, 'Statistical parity difference'] = \
                self.statistical_parity_difference(label_value=label_value)

        self.metrics = metrics
