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

    def __init__(self, dataset, attr, privileged_values):
        """
        Parameters
        ----------
        dataset: pd.DataFrame
            Dataframe to inspect
        attr: str
            Name of the attribute to analyse (it has to be into dataset columns)
        target: str
            Name of the label (or target) column
        privileged_values: list
            List with privileged values inside the column (e.g. ['Male'] for 'gender' attribute)  
        """
        self.name = attr
        self.target = dataset.target
        self.privileged_values = privileged_values
        self.unprivileged_values = [
            v for v in dataset.df[attr].unique() if v not in privileged_values]

        self.values = pd.Series(
            np.where(dataset.df[attr].isin(privileged_values), 1, 0), name=attr)
        self.labels = dataset.df[dataset.target]
        self.crosstab = pd.crosstab(self.values, self.labels, margins=True)

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