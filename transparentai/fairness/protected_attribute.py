import pandas as pd
import numpy as np


class ProtectedAttribute():
    """
    """

    def __init__(self, dataset, attr, privileged_values, favorable_label=None):
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
        self.favorable_label = favorable_label
        self.privileged_values = privileged_values
        self.unprivileged_values = [
            v for v in dataset.df[attr].unique() if v not in privileged_values]

        self.values = pd.Series(
            np.where(dataset.df[attr].isin(privileged_values), 1, 0), name=attr)
        self.labels = dataset.df[dataset.target]
        self.crosstab = pd.crosstab(self.values, self.labels, margins=True)

    def to_frame(self):
        """
        """
        df = pd.DataFrame()
        df[self.name] = self.values
        df[self.target] = self.labels
        return df

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

    def compute_bias_metrics(self, metrics_dict):
        """

        Parameters
        ----------
        metrics_dict: dict
            Dictionnary with metric name on keys and 
            metric function as value
        """
        if self.favorable_label is not None:
            target_values = [self.favorable_label]
        else:
            target_values = self.labels.unique()

        metrics = pd.DataFrame()
        metrics.name = self.name

        for target_value in target_values:
            for metric, func in metrics_dict.items():
                val = func(target_value=target_value)
                metrics.loc[target_value, metric] = val

        self.metrics = metrics
