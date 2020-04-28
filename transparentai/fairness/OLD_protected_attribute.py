import pandas as pd
import numpy as np

from abc import abstractmethod


class ProtectedAttribute():
    """
    This class retrieves all informations on a protected attribute on a specific dataset.

    Parent class of DatasetProtectedAttribute and ModelProtectedAttribute.

    Attributes
    ----------
    name: str
        Attribute name (the column's name in original dataframe)
    target:
        Name of the label (or target) column
    favorable_label:
        A specific value that is has an advantage over the other(s) values
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
    """

    def __init__(self, dataset, attr, privileged_values, favorable_label=None):
        """
        Parameters
        ----------
        dataset: pd.DataFrame
            Dataframe to inspect
        attr: str
            Name of the attribute to analyse (it has to be into dataset columns)
        privileged_values: list
            List with privileged values inside the column (e.g. ['Male'] for 'gender' attribute)  
        favorable_label:
            A specific value that is has an advantage over the other(s) values
        """
        self.name = attr
        self.target = dataset.target
        self.favorable_label = favorable_label
        self.privileged_values = privileged_values
        self.unprivileged_values = [
            v for v in dataset.df[attr].unique() if v not in privileged_values]

        self.values = pd.Series(
            np.where(dataset.df[attr].isin(privileged_values), 1, 0), name=attr).values
        self.labels = dataset.df[dataset.target].values
        self.crosstab = pd.crosstab(self.values, self.labels, margins=True)

        if 0 not in self.crosstab.index:
            self.crosstab[0] = 0
        if 1 not in self.crosstab.index:
            self.crosstab[1] = 0

    def to_frame(self):
        """
        Returns a dataframe with 2 columns : values in first column and 
        labels in the second one

        Returns
        -------
        pd.DataFrame
            formated dataframe
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
        privileged (bool, default None): 
                Boolean prescribing whether to
                condition this metric on the `privileged_groups`, if `True`, or
                the `unprivileged_groups`, if `False`. Defaults to `None`
                meaning this metric is computed over the entire dataset.

        Returns
        -------
        int:
            Number of instances of all data if privileged is None, privileged
            values if privileged is True and unprivileged values if False.
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

        Parameters
        ----------
        metric_fun: function
            metric function that returns a number
        target_value:
            Specific value of the target

        Returns
        -------
        float:
            Difference of a metric for unprivileged and privileged groups.
        """
        return (metric_fun(target_value=target_value, privileged=False)
                - metric_fun(target_value=target_value, privileged=True))

    def ratio(self, metric_fun, target_value):
        """
        Compute ratio of the metric for unprivileged and privileged groups.

        Parameters
        ----------
        metric_fun: function
            metric function that returns a number
        target_values:
            Specific value of the target

        Returns
        -------
        float:
            Ratio of a metric for unprivileged and privileged groups.
        """
        return (metric_fun(target_value=target_value, privileged=False)
                / metric_fun(target_value=target_value, privileged=True))

    def compute_bias_metrics(self, metrics_dict):
        """
        Computes bias metrics using a metrics_dict which has 
        metric names on keys and the associated function as values.

        Parameters
        ----------
        metrics_dict: dict
            Dictionnary with metric name on keys and 
            metric function as value
        """
        if self.favorable_label is not None:
            target_values = [self.favorable_label]
        else:
            if type(self.labels) in [pd.Series, pd.DataFrame]:
                target_values = self.labels.unique()
            else:
                target_values = list(set(self.labels))

        metrics = pd.DataFrame()
        metrics.name = self.name

        for target_value in target_values:
            for metric, func in metrics_dict.items():
                val = func(target_value=target_value)
                metrics.loc[target_value, metric] = val

        self.metrics = metrics

    # Abstract methods

    @abstractmethod
    def get_freq(self):
        return None