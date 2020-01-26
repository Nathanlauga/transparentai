import pandas as pd
import numpy as np

from transparentai.datasets.structured_dataset import StructuredDataset
from transparentai.fairness.dataset_protected_attribute import DatasetProtectedAttribute
import transparentai.fairness.fairness_plots as plots


class DatasetBiasMetric():
    """
    """

    def __init__(self, dataset, privileged_groups):
        """

        privileged_values: dict
            Dictionnary with all protected attributes (category dtype) as key
            and a list of privileged value for the variable
            e.g. { 'gender' : ['Male'] }
        """

        if not isinstance(dataset, StructuredDataset):
            raise TypeError(
                'dataset attribute must be a StructuredDataset object!')
        if dataset.target is None:
            raise ValueError('target from StructuredDataset has to be set')
        if privileged_groups is None:
            raise ValueError('privileged_groups has to be set')
        if (privileged_groups is not None) and (any([v not in dataset.df.columns for v in privileged_groups])):
            print('error : privileged variables not in dataset')

        self.dataset = dataset
        self.labels = dataset.df[dataset.target]
        protected_attributes = {}

        for attr in privileged_groups:
            values = privileged_groups[attr]
            protected_attr = DatasetProtectedAttribute(dataset=dataset,
                                                       attr=attr,
                                                       privileged_values=values
                                                       )
            protected_attributes[attr] = protected_attr

        self.protected_attributes = protected_attributes

    def get_bias_metrics(self, attr=None):
        """
        Retrieve all bias metrics of protected attributes and format
        them to a DataFrame.

        Current dataset bias metrics : 
        - Disparate impact 
        - Statistical parity difference

        Parameters
        ----------
        attr: str (optional)
            Protected attribute to inspect (if None display bias for all attributes)

        Returns
        -------
        pd.DataFrame
            formated bias metrics dataframe
        """
        if attr == None:
            metrics = list()
            for attr in self.protected_attributes:
                metrics.append(self.protected_attributes[attr].metrics)
            metrics = pd.DataFrame(metrics)
            metrics.index = list(self.protected_attributes.keys())
            metrics.columns = ['Metrics dataframe']
        else:
            metrics = self.protected_attributes[attr].metrics
        return metrics

    def plot_bias(self, attr=None, target_value=None):
        """
        Display a custom matplotlib graphic to show if a protected attribute is biased or not

        Current dataset bias metrics : 
        - Disparate impact 
        - Statistical parity difference

        Parameters
        ----------
        attr: str (optional)
            Protected attribute to inspect (if None display bias for all attributes)
        target_value: str (optional)
            Specific value of the target (if None display bias for all target values)
            It's usefull to avoid repetition in case of a binary classifier
        """
        if attr != None:
            plots.plot_dataset_metrics(
                protected_attr=self.protected_attributes[attr], target_value=target_value)
        else:
            for attr in self.protected_attributes:
                display(Markdown(''))
                plots.plot_dataset_metrics(
                    protected_attr=self.protected_attributes[attr], target_value=target_value)
