import pandas as pd
import numpy as np

from transparentai.datasets.structured_dataset import StructuredDataset


class BiasMetric():
    """
    """

    def __init__(self, dataset, privileged_groups, favorable_label=None):
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
        if (favorable_label is not None) and (favorable_label not in dataset.df[dataset.target].unique()):
            raise ValueError('favorable_label has to be a value of dataset target columns.')
        if privileged_groups is None:
            raise ValueError('privileged_groups has to be set')
        if (privileged_groups is not None) and (any([v not in dataset.df.columns for v in privileged_groups])):
            raise ValueError('privileged variables not in dataset')

        self.dataset = dataset
        self.favorable_label = favorable_label
        self.labels = dataset.df[dataset.target]

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
