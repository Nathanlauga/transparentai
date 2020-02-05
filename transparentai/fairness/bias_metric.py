import pandas as pd
import numpy as np
from IPython.display import display, Markdown

from transparentai.datasets.structured_dataset import StructuredDataset
import transparentai.fairness.fairness_plots as plots


class BiasMetric():
    """
    """
    bias_type = None

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
            raise ValueError(
                'favorable_label has to be a value of dataset target columns.')
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

    def set_bias_type(self, bias_type):
        """
        Set the bias type of the Bias metric class.
        Can only be dataset or model.

        Parameters
        ----------
        bias_type: str
            Bias type of the object : only 'dataset' or 'model'
        """
        if bias_type not in ['dataset', 'model']:
            raise ValueError("bias_type should be 'dataset' or 'model'")

        self.bias_type = bias_type

    def plot_bias(self, attr=None, target_value=None):
        """
        Display a custom matplotlib graphic to show if a protected attribute is biased or not

        Current dataset bias metrics : 
        - Disparate impact 
        - Statistical parity difference

        Current model bias metrics :
        - Disparate impact 
        - Statistical parity difference
        - Equal opportunity difference
        - Average abs odds difference
        - Theil index

        Parameters
        ----------
        attr: str (optional)
            Protected attribute to inspect (if None display bias for all attributes)
        target_value: str (optional)
            Specific value of the target (if None display bias for all target values)
            It's usefull to avoid repetition in case of a binary classifier
        """
        if self.bias_type is None:
            raise ValueError(
                'Please set a bias type on this BiasMetric object (dataset or model)')

        if (target_value is None) and (self.favorable_label is not None):
            target_value = self.favorable_label
        if attr != None:
            plots.plot_bias_metrics(
                protected_attr=self.protected_attributes[attr],
                bias_type=self.bias_type,
                target_value=target_value)
        else:
            for attr in self.protected_attributes:
                display(Markdown(''))
                plots.plot_bias_metrics(
                    protected_attr=self.protected_attributes[attr],
                    bias_type=self.bias_type,
                    target_value=target_value)
