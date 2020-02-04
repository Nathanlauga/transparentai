import pandas as pd
import numpy as np
from IPython.display import display, Markdown

from transparentai.fairness.model_protected_attribute import ModelProtectedAttribute
from transparentai.fairness.bias_metric import BiasMetric
import transparentai.fairness.fairness_plots as plots


class ModelBiasMetric(BiasMetric):
    """
    """

    def __init__(self, dataset, preds, privileged_groups, favorable_label=None):
        """
        """
        super().__init__(dataset, privileged_groups, favorable_label)
        self.preds = preds
        protected_attributes = {}

        for attr in privileged_groups:
            values = privileged_groups[attr]
            protected_attr = ModelProtectedAttribute(dataset=dataset,
                                                     attr=attr,
                                                     privileged_values=values,
                                                     preds=preds,
                                                     favorable_label=favorable_label
                                                     )
            protected_attributes[attr] = protected_attr

        self.protected_attributes = protected_attributes

    
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
        if (target_value is None) and (self.favorable_label is not None):
            target_value = self.favorable_label
        if attr != None:
            plots.plot_bias_metrics(
                protected_attr=self.protected_attributes[attr], 
                bias_type='model',
                target_value=target_value)
        else:
            for attr in self.protected_attributes:
                display(Markdown(''))
                plots.plot_bias_metrics(
                    protected_attr=self.protected_attributes[attr], 
                    bias_type='model',
                    target_value=target_value)
