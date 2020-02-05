import pandas as pd
import numpy as np
from IPython.display import display, Markdown

from transparentai.fairness.dataset_protected_attribute import DatasetProtectedAttribute
from transparentai.fairness.bias_metric import BiasMetric
import transparentai.fairness.fairness_plots as plots


class DatasetBiasMetric(BiasMetric):
    """
    """

    def __init__(self, dataset, privileged_groups, favorable_label=None):
        super().__init__(dataset, privileged_groups, favorable_label)
        protected_attributes = {}

        for attr in privileged_groups:
            values = privileged_groups[attr]
            protected_attr = DatasetProtectedAttribute(dataset=dataset,
                                                       attr=attr,
                                                       privileged_values=values,
                                                       favorable_label=favorable_label
                                                       )
            protected_attributes[attr] = protected_attr

        self.protected_attributes = protected_attributes
        self.set_bias_type(bias_type='dataset')