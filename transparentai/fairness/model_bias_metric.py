import pandas as pd
import numpy as np

from transparentai.fairness.model_protected_attribute import ModelProtectedAttribute
from transparentai.fairness.bias_metric import BiasMetric


class ModelBiasMetric(BiasMetric):
    """
    """

    def __init__(self, dataset, preds, privileged_groups, favorable_label=None):
        """
        """
        super().__init__(dataset, privileged_groups, favorable_label)
        self.preds = preds
        protected_attributes = {}

        if (dataset.target_mean is not None):
            preds = np.where(preds > dataset.target_mean,
                             f'>{dataset.target_mean}', f'<={dataset.target_mean}')

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
        self.set_bias_type(bias_type='model')
