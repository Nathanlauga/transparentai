import pandas as pd
import numpy as np

from transparentai.fairness.model_protected_attribute import ModelProtectedAttribute
from transparentai.fairness.bias_metric import BiasMetric


class ModelBiasMetric(BiasMetric):
    """
    Dataset class for bias metrics. It stored different metrics for a set of
    protected attributes (you can find some examples here_).

    List of bias metrics for DatasetBiasMetric : 

    1. Disparate impact
    2. Statistical parity difference
    3. Equal opportunity difference
    4. Average abs odds difference
    5. Theil index

    .. _here: https://www.fairwork.gov.au/employee-entitlements/protections-at-work/protection-from-discrimination-at-work

    Example
    -------
    For binary classification :

    .. code-block:: python

        from transparentai.fairness import ModelBiasMetric
        
        ...
        adult = ... # dataset
        clf = ... # train classifier
        preds = ... # predictions from classifier
        ...

        target = 'income'
        favorable_label = '>50K'
        dataset = StructuredDataset(df=adult, target=target)

        privileged_groups = {
            'marital-status': ['Married-civ-spouse','Married-AF-spouse'],
            'race': ['White'],
            'gender': ['Male']
        }   

        model_bias = ModelBiasMetric(dataset=dataset, preds=preds,
                                    privileged_groups=privileged_groups,
                                    favorable_label=favorable_label)

    Attributes
    ----------
    bias_type: str
        Bias type of the object : only 'dataset' or 'model'
    dataset: StructuredDataset
        dataset with a target set as attribute
    favorable_label:
        A specific value that is has an advantage over the other(s) 
        values
    labels: array like
        list of values inside target column
    protected_attributes: dict
        Dictionary with protected attributes as keys (e.g. gender) and
        ProtectedAttribute objects as values
    """

    def __init__(self, dataset, preds, privileged_groups, favorable_label=None):
        """
        Parameters
        ----------
        dataset: StructuredDataset
            dataset with a target set as attribute
        preds: array like (same length than dataset.labels)
            Predictions for the dataset 
        privileged_groups: dict(list())
            Dictionary with protected attributes as keys (e.g. gender) and
            list of value(s) that are considered privileged (e.g. ['Male'])
        privileged_values: dict
            Dictionnary with all protected attributes (category dtype) as key
            and a list of privileged value for the variable
            e.g. { 'gender' : ['Male'] }
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
