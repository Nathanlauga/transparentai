import pandas as pd
import numpy as np
from IPython.display import display, Markdown

from .dataset_protected_attribute import DatasetProtectedAttribute
from .bias_metric import BiasMetric
from ..fairness import fairness_plots as plots


class DatasetBiasMetric(BiasMetric):
    """
    Dataset class for bias metrics. It stored different metrics for a set of
    protected attributes (you can find some examples here_).

    List of bias metrics for DatasetBiasMetric : 

    1. Disparate impact
    2. Statistical parity difference

    .. _here: https://www.fairwork.gov.au/employee-entitlements/protections-at-work/protection-from-discrimination-at-work


    Example
    -------
    For binary classification :

    .. code-block:: python

        import numpy as np
        from transparentai.datasets import StructuredDataset, load_adult
        from transparentai.fairness import DatasetBiasMetric

        adult = load_adult()

        target = 'income'
        dataset = StructuredDataset(df=adult, target=target)

        privileged_groups = {
            'marital-status': ['Married-civ-spouse','Married-AF-spouse'],
            'race': ['White'],
            'gender': ['Male']
        }   
        dataset_bias = DatasetBiasMetric(dataset, privileged_groups, favorable_label='>50K')

    
    For regression :

    .. code-block:: python

        import numpy as np
        from transparentai.datasets import StructuredDataset, load_boston
        from transparentai.fairness import DatasetBiasMetric
        
        boston = load_boston()

        # Transform age to be a categorical variable for protected attributes
        boston['age category'] = np.where(boston['AGE'] < 26, 'Young',
                                        np.where(boston['AGE'] < 61, 'Adult','Elder'))

        target = 'MEDV'
        dataset = StructuredDataset(df=boston, target=target, target_regr=True)

        privileged_groups = {
            'age category': ['Adult']
        }   
        dataset_bias = DatasetBiasMetric(dataset, privileged_groups)

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

    def __init__(self, dataset, privileged_groups, favorable_label=None):
        """
        Parameters
        ----------
        dataset: StructuredDataset
            dataset with a target set as attribute
        privileged_groups: dict(list())
            Dictionary with protected attributes as keys (e.g. gender) and
            list of value(s) that are considered privileged (e.g. ['Male'])
        privileged_values: dict
            Dictionnary with all protected attributes (category dtype) as key
            and a list of privileged value for the variable
            e.g. { 'gender' : ['Male'] }
        """
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