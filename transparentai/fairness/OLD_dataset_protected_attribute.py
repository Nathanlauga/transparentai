import pandas as pd
from .protected_attribute import ProtectedAttribute


class DatasetProtectedAttribute(ProtectedAttribute):
    """
    This class retrieves all informations on a protected attribute on a specific dataset.
    It computes automatically the `Disparate impact` and the `Statistical parity difference` to
    get some insight about bias in the dataset.

    This class is inspired by the BinaryLabelDatasetMetric_ class from aif360 module but it 
    depends on a unique attribute.

    .. _BinaryLabelDatasetMetric: https://aif360.readthedocs.io/en/latest/modules/metrics.html#binary-label-dataset-metric

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
        super().__init__(dataset, attr, privileged_values, favorable_label)
        self.compute_dataset_bias_metrics()

    def num_spec_value(self, target_value, privileged=None):
        r"""
        Computes the number of a particular value,
        :math:`P = \sum_{i=1}^n \mathbb{1}[y_i = v]`,
        optionally conditioned on protected attributes.

        Parameters
        ----------
        target_value:
            Specific value of the target
        privileged (bool, optional): 
            Boolean prescribing whether to
            condition this metric on the `privileged_groups`, if `True`, or
            the `unprivileged_groups`, if `False`. Defaults to `None`
            meaning this metric is computed over the entire dataset.

        Returns
        -------
        int:
            Number of a specified target value of all data if privileged is None, privileged
            values if privileged is True and unprivileged values if False.
        """
        if privileged == None:
            return self.crosstab.loc['All', target_value]
        elif privileged:
            return self.crosstab.loc[1, target_value]
        else:
            return self.crosstab.loc[0, target_value]

    
    def get_freq(self, target_value, privileged=None):
        """
        Returns the frequency of a specified target value. 

        Parameters
        ----------
        target_value:
            Specific value of the target
        privileged (bool, default None): 
                Boolean prescribing whether to
                condition this metric on the `privileged_groups`, if `True`, or
                the `unprivileged_groups`, if `False`. Defaults to `None`
                meaning this metric is computed over the entire dataset.
                
        Returns
        -------
        float
            Frequency of a specified target value
        """
        n_total = self.num_instances(privileged=privileged)
        n = self.num_spec_value(
            target_value=target_value, privileged=privileged)

        return n / n_total

    def base_rate(self, target_value, privileged=None):
        """
        Computes the base rate, :math:`Pr(Y = 1) = P/(P+N)`, optionally
        conditioned on protected attributes.

        Parameters
        ----------
        target_value:
            Specific value of the target
        privileged (bool, optional): 
                Boolean prescribing whether to
                condition this metric on the `privileged_groups`, if `True`, or
                the `unprivileged_groups`, if `False`. Defaults to `None`
                meaning this metric is computed over the entire dataset.

        Returns
        -------
        float: 
            Base rate
        """
        return (self.num_spec_value(target_value=target_value, privileged=privileged)
                / self.num_instances(privileged=privileged))

    def disparate_impact(self, target_value):
        r"""
        Computes the disparate impact for a specific label value

        .. math::
           \frac{Pr(Y = v | D = \text{unprivileged})}
           {Pr(Y = v | D = \text{privileged})}

        Parameters
        ----------
        target_value:
            Specific label value for which it will compute
            the metric

        Returns
        -------
        float:
            Disparate impact bias metric
        """
        return self.ratio(self.base_rate, target_value=target_value)

    def statistical_parity_difference(self, target_value):
        r"""
        Computes the statistical parity difference for a specific label value

        .. math::
           Pr(Y = v | D = \text{unprivileged})
           - Pr(Y = v | D = \text{privileged})

        Parameters
        ----------
        target_value:
            Specific label value for which it will compute
            the metric

        Returns
        -------
        number:
            Statistical parity difference bias metric
        """
        return self.difference(self.base_rate, target_value=target_value)

    def compute_dataset_bias_metrics(self):
        """
        Computes automaticaly all dataset bias metrics for this
        protected attribute.
        """
        metrics_dict = {
            'Disparate impact': self.disparate_impact,
            'Statistical parity difference': self.statistical_parity_difference
        }

        self.compute_bias_metrics(metrics_dict=metrics_dict)
