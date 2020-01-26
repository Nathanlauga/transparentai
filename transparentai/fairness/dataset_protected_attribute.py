import pandas as pd
from transparentai.fairness.protected_attribute import ProtectedAttribute


class DatasetProtectedAttribute(ProtectedAttribute):
    """
    """

    def __init__(self, dataset, attr, privileged_values):
        """
        """
        super().__init__(dataset, attr, privileged_values)
        self.compute_dataset_bias_metrics()

    def disparate_impact(self, target_value):
        r"""
        Compute the disparate impact for a specific label value

        .. math::
           \frac{Pr(Y = v | D = \text{unprivileged})}
           {Pr(Y = v | D = \text{privileged})}

        Parameters
        ----------
        target_value:
            Specific label value for which it will compute
            the metric
        """
        return self.ratio(self.base_rate, target_value=target_value)

    def statistical_parity_difference(self, target_value):
        r"""
        Compute the statistical parity difference for a specific label value

        .. math::
           Pr(Y = v | D = \text{unprivileged})
           - Pr(Y = v | D = \text{privileged})

        Parameters
        ----------
        target_value:
            Specific label value for which it will compute
            the metric
        """
        return self.difference(self.base_rate, target_value=target_value)

    def compute_dataset_bias_metrics(self):
        """
        Compute automaticaly all dataset bias metrics for this
        protected attribute.
        """
        metrics = pd.DataFrame()
        metrics.name = self.name

        for target_value in self.labels.unique():
            metrics.loc[target_value, 'Disparate impact'] = self.disparate_impact(
                target_value=target_value)
            metrics.loc[target_value, 'Statistical parity difference'] = \
                self.statistical_parity_difference(target_value=target_value)

        self.metrics = metrics
