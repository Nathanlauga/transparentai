import pandas as pd
import numpy as np
from IPython.display import display, Markdown

from transparentai.datasets.protected_attribute import ProtectedAttribute
from transparentai.plots import plot_dataset_metrics


class ClassificationDataset():
    """
    """

    def __init__(self, df, target, privileged_values):
        """

        """
        if df is None:
            raise TypeError("Must provide a pandas DataFrame representing "
                            "the data (features, labels, protected attributes)")
        if target not in df.columns:
            print('error : label not in dataframe')
        if any([v not in df.columns for v in privileged_values]):
            print('error : privileged variables not in dataframe')

        self.df = df.copy()
        self.target = target

        self.protected_attributes = {}
        for attr in privileged_values:
            protected_attr = ProtectedAttribute(df=df, attr=attr,
                                                target=target,
                                                privileged_values=privileged_values[attr])
            self.protected_attributes[attr] = protected_attr
            self.df[attr] = protected_attr.values

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        return str(self)

    def __str__(self):
        return str(self.df)

    def get_metrics(self, attr=None):
        """
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

    def show_bias_metrics(self, attr=None, label_value=None):
        """
        """
        if attr != None:
            plot_dataset_metrics(
                protected_attr=self.protected_attributes[attr], label_value=label_value)
        else:
            for attr in self.protected_attributes:
                display(Markdown(''))
                plot_dataset_metrics(
                    protected_attr=self.protected_attributes[attr], label_value=label_value)

    def show_bias_insight(self, attr=None, label_value=None):
        """
        """
        print('')
