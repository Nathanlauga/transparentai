import pandas as pd
import numpy as np
from IPython.display import display, Markdown

from .ProtectedAttribute import ProtectedAttribute
from ..visuals import plot_dataset_metrics


class StructuredDataset():
    """
    """

    def __init__(self, df, privileged_values, target=None):
        """
        Parameters
        ----------
        df: pd.DataFrame
            Dataframe with appropriate dtypes
        privileged_values: dict
            Dictionnary with all protected attributes (category dtype) as key
            and a list of privileged value for the variable
            e.g. { 'gender' : ['Male'] }
        target: str
            target column, if None then this StructuredDataset will not
            be abble to use bias insight
        """
        if df is None:
            raise TypeError("Must provide a pandas DataFrame representing "
                            "the data (features, labels, protected attributes)")
        if label_name not in df.columns:
            print('error : label not in dataframe')
        if any([v not in df.columns for v in privileged_values]):
            print('error : privileged variables not in dataframe')

        self.df = df.copy()
        self.target = target

        self.protected_attributes = {}
        for attr in privileged_values:
            protected_attr = ProtectedAttribute(df=df, 
                                                attr=attr,
                                                label_name=target,
                                                privileged_values=privileged_values[attr]
                                                )
            self.protected_attributes[attr] = protected_attr
            # self.df[attr] = protected_attr.values

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        return str(self)

    def __str__(self):
        return str(self.df)

    def split_train_test(self, freq, validation=False):
        """
        Parameters
        ----------
        freq: tuple
            tuple containing train freq, test freq and validation freq
            if validation is True
        validation: bool
            split dataset in 3 : train, test and validation
        """
        if sum(freq) != 1:
            print('error should sum to 1')
        if validation & (len(freq) != 3):
            print('error for validation split 3 value in freq') 
        

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
