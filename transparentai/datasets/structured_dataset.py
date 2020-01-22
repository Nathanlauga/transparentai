import pandas as pd
import numpy as np
import scipy.stats as ss
from IPython.display import display, Markdown

from transparentai.datasets.protected_attribute import ProtectedAttribute
import transparentai.plots.datasets_plots as plots
import transparentai.utils as utils


class StructuredDataset():
    """
    Class to inspect a strutured (tabular) dataset based on a pandas DataFrame object.
    It could help you to explore your data, understand what's in it with plot functions.
    And you can also find out dataset bias with `plot_bias` function.
    """

    def __init__(self, df, target=None, privileged_values=None):
        """
        Parameters
        ----------
        df: pd.DataFrame
            Dataframe with appropriate dtypes
        target: str
            target column, if None then this StructuredDataset will not
            be abble to use bias insight
        privileged_values: dict
            Dictionnary with all protected attributes (category dtype) as key
            and a list of privileged value for the variable
            e.g. { 'gender' : ['Male'] }

        Raises
        ------
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

        if privileged_values != None:
            for attr in privileged_values:
                values = privileged_values[attr]
                protected_attr = ProtectedAttribute(df=df,
                                                    attr=attr,
                                                    target=target,
                                                    privileged_values=values
                                                    )
                self.protected_attributes[attr] = protected_attr

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

    # Plot functions
    # --------------

    def plot_missing_values(self):
        """
        Display a bar chart of missing values for columns which contains
        at least one missing value.
        """
        plots.plot_missing_values(df=self.df)

    def plot_num_variable(self, var):
        """
        Display multiple graphics of a numerical variable.

        Parameters
        ----------
        var: str
            Variable name of a numerical variable inside dataframe object        

        Raises
        ------
        """
        if var is None:
            raise TypeError("Must provide a valid variable name string")
        if var not in self.df.columns:
            raise ValueError(
                'The variable was not found inside the columns data')
        if var not in self.df.select_dtypes('number').columns:
            raise ValueError(
                'The variable has to be a numerical variable inside the dataframe')

        plots.plot_numerical_var(df=self.df, var=var, target=self.target)

    def plot_cat_variable(self, var):
        """
        Display multiple graphics of a categorical variable.

        Parameters
        ----------
        var: str
            Variable name of a categorical variable inside dataframe object        

        Raises
        ------
        """
        if var is None:
            raise TypeError("Must provide a valid variable name string")
        if var not in self.df.columns:
            raise ValueError(
                'The variable was not found inside the columns data')
        if var not in self.df.select_dtypes(['object', 'category']).columns:
            raise ValueError(
                'The variable has to be a categorical variable inside the dataframe')
        plots.plot_categorical_var(df=self.df, var=var, target=self.target)

    def plot_datetime_variable(self, var):
        """
        Display multiple graphics of a datetime variable.

        Parameters
        ----------
        var: str
            Variable name of a datetime variable inside dataframe object        

        Raises
        ------
        """
        if var is None:
            raise TypeError("Must provide a valid variable name string")
        if var not in self.df.columns:
            raise ValueError(
                'The variable was not found inside the columns data')
        if var not in self.df.select_dtypes('datetime').columns:
            raise ValueError(
                'The variable has to be a datetime variable inside the dataframe')
        plots.plot_datetime_var(df=self.df, var=var, target=self.target)

    def plot_variables(self):
        """
        Display multiple graphics of all differents variables.
        The goal of this function is to do less code and get insightfull 
        informations about the data.

        Data type handle : categorical, numerical, datetime
        """
        cat_vars = self.df.select_dtypes(['object', 'category'])
        num_vars = self.df.select_dtypes('number')
        dat_vars = self.df.select_dtypes('datetime')

        if len(num_vars) > 0:
            display(Markdown('### Numerical variables'))
        for var in num_vars:
            display(Markdown(''))
            plots.display_meta_var(self.df, var)
            if len(self.df[var].unique()) <= 1:
                display(Markdown('Only one value.'))
                continue
            self.plot_num_variable(var=var)

        if len(cat_vars) > 0:
            display(Markdown('### Categorical variables'))
        for var in cat_vars:
            display(Markdown(''))
            plots.display_meta_var(self.df, var)
            if len(self.df[var].unique()) <= 1:
                display(Markdown('Only one value.'))
                continue
            self.plot_cat_variable(var=var)

        if len(num_vars) > 0:
            display(Markdown('### Datetime variables'))
        for var in dat_vars:
            display(Markdown(''))
            plots.display_meta_var(self.df, var)
            if len(self.df[var].unique()) <= 1:
                display(Markdown('Only one value.'))
                continue
            self.plot_datetime_variable(var=var)

    def plot_scatter_two_variables(self, var1, var2):
        """
        Show two numerical variables relations with jointplot.

        Parameters
        ----------
        var1: str
            Column name that contains first numerical values
        var2: str
            Column name that contains second numerical values

        Raises
        ------
        """
        if var1 is None or var2 is None:
            raise TypeError("Must provide valid variables name string")
        if (var1 not in self.df.columns) or (var2 not in self.df.columns):
            raise ValueError(
                'At least one of the variables was not found inside the columns data')
        if (var1 not in self.df.select_dtypes('number').columns) or (var2 not in self.df.select_dtypes('number').columns):
            raise ValueError(
                'Both variables has to be a numerical variables inside the dataframe')

        plots.plot_numerical_jointplot(
            df=self.df, var1=var1, var2=var2, target=self.target)

    def plot_variables_scatter(self):
        """
        Show all numerical variables 2 by 2 with graphics understand their relation.
        If target is set, separate dataset for each target value.
        """
        num_vars = self.df.select_dtypes('number')
        num_vars = utils.remove_var_with_one_value(num_vars)

        cols = num_vars.columns.values
        var_combi = [tuple(sorted([v1, v2]))
                     for v1 in cols for v2 in cols if v1 != v2]
        var_combi = list(set(var_combi))

        for var1, var2 in var_combi:
            display(Markdown(''))
            display(Markdown(f'Joint plot for **{var1}** & **{var2}**'))
            self.plot_scatter_two_variables(var1=var1, var2=var2)

    def plot_boxplot_two_variables(self, var1, var2):
        """
        Show boxplots for a specific pair of categorical and numerical variables
        If target is set, separate dataset for each target value.

        Parameters
        ----------
        var1: str
            Column name that contains categorical values
        var2: str
            Column name that contains numerical values
        """
        if var1 is None or var2 is None:
            raise TypeError("Must provide valid variables name string")
        if (var1 not in self.df.columns) or (var2 not in self.df.columns):
            raise ValueError(
                'At least one of the variables was not found inside the columns data')

        plots.plot_boxplot_two_variables(
            df=self.df, var1=var1, var2=var2, target=self.target)

    def plot_variables_boxplot(self):
        """
        Show boxplots for each pair of categorical and numerical variables
        If target is set, separate dataset for each target value.
        """
        df = utils.remove_var_with_one_value(self.df)

        num_vars = df.select_dtypes('number')
        cat_vars = df.select_dtypes(['object', 'category'])

        var_combi = [(v1, v2) for v1 in num_vars.columns for v2 in cat_vars.columns if (
            v1 != v2) & (v2 != self.target)]

        for num_var in num_vars.columns:
            display(Markdown(''))
            display(
                Markdown(f'Box plot for **{self.target}** & **{num_var}**'))
            self.plot_boxplot_two_variables(var1=self.target, var2=num_var)

        for num_var, cat_var in var_combi:
            display(Markdown(''))
            display(Markdown(f'Box plot for **{cat_var}** & **{num_var}**'))
            self.plot_boxplot_two_variables(var1=cat_var, var2=num_var)

    def plot_correlations(self):
        """
        Show differents correlations matrix for 3 cases :
        - numerical to numerical (using Pearson coeff)
        - categorical to categorical (using Cramers V & Chi square)
        - numerical to categorical (discrete) (using Point Biserial)
        """
        df = utils.remove_var_with_one_value(self.df)

        num_df = df.select_dtypes('number')
        cat_df = df.select_dtypes(['object', 'category'])
        num_vars = num_df.columns
        cat_vars = cat_df.columns

        ignore_cat_vars = list()
        for var in cat_vars:
            if cat_df[var].nunique() > 100:
                ignore_cat_vars.append(var)
        cat_vars = [v for v in cat_vars if v not in ignore_cat_vars]

        if len(ignore_cat_vars) > 0:
            print('Ignored categorical variables because there are more than 100 values :', ', '.join(
                ignore_cat_vars))

        pearson_corr = num_df.corr()
        display(Markdown('#### Pearson correlation matrix for numerical variables'))
        plots.plot_correlation_matrix(pearson_corr)

        var_combi = [tuple(sorted([v1, v2]))
                     for v1 in cat_vars for v2 in cat_vars if v1 != v2]
        var_combi = list(set(var_combi))

        cramers_v_corr = utils.init_corr_matrix(
            columns=cat_vars, index=cat_vars)

        for var1, var2 in var_combi:
            corr = utils.cramers_v(cat_df[var1], cat_df[var2])
            cramers_v_corr.loc[var1, var2] = corr
            cramers_v_corr.loc[var2, var1] = corr

        display(Markdown('#### Cramers V correlation matrix for categorical variables'))
        plots.plot_correlation_matrix(cramers_v_corr)

        data_encoded = utils.encode_categorical_vars(df)
        # pearson_corr = data_encoded.corr()
        # display(Markdown('#### Pearson correlation matrix for categorical variables'))
        # plots.plot_correlation_matrix(pearson_corr)

        var_combi = [(v1, v2)
                     for v1 in cat_vars for v2 in num_vars if v1 != v2]

        pbs_corr = utils.init_corr_matrix(
            columns=num_vars, index=cat_vars, fill_diag=0.)

        for cat_var, num_var in var_combi:
            corr, p_value = ss.pointbiserialr(
                data_encoded[cat_var], data_encoded[num_var])
            pbs_corr.loc[cat_var, num_var] = corr

        display(Markdown(
            '#### Point Biserial correlation matrix for numerical & categorical variables'))
        plots.plot_correlation_matrix(pbs_corr)

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
        if attr != None:
            plots.plot_dataset_metrics(
                    protected_attr=self.protected_attributes[attr], target_value=target_value)
        else:
            for attr in self.protected_attributes:
                display(Markdown(''))
                plots.plot_dataset_metrics(
                        protected_attr=self.protected_attributes[attr], target_value=target_value)