import pandas as pd
import numpy as np
import scipy.stats as ss
from IPython.display import display, Markdown

import transparentai.datasets.datasets_plots as plots
import transparentai.utils as utils


class StructuredDataset():
    """
    Class to inspect a strutured (tabular) dataset based on a pandas DataFrame object.
    It could help you to explore your data, understand what's in it with plot functions.
    
    If target is None then display different plots without splitting it by target. 

    In case the target is a continuous numeric variable and you specify it with `target_regr`
    attribute then it convert the target column to a binary classification variable greater than
    the mean and lesser or equal.   

    It helps to have a better understanding of your data.

    Example
    -------
    For binary classification :

    >>> from transparentai.datasets import StructuredDataset, load_adult
    >>> adult = load_adult()
    >>> dataset = StructuredDataset(df=adult, target='income')

    For regression :

    >>> from transparentai.datasets import StructuredDataset, load_boston
    >>> boston = load_boston()
    >>> dataset = StructuredDataset(df=boston, target='MEDV', target_regr=True)

    With no target column :

    >>> from transparentai.datasets import StructuredDataset, load_adult
    >>> adult = load_adult()
    >>> dataset = StructuredDataset(df=adult)

    Attributes
    ----------
    df: pd.DataFrame
        Dataframe with appropriate dtypes
    target: str (optional)
        target column, if None then this StructuredDataset will not
        be abble to use bias insight
    target_mean: number (optional)
        If the target is for a regression model (continuous variable) 
        Mean of the target variable 
    orig_target_value: pd.Series (optional)
        If the target is for a regression model (continuous variable) 
        this attribute contains original target values.
    """
    orig_target_value = None
    target_mean = None

    def __init__(self, df, target=None, mean=None, target_regr=False):
        """
        Parameters
        ----------
        df: pd.DataFrame
            Dataframe with appropriate dtypes
        target: str
            target column, if None then this StructuredDataset will not
            be abble to use bias insight
        mean: number (optional)
            If the target is for a regression model (continuous variable) 
            Mean of the target variable 
        target_regr: bool
            Whether the target is for a regression model (continuous variable)
            or not.

        Raises
        ------
        TypeError:
            Must provide a pandas DataFrame representing the data (features, target)
        ValueError:
            target attribute has to be in the df columns   
        """
        if (df is None) & (type(df) is not pd.DataFrame):
            raise TypeError("Must provide a pandas DataFrame representing " +
                            "the data (features, target)")
        if (target is not None) and (target not in df.columns):
            raise ValueError('target attribute has to be in the df columns')

        df = df.copy()

        if target is not None:
            if target in df.select_dtypes('object').columns:
                df[target] = df[target].astype('category')

            elif (target in df.select_dtypes('number').columns) & (target_regr):
                self.target_mean = mean if mean is not None else np.mean(
                    df[target]).round(3)
                df, orig_val = utils.regression_to_classification(
                    df, target, self.target_mean)
                self.orig_target_value = orig_val

        self.df = df.copy()
        self.target = target

    def __str__(self):
        return str(self.df)

    def _reduce_df_nrows(self, nrows=None):
        """
        Returns a reduce version of the df attribute dataframe.

        It usefull if there is too much data.
        
        Parameters
        ----------
        nrows: int (optional)
            If None then returns the original dataset
            else returns a sample of the dataset
        Returns
        -------
        pd.DataFrame:
            Reduced df attribute dataframe
        """
        if nrows is not None:
            df = utils.reduce_df_nrows(df=self.df, nrows=nrows)
        else:
            df = self.df
        return df

    def plot_dataset_overview(self):
        """
        Displays an overview of the dataset :

        - Shape 
        - Head
        """
        display(Markdown(f'#### Dataset shape : {self.df.shape}'))
        display(Markdown('First 5 rows of the dataset : '))
        display(self.df.head())

    def plot_missing_values(self):
        """
        Displays a bar chart of missing values for columns which contains
        at least one missing value.
        """
        plots.plot_missing_values(df=self.df)

    def plot_one_numeric_variable(self, var, nrows=None):
        """
        Display multiple graphics of a numerical variable.

        Parameters
        ----------
        var: str
            Variable name of a numerical variable inside dataframe object 
        nrows: int (optional)       
            If not None reduce the data to a sample of nrows

        Raises
        ------
        TypeError:
            Must provide a valid variable name string
        ValueError:
            The variable must be in the columns of the data
        ValueError:
            The variable has to be a numerical variable inside the dataframe
        """
        if var is None:
            raise TypeError("Must provide a valid variable name string")
        if var not in self.df.columns:
            raise ValueError(
                'The variable was not found inside the columns data')
        if var not in self.df.select_dtypes('number').columns:
            raise ValueError(
                'The variable has to be a numerical variable inside the dataframe')

        df = self._reduce_df_nrows(nrows=nrows)
        
        if var == self.target:
            plots.plot_numerical_var(df=df, var=self.target) 
        else:
            plots.plot_numerical_var(df=df, var=var, target=self.target)

    def plot_one_categorical_variable(self, var, nrows=None):
        """
        Display multiple graphics of a categorical variable.

        Parameters
        ----------
        var: str
            Variable name of a categorical variable inside dataframe object   
        nrows: int (optional)       
            If not None reduce the data to a sample of nrows     

        Raises
        ------
        TypeError:
            Must provide a valid variable name string
        ValueError:
            The variable must be in the columns of the data
        ValueError:
            The variable has to be a categorical variable inside the dataframe
        """
        if var is None:
            raise TypeError("Must provide a valid variable name string")
        if var not in self.df.columns:
            raise ValueError(
                'The variable was not found inside the columns data')
        if var not in self.df.select_dtypes(['object', 'category']).columns:
            raise ValueError(
                'The variable has to be a categorical variable inside the dataframe')

        df = self._reduce_df_nrows(nrows=nrows)

        if var == self.target:
            plots.plot_categorical_var(df=df, var=self.target) 
        else:
            plots.plot_categorical_var(df=df, var=var, target=self.target)

    def plot_one_datetime_variable(self, var, nrows=None):
        """
        Display multiple graphics of a datetime variable.

        Parameters
        ----------
        var: str
            Variable name of a datetime variable inside dataframe object  
        nrows: int (optional)       
            If not None reduce the data to a sample of nrows      

        Raises
        ------
        TypeError:
            Must provide a valid variable name string
        ValueError:
            The variable must be in the columns of the data
        ValueError:
            The variable has to be a datetime variable inside the dataframe
        """
        if var is None:
            raise TypeError("Must provide a valid variable name string")
        if var not in self.df.columns:
            raise ValueError(
                'The variable was not found inside the columns data')
        if var not in self.df.select_dtypes('datetime').columns:
            raise ValueError(
                'The variable has to be a datetime variable inside the dataframe')

        df = self._reduce_df_nrows(nrows=nrows)

        plots.plot_datetime_var(df=df, var=var, target=self.target)

    def plot_variables(self, nrows=None):
        """
        Display multiple graphics of all differents variables.
        The goal of this function is to do less code and get insightfull 
        informations about the data.

        Data type handle : categorical, numerical, datetime

        Parameters
        ----------
        nrows: int (optional)       
            If not None reduce the data to a sample of nrows
        """
        cat_vars = self.df.select_dtypes(['object', 'category'])
        num_vars = self.df.select_dtypes('number')
        dat_vars = self.df.select_dtypes('datetime')

        if self.target is not None:
            df = self._reduce_df_nrows(nrows=nrows)
            display(Markdown('### Target variable'))

            if self.target in cat_vars.columns:
                plots.plot_categorical_var(df=df, var=self.target)
                cat_vars = cat_vars.drop(columns=self.target)

            elif self.target in num_vars.columns:
                plots.plot_numerical_var(df=df, var=self.target)
                num_vars = num_vars.drop(columns=self.target)

        if len(num_vars) > 0:
            display(Markdown('### Numerical variables'))
        for var in num_vars:
            display(Markdown(''))
            plots.display_meta_var(self.df, var)
            if len(self.df[var].unique()) <= 1:
                display(Markdown('Only one value.'))
                continue
            self.plot_one_numeric_variable(var=var, nrows=nrows)

        if len(cat_vars) > 0:
            display(Markdown('### Categorical variables'))
        for var in cat_vars:
            display(Markdown(''))
            plots.display_meta_var(self.df, var)
            if len(self.df[var].unique()) <= 1:
                display(Markdown('Only one value.'))
                continue
            self.plot_one_categorical_variable(var=var, nrows=nrows)

        if len(num_vars) > 0:
            display(Markdown('### Datetime variables'))
        for var in dat_vars:
            display(Markdown(''))
            plots.display_meta_var(self.df, var)
            if len(self.df[var].unique()) <= 1:
                display(Markdown('Only one value.'))
                continue
            self.plot_one_datetime_variable(var=var, nrows=nrows)

    def plot_two_numeric_variables(self, var1, var2, nrows=None):
        """
        Show two numerical variables relations with jointplot.

        Parameters
        ----------
        var1: str
            Column name that contains first numerical values
        var2: str
            Column name that contains second numerical values
        nrows: int (optional)       
            If not None reduce the data to a sample of nrows

        Raises
        ------
        TypeError:
            Must provide a valid variable name string
        ValueError:
            Both variables must be in the columns of the data
        ValueError:
            Both variables has to be numeric variables inside the dataframe
        """
        if var1 is None or var2 is None:
            raise TypeError("Must provide valid variables name string")
        if (var1 not in self.df.columns) or (var2 not in self.df.columns):
            raise ValueError(
                'At least one of the variables was not found inside the columns data')
        if (var1 not in self.df.select_dtypes('number').columns) or (var2 not in self.df.select_dtypes('number').columns):
            raise ValueError(
                'Both variables has to be a numerical variables inside the dataframe')

        df = self._reduce_df_nrows(nrows=nrows)

        plots.plot_numerical_jointplot(
            df=df, var1=var1, var2=var2, target=self.target)

    def plot_numeric_var_relation(self, nrows=None):
        """
        Show all numerical variables 2 by 2 with graphics understand their relation.
        If target is set, separate dataset for each target value.
        """
        num_vars = self.df.select_dtypes('number')
        num_vars = utils.remove_var_with_one_value(num_vars)

        cols = num_vars.columns.values
        var_combi = [tuple(sorted([v1, v2]))
                     for v1 in cols for v2 in cols if (v1 != v2) & (self.target not in [v1, v2])]
        var_combi = list(set(var_combi))

        for var1, var2 in var_combi:
            display(Markdown(''))
            display(Markdown(f'Joint plot for **{var1}** & **{var2}**'))
            self.plot_two_numeric_variables(var1=var1, var2=var2, nrows=nrows)

    def plot_one_cat_and_num_variables(self, var1, var2, nrows=None):
        """
        Show boxplots for a specific pair of categorical and numerical variables
        If target is set, separate dataset for each target value.

        Parameters
        ----------
        var1: str
            Column name that contains categorical values
        var2: str
            Column name that contains numerical values
        nrows: int (optional)       
            If not None reduce the data to a sample of nrows

        Raises
        ------
        TypeError:
            Must provide valid variables name string
        ValueError:
            Both variables must be in the columns of the data
        """
        if var1 is None or var2 is None:
            raise TypeError("Must provide valid variables name string")
        if (var1 not in self.df.columns) or (var2 not in self.df.columns):
            raise ValueError(
                'At least one of the variables was not found inside the columns data')

        df = self._reduce_df_nrows(nrows=nrows)
        
        if var1 == self.target:
            plots.plot_one_cat_and_num_variables(
                df=df, var1=var1, var2=var2)
        else:
            plots.plot_one_cat_and_num_variables(
                df=df, var1=var1, var2=var2, target=self.target)

    def plot_cat_and_num_variables(self, nrows=None):
        """
        Show boxplots for each pair of categorical and numerical variables
        If target is set, separate dataset for each target value.

        Parameters
        ----------
        nrows: int (optional)       
            If not None reduce the data to a sample of nrows
        """
        df = utils.remove_var_with_one_value(self.df)

        num_vars = df.select_dtypes('number')
        cat_vars = df.select_dtypes(['object', 'category'])

        var_combi = [(v1, v2) for v1 in num_vars.columns for v2 in cat_vars.columns if (
            v1 != v2) & (self.target not in [v1, v2])]

        if self.target is not None:
            for num_var in num_vars.columns:
                display(Markdown(''))
                display(
                    Markdown(f'Box plot for **{self.target}** & **{num_var}**'))
                self.plot_one_cat_and_num_variables(
                    var1=self.target, var2=num_var, nrows=nrows)

        for num_var, cat_var in var_combi:
            display(Markdown(''))
            display(Markdown(f'Box plot for **{cat_var}** & **{num_var}**'))
            self.plot_one_cat_and_num_variables(
                var1=cat_var, var2=num_var, nrows=nrows)

    def plot_correlations(self, nrows=None):
        """
        Show differents correlations matrix for 3 cases :
        
        - numerical to numerical (using Pearson coeff)
        - categorical to categorical (using Cramers V & Chi square)
        - numerical to categorical (discrete) (using Point Biserial)

        Parameters
        ----------
        nrows: int (optional)       
            If not None reduce the data to a sample of nrows
        """
        df = self._reduce_df_nrows(nrows=nrows)
        df = utils.remove_var_with_one_value(df)

        if self.orig_target_value is not None:
            df[self.target] = self.orig_target_value

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

        if len(num_df) > 0:
            pearson_corr = num_df.corr()
            display(
                Markdown('#### Pearson correlation matrix for numerical variables'))
            plots.plot_correlation_matrix(pearson_corr, fname='pearson_corr.png')

        if len(cat_vars) > 0:
            var_combi = [tuple(sorted([v1, v2]))
                         for v1 in cat_vars for v2 in cat_vars if v1 != v2]
            var_combi = list(set(var_combi))

            cramers_v_corr = utils.init_corr_matrix(
                columns=cat_vars, index=cat_vars)

            for var1, var2 in var_combi:
                corr = utils.cramers_v(cat_df[var1], cat_df[var2])
                cramers_v_corr.loc[var1, var2] = corr
                cramers_v_corr.loc[var2, var1] = corr

            display(
                Markdown('#### Cramers V correlation matrix for categorical variables'))
            plots.plot_correlation_matrix(cramers_v_corr, fname='cramers_v_corr.png')

        if (len(cat_vars) > 0) and (len(num_df) > 0):
            data_encoded, _ = utils.encode_categorical_vars(df)

            var_combi = [(v1, v2)
                         for v1 in cat_vars for v2 in num_vars if v1 != v2]

            pbs_corr = utils.init_corr_matrix(
                columns=num_vars, index=cat_vars, fill_diag=0.)

            for cat_var, num_var in var_combi:
                tmp_df = data_encoded[[cat_var, num_var]].dropna()
                if len(tmp_df) == 0:
                    continue
                corr, p_value = ss.pointbiserialr(
                    tmp_df[cat_var], tmp_df[num_var])
                pbs_corr.loc[cat_var, num_var] = corr

            display(Markdown(
                '#### Point Biserial correlation matrix for numerical & categorical variables'))
            plots.plot_correlation_matrix(pbs_corr, fname='pointbiserialr_corr.png')
