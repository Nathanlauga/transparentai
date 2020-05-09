import warnings
import scipy.stats as ss
import gc

import pandas as pd
import numpy as np

from transparentai import utils


def cramers_v(x, y):
    """Returns the Cramer V value of two categorical variables using
    chi square. This correlation metric is between 0 and 1.

    Code source found in this article : 
    https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9

    Parameters
    ----------
    x: array like
        first categorical variable
    y: array like
        second categorical variable

    Returns
    -------
    float:
        Cramer V value
    """
    confusion_matrix = pd.crosstab(x, y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    if min((kcorr-1), (rcorr-1)) == 0:
        return 0
    return np.sqrt(phi2corr/min((kcorr-1), (rcorr-1)))


def merge_corr_df(df_list):
    """Merges correlation matrix from compute_correlation() function
    to one. Needs 3 dataframe : pearson_corr, cramers_v_corr and pbs_corr.

    This matrix has a default : the cramers_v_corr is scale from 0 to 1, but 
    the others are from to -1 to 1. Be sure to understand this.

    Parameters
    ----------
    df_list: list
        List of correlation matrices

    Returns
    -------
    pd.DataFrame:
        Merged dataframe of correlation matrices
    """
    pearson_corr = df_list[0]
    cramers_v_corr = df_list[1]
    pbs_corr = df_list[2]

    cat_feats = pbs_corr.index.values.tolist()
    num_feats = pbs_corr.columns.values.tolist()

    feats = cat_feats + num_feats

    corr_df = utils.init_corr_matrix(feats, feats)

    corr_df.loc[num_feats, num_feats] = pearson_corr.loc[num_feats, num_feats]
    corr_df.loc[cat_feats, cat_feats] = cramers_v_corr.loc[cat_feats, cat_feats]

    for cat_feat in cat_feats:
        for num_feat in num_feats:
            corr_df.loc[cat_feat, num_feat] = pbs_corr.loc[cat_feat, num_feat]
            corr_df.loc[num_feat, cat_feat] = pbs_corr.loc[cat_feat, num_feat]

    return corr_df


def compute_cramers_v_corr(df):
    """Computes Cramers V correlation for a dataframe.

    `Cramers V Wikipedia definition`_ :

    In statistics, Cramér's V (sometimes referred to as Cramér's phi and denoted as φc) 
    is a measure of association between two nominal variables, giving a value between 0 and +1 (inclusive). 
    It is based on Pearson's chi-squared statistic and was published by Harald Cramér in 1946.

    .. _Cramers V Wikipedia definition: https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V

    Parameters
    ----------
    df: pd.DataFrame
        pandas Dataframe with values to compute 
        Cramers V correlation

    Returns
    -------
    pd.DataFrame:
        Correlation matrix computed for Cramers V coeff

    Raises
    ------
    TypeError:
        Must provide a pandas DataFrame representing the data
    """
    if type(df) is not pd.DataFrame:
        raise TypeError(
            "Must provide a pandas DataFrame representing the data")

    cat_feats = df.columns.values.tolist()

    var_combi = [tuple(sorted([v1, v2]))
                 for v1 in cat_feats for v2 in cat_feats if v1 != v2]
    var_combi = list(set(var_combi))

    cramers_v_corr = utils.init_corr_matrix(
        columns=cat_feats, index=cat_feats)

    for var1, var2 in var_combi:
        corr = cramers_v(df[var1], df[var2])
        cramers_v_corr.loc[var1, var2] = corr
        cramers_v_corr.loc[var2, var1] = corr

    return cramers_v_corr


def compute_pointbiserialr_corr(df, cat_feats=None, num_feats=None):
    """Computes Point Biserial correlation for a dataframe.

    `Point Biserial Wikipedia definition`_ :

    The point biserial correlation coefficient (rpb) is a correlation coefficient used when one variable (e.g. Y) 
    is dichotomous; Y can either be "naturally" dichotomous, like whether a coin lands heads or tails, 
    or an artificially dichotomized variable. In most situations it is not advisable to dichotomize variables
    artificially[citation needed]. When a new variable is artificially dichotomized the new dichotomous variable may 
    be conceptualized as having an underlying continuity. If this is the case, a biserial correlation
    would be the more appropriate calculation. 

    .. _Point Biserial Wikipedia definition: https://en.wikipedia.org/wiki/Point-biserial_correlation_coefficient

    Parameters
    ----------
    df: pd.DataFrame
        pandas Dataframe with values to compute 
        Point Biserial correlation

    Returns
    -------
    pd.DataFrame:
        Correlation matrix computed for Point Biserial coeff

    Raises
    ------
    TypeError:
        Must provide a pandas DataFrame representing the data
    ValueError:
        cat_feats and num_feats must be set or be both None
    TypeError:
        cat_feats must be a list
    TypeError:
        num_feats must be a list
    """
    if type(df) is not pd.DataFrame:
        raise TypeError(
            "Must provide a pandas DataFrame representing the data")

    if ((cat_feats is not None) & (num_feats is None)) | (
            (cat_feats is None) & (num_feats is not None)):
        raise ValueError('cat_feats and num_feats must be set or be both None')

    if type(cat_feats) != list:
        TypeError('cat_feats must be a list')
    if type(num_feats) != list:
        TypeError('num_feats must be a list')

    if (cat_feats is None) & (num_feats is None):
        num_feats = df.select_dtypes('number').columns.values.tolist()
        cat_feats = [c for c in df.columns if c not in num_feats]

    data_encoded, _ = utils.encode_categorical_vars(df)
    var_combi = [(v1, v2)
                 for v1 in cat_feats for v2 in num_feats if v1 != v2]

    pbs_corr = utils.init_corr_matrix(
        columns=num_feats, index=cat_feats, fill_diag=0.)

    for cat_feat, num_feat in var_combi:
        tmp_df = data_encoded[[cat_feat, num_feat]].dropna()
        if len(tmp_df) == 0:
            continue
        corr, p_value = ss.pointbiserialr(
            tmp_df[cat_feat], tmp_df[num_feat]
        )
        pbs_corr.loc[cat_feat, num_feat] = corr

    return pbs_corr


def compute_correlation(df, nrows=None, max_cat_val=100):
    """Computes differents correlations matrix for 
    three cases and merge them:

    - numerical to numerical (using Pearson coeff)
    - categorical to categorical (using Cramers V & Chi square)
    - numerical to categorical (discrete) (using Point Biserial)

    .. raw:: html
        
        <b>/!\ ==== Caution ==== /!\\</b>

    This matrix has a default : the cramers_v_corr is scale from 0 to 1, but 
    the others are from to -1 to 1. Be sure to understand this.

    `Pearson coeff Wikipedia definition`_ :

    In statistics, the Pearson correlation coefficient, also referred to as Pearson's r, 
    the Pearson product-moment correlation coefficient (PPMCC) or the bivariate correlation,
    is a statistic that measures linear correlation between two variables X and Y. 
    It has a value between +1 and −1, where 1 is total positive linear correlation, 
    0 is no linear correlation, and −1 is total negative linear correlation 
    (that the value lies between -1 and 1 is a consequence of the Cauchy–Schwarz inequality). 
    It is widely used in the sciences. 

    `Cramers V Wikipedia definition`_ :

    In statistics, Cramér's V (sometimes referred to as Cramér's phi and denoted as φc) 
    is a measure of association between two nominal variables, giving a value between 0 and +1 (inclusive). 
    It is based on Pearson's chi-squared statistic and was published by Harald Cramér in 1946.

    `Point Biserial Wikipedia definition`_ :

    The point biserial correlation coefficient (rpb) is a correlation coefficient used when one variable (e.g. Y) 
    is dichotomous; Y can either be "naturally" dichotomous, like whether a coin lands heads or tails, 
    or an artificially dichotomized variable. In most situations it is not advisable to dichotomize variables
    artificially[citation needed]. When a new variable is artificially dichotomized the new dichotomous variable may 
    be conceptualized as having an underlying continuity. If this is the case, a biserial correlation
    would be the more appropriate calculation. 

    .. _Pearson coeff Wikipedia definition: https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
    .. _Cramers V Wikipedia definition: https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V
    .. _Point Biserial Wikipedia definition: https://en.wikipedia.org/wiki/Point-biserial_correlation_coefficient

    Parameters
    ----------
    df: pd.DataFrame
        pandas Dataframe with values to compute correlation
    nrows: None or int or float (default None)
        If not None reduce the data to a sample of nrows if int
        else if float reduce to len(df) * nrows
    max_cat_val: int or None (default 100)
        Number max of unique values in a categorical feature
        if there are more distinct values than this number
        then the feature is ignored

    Returns
    -------
    pd.DataFrame:
        Correlation matrix computed with Pearson coeff for numerical features
        to numerical features, Cramers V for categorical features to categorical features and
        Point Biserial for categorical features to numerical features

    Raises
    ------
    TypeError:
        Must provide a pandas DataFrame representing the data
    """
    if type(df) is not pd.DataFrame:
        raise TypeError(
            "Must provide a pandas DataFrame representing the data")

    df = df.copy()

    if nrows is not None:
        if nrows < 1.:
            nrows = int(len(df)*nrows)
        elif nrows > len(df):
            nrows = len(df)
        np.random.seed(42)
        df = df.sample(nrows)

    num_feats = df.select_dtypes('number').columns.values.tolist()
    cat_feats = [c for c in df.columns if c not in num_feats]

    if max_cat_val is not None:
        ignore_cat_feats = list()
        for feat in cat_feats:
            if df[feat].nunique() > max_cat_val:
                ignore_cat_feats.append(feat)
                warnings.warn('%s feature ignored because there are more than %i unique values' % (
                    feat, max_cat_val))
        cat_feats = [v for v in cat_feats if v not in ignore_cat_feats]

    # Pearson's Correlation for numerical var
    if len(num_feats) > 0:
        pearson_corr = df[num_feats].corr()
        gc.collect()

    # Cramer's V Correlation for categorical var
    if len(cat_feats) > 0:
        cramers_v_corr = compute_cramers_v_corr(df[cat_feats])
        gc.collect()

    # Point Biserial Correlation for categorical and numerical var
    if (len(num_feats) > 0) & (len(cat_feats) > 0):
        pbs_corr = compute_pointbiserialr_corr(df, cat_feats, num_feats)
        gc.collect()

    if len(cat_feats) == 0:
        return pearson_corr

    elif len(cat_feats) == 0:
        return cramers_v_corr

    return merge_corr_df([pearson_corr, cramers_v_corr, pbs_corr])
