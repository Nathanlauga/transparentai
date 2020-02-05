import pandas as pd
import os


def load_adult():
    """
    Load Adult dataset.
    Source : https://archive.ics.uci.edu/ml/datasets/Adult
    """

    names = [
        'age',
        'workclass',
        'fnlwgt',
        'education',
        'education-num',
        'marital-status',
        'occupation',
        'relationship',
        'race',
        'gender',
        'capital-gain',
        'capital-loss',
        'hours-per-week',
        'native-country',
        'income',
    ]
    dtypes = {
        'workclass': 'category',
        'education': 'category',
        'marital-status': 'category',
        'occupation': 'category',
        'relationship': 'category',
        'race': 'category',
        'gender': 'category',
        'native-country': 'category',
        'income': 'category'
    }
    adult = pd.read_csv(
        os.path.join(os.path.dirname(__file__), 'data', 'adult.csv'),
        names=names,
        header=None,
        dtype=dtypes
    )
    return adult


def load_iris():
    """
    Load Iris dataset.
    Source : http://archive.ics.uci.edu/ml/datasets/Iris/
    """

    names = [
        'sepal length (cm)',
        'sepal width (cm)',
        'petal length (cm)',
        'petal width (cm)',
        'iris plant'
    ]
    dtypes = {
        'iris plant': 'category'
    }
    iris = pd.read_csv(
        os.path.join(os.path.dirname(__file__), 'data', 'iris.csv'),
        names=names,
        header=None,
        dtype=dtypes
    )
    return iris

def load_boston():
    """
    Load boston dataset 
    Source : https://archive.ics.uci.edu/ml/machine-learning-databases/housing/ 
    """
    boston = pd.read_csv(
        os.path.join(os.path.dirname(__file__), 'data', 'boston.csv')
    )
    return boston