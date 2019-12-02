import pandas as pd

import sys
sys.path.append("..")
import utils

def load_questions(path: str):
    """
    Load the file 'questions.csv' that contains questions for
    the user needs section when creating an AI.

    Parameters
    ----------
    path : str
        path to the directory which contains the csv file

    Returns
    -------
    pd.DataFrame
        all questions for user needs section.
    """
    if not utils.is_path_format_correct(path=path):
        path = path+'/'
    return pd.read_csv(path+'questions.csv')
