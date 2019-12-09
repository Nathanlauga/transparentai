from tinydb import TinyDB, Query
import os
import sys
import pandas as pd

class DB():
    """
    Class that handle db connection using TinyDB package
    all db is located into db.json file that way it can be easily exported.

    Attributes
    ----------
    db : tinydb.TinyDB
        database from db.json file

    Methods
    -------
    get_questions()
        Extract all questions from db.json
    close()
        Close TinyDB connection
    """

    def __init__(self):
        parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        self.db = TinyDB(parent_path+'/db/db.json')

    
    def get_questions(self):
        """
        Extract all questions from db.json

        Returns
        -------
        pandas.DataFrame:
            dataframe containing all questions for define ai section
        """
        questions_db = self.db.table('questions')
        questions = questions_db.all()
        return pd.DataFrame(questions)

    def close(self):
        """
        Close TinyDB connection
        """
        self.db.close()


def load_questions_from_db():
    """
    Load questions from db.json

    Returns
    -------
    pandas.DataFrame:
        dataframe containing all questions for define ai section
    """
    db = DB()
    questions = db.get_questions()
    db.close()

    return questions