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
        parent_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), os.pardir))
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

    def get_ai(self):
        """
        Extract all ai from db.json

        Returns
        -------
        list:
            list containing all ai
        """
        ai_db = self.db.table('ai')
        return ai_db.all()

    def init_ai(self):
        """

        """
        ai_db = self.db.table('ai')
        ai_db.insert({'state': 'init'})

    def get_ai_in_creation(self):
        """

        """
        ai_db = self.db.table('ai')
        AI = Query()
        return ai_db.search(AI.state == 'init')

    def is_ai_in_creation(self):
        """

        """
        ai_in_creation = self.get_ai_in_creation()
        return len(ai_in_creation) > 0

    def ai_exists(self, ai_id: int):
        """

        """
        ai_db = self.db.table('ai')
        return ai_db.contains(doc_ids=[ai_id])

    def get_answers_ai(self, ai_id: int):
        """

        """
        ai_db = self.db.table('ai')
        ai = ai_db.get(doc_id=ai_id)

        if type(ai) == type(None):
            return None
        if 'answers' not in ai:
            return None

        return ai['answers']


    def add_answer_ai(self, ai_id, answer):
        """

        """
        ai_db = self.db.table('ai')
        answer_dict = dict()
        for answer_id in answer:
            answer_dict[answer_id] = answer[answer_id]

        ai_db.update({'answers': answer_dict}, doc_ids=[ai_id])

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
