from tinydb import TinyDB, Query
import os
import sys
import collections
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils

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
        index = [q.doc_id for q in questions]
        return pd.DataFrame(questions, index=index)

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
        Initialize an ai.
        Create a new ai into the TinyDB json file with state set to init
        """
        ai_db = self.db.table('ai')
        ai_db.insert({'state': 'init'})

    def get_ai_in_creation(self):
        """
        Get the ai with init state created by `init_ai()` function

        Returns
        -------
        list:
            list of ai with init state (normally it supposed to have only one element)
        """
        ai_db = self.db.table('ai')
        AI = Query()
        return ai_db.search(AI.state == 'init')

    def is_ai_in_creation(self):
        """
        Test if there is an ai with init state so in creation

        Returns
        -------
        bool:
            True if there is an ai with init state else False
        """
        ai_in_creation = self.get_ai_in_creation()
        return len(ai_in_creation) > 0

    def ai_exists(self, ai_id: int):
        """
        Given an id return if this ai exists into the TinyDB

        Parameters
        ----------
        ai_id: int
            ai id to look into TinyDB
        
        Returns
        -------
        bool:
            True if the ai exists else False
        """
        ai_db = self.db.table('ai')
        return ai_db.contains(doc_ids=[ai_id])

    def get_answers_ai(self, ai_id: int):
        """
        Given an ai id find the answer if stored into the TinyDB

        Parameters
        ----------
        ai_id: int
            ai id to look into TinyDB
        
        Returns
        -------
        dict:
            dictionnary of answers for the define AI section 
        """
        ai_db = self.db.table('ai')
        ai = ai_db.get(doc_id=ai_id)

        if type(ai) == type(None):
            return None
        if 'answers' not in ai:
            return None

        return ai['answers']


    def add_answer_ai(self, ai_id: int, answers: dict):
        """
        Given an ai id and answers add them to the TinyDB
        answers are stored this way {"index answer" : "answer wrote by user"}

        Parameters
        ----------
        ai_id: int
            ai id to look into TinyDB
        answers: dict
            answer extract from the form into define AI section and formated
        """
        ai_db = self.db.table('ai')
        answers_dict = dict()
        for answer_id in answers:
            answers_dict[answer_id] = answers[answer_id]

        ai_db.update({'answers': answers_dict}, doc_ids=[ai_id])

    def close(self):
        """
        Close TinyDB connection
        """
        self.db.close()


def format_anwser_for_db(answers: dict):
    """
    Format answer before storage into TinyDB. Mostly used for questions that have 
    multiple answers like "Who is participating during the AI creation ?"

    there are two possibles possible answer formats:
    1. For answers that contains only one input for one answer 
        {'question_key': {'num_answer': 'answer'}, ... }
    2. For answers that contains at least two input for one answer 
        {'question_key': {'num_answer': {'col1': 'answer1'}, {'col2': 'answer2'}, ... }, ... }

    Parameters
    ----------
    answers: dict
        answers extract from the form into define AI section
    Returns
    -------
    dict:
        formated answers
    """
    answers_formated = dict()

    answers = utils.remove_empty_from_dict(d=answers)
    idx = answers.keys()
    print(idx)
    idx = [i.split('-')[0] for i in idx]
    print(idx)
    
    for key in idx:
        answers_formated[key] = dict()

    for key in answers:
        split_key = key.split('-')
        question_key = split_key[0]

        if len(split_key) > 2:
            detail, num = split_key[1], split_key[2]

            if num not in answers_formated[question_key]:
                answers_formated[question_key][num] = dict()                    

            answers_formated[question_key][num][detail] = answers[key]
        elif len(split_key) == 2:
            num = split_key[1]
            answers_formated[question_key][num] = answers[key]
        else:
            answers_formated[question_key]['1'] = answers[key]
    
    for question_key in answers_formated:
        current_answer = answers_formated[question_key]
        length = len(current_answer)
        keys = list(current_answer.keys())

        for i in range(0,length):
            answers_formated[question_key][str(i+1)] = current_answer.pop(keys[i])

    return answers_formated
