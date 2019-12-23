"""
Script that update questions into the database.
CSV source : data/questions.csv
"""

from tinydb import TinyDB, Query
import os
import sys

import data


def load_db():
    parent_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), os.pardir))
    return TinyDB(parent_path+'/transparentai/app/db/db.json')


def load_db_test():
    return TinyDB('scripts/test.json')


def main():
    db = load_db()
    # db = load_db_test()
    # db.purge_tables()
    db.purge_table('questions')
    questions = db.table('questions')

    for idx, row in data.questions.iterrows():
        question = {
            'section': row['section'],
            'question_type': row['question_type'],
            'question': row['questions']
        }
        questions.insert(question)

    db.close()


if __name__ == '__main__':
    main()
