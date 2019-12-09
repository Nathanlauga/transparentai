"""
Script that update questions into the database.
CSV source : data/questions.csv
"""

from tinydb import TinyDB, Query
import os
import sys

import data

def load_db():
    parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    return TinyDB(parent_path+'/transparentai/app/db/db.json')

def main():
    db = load_db()
    # db.purge_tables()
    db.purge_table('questions')
    questions = db.table('questions')

    for idx, row in data.questions.iterrows():
        questions.insert({
            'section': row['section'], 
            'question_type': row['question_type'], 
            'question':row['questions']
            })

    db.close()


if __name__ == '__main__':
    main()
