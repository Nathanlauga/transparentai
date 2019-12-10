import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..')+'/scripts'))
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..')+'/transparentai/app'))

import db
import data


class TestData(unittest.TestCase):

    def test_load_questions_from_file(self):
        self.assertEqual(data.questions.columns.values[0], 'section')

    def test_load_questions_from_db(self):
        self.assertEqual(db.questions.columns.values[0], 'section')

    def test_get_ai_in_creation(self):
        # TODO : test ai creation (init)
        self.assertEqual(1,1)


if __name__ == '__main__':
    unittest.main()
