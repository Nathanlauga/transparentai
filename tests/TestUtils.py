import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')+'/transparentai/app'))
import utils

class TestData(unittest.TestCase):

    def test_is_path_format_correct(self):
        self.assertFalse(utils.is_path_format_correct('path/without'))
        self.assertTrue(utils.is_path_format_correct('path/with/'))
        

if __name__ == '__main__':
    unittest.main()